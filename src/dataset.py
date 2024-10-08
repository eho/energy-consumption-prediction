import pickle  # load/save model as a pickle file
from pathlib import Path
from typing import Any, Dict, List

import lightning as L
import pandas as pd  # data manipulation and analysis
import torch  # pytorch library for tensor operations
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = ".."  # the document root is one level up, that contains all code structure
DATA = Path(ROOT) / "data"  # the directory contains all data files
RAW_DATA = DATA / "raw"

# processed data directory can be used, such that preprocessing steps is not
# required to run again-and-again each time on kernel restart
PROCESSED_DATA = DATA / "processed"


def to_seconds(time_obj):
    """
    Convert a time of day object into a number representing the number of seconds since the start of the day.

    Args:
        time_obj (datetime.time): Time of day object.

    Returns:
        float: Number of seconds since the start of the day.
    """
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second


class EnergyConsumptionDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_file: str,
        feature_columns: List[str],
        sequence_length: int = 24 * (60 // 5),
        mins_per_time_step: int = 5,
        batch_size: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_file = data_file
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.mins_per_time_step = mins_per_time_step
        self.batch_size = batch_size
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler(feature_range=(0, 1))

        # Load and preprocess data
        self.column_transformer = ColumnTransformer(
            transformers=[
                ("standard", self.standard_scaler, ["total_home_usage"]),
                (
                    "minmax",
                    self.minmax_scaler,
                    [col for col in feature_columns if col != "total_home_usage"],
                ),
            ]
        )
        self.column_transformer.set_output(transform="pandas")
        self.data = None

    # Create sequences for LSTM
    def create_sequences(self, dataframe: pd.DataFrame, sequence_length):
        dataset = dataframe.values

        features = []
        targets = []
        for i in range(len(dataset) - sequence_length):
            features.append(dataset[i : i + sequence_length])
            targets.append(dataset[i + sequence_length][-1])
        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            targets, dtype=torch.float32
        )  # features, targets

    def prepare_data(self):
        # Load and preprocess data
        df = pd.read_csv(self.data_file, parse_dates=True)

        # Convert the "timestamp" column to datetime type
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(
            "Australia/Sydney"
        )
        df["time_of_day"] = df["timestamp"].dt.time.apply(to_seconds)
        df["day_of_week"] = df["timestamp"].dt.day_of_week
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        df["day"] = df["timestamp"].dt.day
        df["total_home_usage"] = df["total_home_usage"].interpolate(method="linear")
        df["total_battery_charge"] = df["total_battery_charge"].interpolate(
            method="linear"
        )
        df["total_battery_discharge"] = df["total_battery_discharge"].interpolate(
            method="linear"
        )
        df["total_grid_energy_exported"] = df["total_grid_energy_exported"].interpolate(
            method="linear"
        )
        df["total_battery_charge"] = df["total_battery_charge"].interpolate(
            method="linear"
        )
        df["total_solar_generation"] = df["total_solar_generation"].interpolate(
            method="linear"
        )
        df.drop(
            columns=[
                "total_battery_charge",
                "total_battery_charge",
                "total_grid_energy_exported",
            ],
            inplace=True,
        )
        assert (df.isnull().sum() == 0).all(), "Validate no missing values"

        data_to_resample = df[["timestamp", "total_home_usage"]]
        data_half_hourly = (
            data_to_resample.resample("5min", on="timestamp").sum().reset_index()
        )
        data_half_hourly["timestamp"] = pd.to_datetime(
            data_half_hourly["timestamp"], utc=True
        ).dt.tz_convert("Australia/Sydney")
        data_half_hourly["time_of_day"] = data_half_hourly["timestamp"].dt.time.apply(
            to_seconds
        )
        data_half_hourly["day_of_week"] = data_half_hourly["timestamp"].dt.day_of_week
        data_half_hourly["year"] = data_half_hourly["timestamp"].dt.year
        data_half_hourly["month"] = data_half_hourly["timestamp"].dt.month
        data_half_hourly["day"] = data_half_hourly["timestamp"].dt.day
        self.data = data_half_hourly
        self.last_sequence_data = self.data[-self.sequence_length :][
            "total_home_usage"
        ].tolist()

        # When using a column transformer, the column order is not preserved. To preseve the column order,
        # we set the out to be a dataframe so we can "fix" the order before returning it as an array.
        # See https://stackoverflow.com/a/77702955
        normalized_data = self.column_transformer.fit_transform(
            self.data[self.data["year"] == 2024][self.feature_columns]
        )
        normalized_data = normalized_data.rename(columns=lambda x: x.split("__", 1)[-1])
        normalized_data = normalized_data[self.feature_columns]

        # Create sequences for training
        features, targets = self.create_sequences(normalized_data, self.sequence_length)
        # features = features.to(device)
        # targets = targets.to(device)

        # Split data into training and validation sets
        train_size = int(0.8 * len(normalized_data))
        print(
            f"train_size: {train_size} valid_size: {len(normalized_data) - train_size}"
        )
        train_ds = TensorDataset(features[:train_size], targets[:train_size])
        valid_ds = TensorDataset(features[train_size:], targets[train_size:])

        self.trainloader = DataLoader(
            train_ds, batch_size=self.batch_size, num_workers=4, shuffle=True
        )
        self.validloader = DataLoader(
            valid_ds, batch_size=self.batch_size, num_workers=4, shuffle=False
        )

    def train_dataloader(self):
        return self.trainloader

    def val_dataloader(self):
        return self.validloader

    def target_scaler(self):
        return self.column_transformer.named_transformers_["standard"]

    def state_dict(self) -> Dict[str, Any]:
        # Create a custom state dict that includes both model parameters and the transformer
        state_dict = super().state_dict()

        # Save the fitted column transformer. It will be used for inference.
        column_transformer_data = pickle.dumps(self.column_transformer)
        state_dict["column_transformer"] = column_transformer_data

        # Save the last {sequence_length} rows of the data
        state_dict["last_sequence_data"] = self.last_sequence_data

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Load the transformer from the state dict
        column_transformer_data = state_dict.pop("column_transformer", None)
        if column_transformer_data:
            self.column_transformer = pickle.loads(column_transformer_data)

        self.last_sequence_data = state_dict.pop("last_sequence_data", None)

        return super().load_state_dict(state_dict)
