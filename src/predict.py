from datetime import date, datetime, timedelta
from typing import Any, Dict, List

import click
import matplotlib.pyplot as plt
import pandas as pd  # data manipulation and analysis
import torch  # pytorch library for tensor operations
import torch.utils
import torch.utils.data
from IPython.display import display
from sklearn.compose import ColumnTransformer

from .dataset import EnergyConsumptionDataModule
from .model import LSTMModel
from .plot import plot_consumption


class Prediction:
    def __init__(
        self,
        model: LSTMModel,
        last_sequence_data: List[float],
        sequence_length: int,
        mins_per_time_step: int,
        feature_columns: List[str],
        column_transformer: ColumnTransformer,
    ):
        self.model = model
        self.last_sequence_data = last_sequence_data
        self.sequence_length = sequence_length
        self.mins_per_time_step = mins_per_time_step
        self.feature_columns = feature_columns
        self.column_transformer = column_transformer
        self.target_scaler = column_transformer.named_transformers_["standard"]
        assert len(self.last_sequence_data) == self.sequence_length

    def generate_time_steps(self, start_datetime: datetime, num_steps: int):
        """
        Generates the last `num_steps` time steps from the given datetime.
        We use last_sequence_data as as template for "existing value". We will juse use it to predict the next value.
        We need to modify the month, time_of_day, and day_of_week to match the start_date.

        Args:
            start_datetime (datetime): The reference datetime.
            num_steps (int): Number of time steps to generate (default is 24).

        Returns:
            pd.DataFrame: DataFrame containing the month, time_of_day (seconds from start of the day),
                        day_of_week (0-6), and consumption (set to 0).
        """
        time_steps = []

        for i in range(num_steps):
            step_datetime = start_datetime - timedelta(
                minutes=i * self.mins_per_time_step
            )
            month = step_datetime.month
            time_of_day = (
                step_datetime.hour * 3600
                + step_datetime.minute * 60
                + step_datetime.second
            )  # seconds from start of day
            day_of_week = step_datetime.weekday()  # Monday is 0, Sunday is 6

            time_steps.append(
                {
                    "timestamp": step_datetime,
                    "month": month,
                    "time_of_day": time_of_day,
                    "day_of_week": day_of_week,
                    "total_home_usage": self.last_sequence_data[i],
                }
            )

        # Reverse the order of the list since we generate the past timesteps from the reference datetime
        time_steps.reverse()

        return pd.DataFrame(time_steps)

    def predict(self, start_date: date, predict_steps: int = 10):
        # Initialize the result list
        result = []
        start_datetime = datetime.combine(start_date, datetime.min.time())

        # Generate the initial input data (past {sequence_length} time steps)
        input_data_df = self.generate_time_steps(start_datetime, self.sequence_length)

        # Transform/scale the input features
        input_features = self.column_transformer.transform(
            input_data_df[self.feature_columns]
        )
        input_features = input_features.rename(columns=lambda x: x.split("__", 1)[-1])

        # This makes sure the columns are in the right order.
        input_features = input_features[self.feature_columns]

        # Convert the input features to a tensor with shape (1, sequence_length, len(feature_columns))
        input_tensor = torch.tensor(
            input_features.values, dtype=torch.float32, device=self.model.device
        ).unsqueeze(0)
        assert input_tensor.shape == (
            1,
            self.sequence_length,
            len(self.feature_columns),
        )

        self.model.eval()

        # Predict for the next `predict_steps` time steps
        for i in range(predict_steps):
            # Make a prediction
            with torch.no_grad():
                prediction_from_model = self.model(input_tensor)
                transformed_prediction = self.target_scaler.inverse_transform(
                    prediction_from_model.cpu()
                )

                # Extract the predicted consumption value
                predicted_consumption = transformed_prediction[0, -1]

            # Get the next timestamp
            next_datetime = start_datetime + timedelta(
                minutes=self.mins_per_time_step * (i + 1)
            )

            # Get the month, time_of_day, and day_of_week for the next timestep
            month = next_datetime.month
            time_of_day = (
                next_datetime.hour * 3600
                + next_datetime.minute * 60
                + next_datetime.second
            )
            day_of_week = next_datetime.weekday()

            # Store the result
            result.append(
                {
                    "timestamp": next_datetime,
                    "month": month,
                    "day_of_week": day_of_week,
                    "time_of_day": time_of_day,
                    "total_home_usage": predicted_consumption,
                }
            )

            # Update input_tensor for the next prediction by appending new feature data
            next_input_features = self.column_transformer.transform(
                pd.DataFrame([result[-1]])
            )
            next_input_features = next_input_features.rename(
                columns=lambda x: x.split("__", 1)[-1]
            )
            next_input_features = next_input_features[self.feature_columns]
            input_tensor = torch.cat(
                (
                    input_tensor[:, 1:, :],
                    torch.tensor(
                        next_input_features.values,
                        dtype=torch.float32,
                        device=self.model.device,
                    ).unsqueeze(0),
                ),
                dim=1,
            )

        # Convert the result into a DataFrame
        result_df = pd.DataFrame(result)

        return result_df


def predict(model_checkpoint_path: str, start_date: date, predict_days: int = 1):
    print(f"Loading model from {model_checkpoint_path}")

    saved_model = LSTMModel.load_from_checkpoint(model_checkpoint_path)
    saved_data_module = EnergyConsumptionDataModule.load_from_checkpoint(
        model_checkpoint_path
    )
    print(saved_data_module.feature_columns)
    prediction = Prediction(
        saved_model,
        saved_data_module.last_sequence_data,
        saved_data_module.sequence_length,
        saved_data_module.mins_per_time_step,
        saved_data_module.feature_columns,
        saved_data_module.column_transformer,
    )

    return prediction.predict(
        start_date, predict_steps=saved_data_module.sequence_length * predict_days
    )


@click.command()
@click.option(
    "--model-checkpoint-path",
    help="Path to the model checkpoint",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--start-date",
    help="Start date (YYYY-MM-DD format)",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
)
@click.option(
    "--predict-days",
    help="Number of days to predict",
    type=int,
    default=1,
    show_default=True,
)
def predict_command(model_checkpoint_path, start_date, predict_days):
    """Make predictions using a trained model."""
    prediction_df = predict(model_checkpoint_path, start_date.date(), predict_days)
    # You can add additional logic here to handle the prediction_df
    # For example, you can save it to a file or print it to the console
    print("done")
    print(prediction_df)

    facet_grid = plot_consumption(
        prediction_df, start_date.year, col_wrap=1, title_suffix="Prediction"
    )
    facet_grid.savefig("consumption_plot.png")
    print("Consumption plot saved to consumption_plot.png")


def main():
    predict_command()


if __name__ == "__main__":
    main()
