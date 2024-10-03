import json
import csv
import os
import click
import pathlib


# Define the input directory and output file
@click.command()
@click.option(
    "--input-dir", help="Input directory containing JSON files", default="data/raw"
)
@click.option(
    "--output-file", help="Output CSV file", default="data/time_series_data.csv"
)
def extract_powerwall_data(input_dir, output_file):
    FIELD_NAMES = [
        "date",
        "timestamp",
        "solar_energy_exported",
        "generator_energy_exported",
        "grid_energy_imported",
        "grid_services_energy_imported",
        "grid_services_energy_exported",
        "grid_energy_exported_from_solar",
        "grid_energy_exported_from_generator",
        "grid_energy_exported_from_battery",
        "battery_energy_exported",
        "battery_energy_imported_from_grid",
        "battery_energy_imported_from_solar",
        "battery_energy_imported_from_generator",
        "consumer_energy_imported_from_grid",
        "consumer_energy_imported_from_solar",
        "consumer_energy_imported_from_battery",
        "consumer_energy_imported_from_generator",
        "raw_timestamp",
        "total_battery_charge",
        "total_solar_generation",
        "total_home_usage",
        "total_battery_discharge",
        "total_grid_energy_exported",
    ]

    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w+", newline="") as csvfile:
        writer = None

        # Iterate over the files in the input directory in ascending order
        for filename in sorted(os.listdir(input_dir)):
            # Check if the file is a JSON file
            if filename.endswith(".json"):
                # Open the JSON file and load the data
                with open(os.path.join(input_dir, filename), "r") as f:
                    data = json.load(f)

                # Extract the time series data from the JSON file
                for entry in data["response"]["time_series"]:
                    # Create a new dictionary with the date and time series data
                    entry["date"] = filename.split(".")[0]

                    # Write the entry to the CSV file
                    if writer is None:
                        writer = csv.DictWriter(csvfile, fieldnames=FIELD_NAMES)
                        writer.writeheader()
                    writer.writerow(entry)


if __name__ == "__main__":
    extract_powerwall_data()
