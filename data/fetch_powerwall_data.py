import click
import requests
import datetime
import os
import urllib.parse
import pytz

SYDNEY_TZ = pytz.timezone("Australia/Sydney")
TESLA_ENDPOINT = "https://fleet-api.prd.na.vn.cloud.tesla.com"
ENERGY_SITE_ID = "220796430229"


def fetch_and_save_data(access_token, date, output_dir):
    """
    Fetch data from API for a given date and save to a file.

    :param date: Date to fetch data for (YYYY-MM-DD format)
    :param api_url: URL of the API to call
    :param output_dir: Directory to save the file to
    """
    start_date = (
        datetime.datetime.combine(date, datetime.time.min)
        .astimezone(SYDNEY_TZ)
        .isoformat()
    )
    end_date = (
        datetime.datetime.combine(date, datetime.time.max)
        .astimezone(SYDNEY_TZ)
        .isoformat()
    )
    params = {
        "kind": "energy",
        "period": "day",
        "start_date": start_date,
        "end_date": end_date,
        "time_zone": "Australia/Sydney",
        "interval": "15m",
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    try:
        url = f"{TESLA_ENDPOINT}/api/1/energy_sites/{ENERGY_SITE_ID}/calendar_history"
        response = requests.get(
            url,
            params=urllib.parse.urlencode(params),
            headers=headers,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {date}: {e.response.text}")
        return

    file_name = f"{date}.json"
    file_path = f"{output_dir}/{file_name}"

    with open(file_path, "w") as f:
        f.write(response.text)


@click.command()
@click.option(
    "--start-date",
    help="Start date (YYYY-MM-DD format)",
    type=click.DateTime(formats=["%Y-%m-%d"]),
)
@click.option("--num-days", type=int, help="Number of days to fetch data for")
@click.option("--access-token", help="Access token to authenticate with the API")
@click.option("--output-dir", help="Directory to save files to", default="raw")
def fetch_data(start_date, num_days, access_token, output_dir):
    """
    Fetch data from API for a range of dates and save to files.

    :param start_date: Start date (YYYY-MM-DD format)
    :param num_days: Number of days to fetch data for
    :param api_url: URL of the API to call
    :param output_dir: Directory to save files to
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_days):
        date = start_date + datetime.timedelta(days=i)
        fetch_and_save_data(access_token, date.date(), output_dir)


if __name__ == "__main__":
    fetch_data()
