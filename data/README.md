
## Fetch Tesla Powerwall historical usage data

This requires Tesla API access token.

```
cd data
python ./fetch_powerwall_data.py --access-token $TESLA_ACCESS_TOKEN --num-days 4 --start-date "2024-09-27"
```

## Convert and combine all Powerwall raw data into a time-series CSV file

```
cd data
python ./extract_powerwall_data.py --input-dir ./raw --output-file ./processed/time_series_data.csv
```