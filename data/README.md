
## Fetch Tesla Powerwall historical usage data

This requires Tesla API access token.

```
cd data
python ./fetch_powerwall_data.py --access-token $TESLA_ACCESS_TOKEN --num-days 1 --start-date "2024-09-26"
```

## Convert and combine all Powerwall raw data into a time-series CSV file

```
python data/extract_powerwall_data.py --input-dir data/raw --output-file data/time_series_data.csv
```