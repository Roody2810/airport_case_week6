from pathlib import Path

import pandas as pd

from src.data_cleaner import AirportDataCleaner
from src.data_merger import DataMerger
import streamlit as st



DATA_DIR = Path(__file__).resolve().parent / "data"


def load_datasets() -> dict[str, pd.DataFrame]:
    """Return all cleaned and merged datasets for dashboard development."""

    schedule_cleaner = AirportDataCleaner.from_schedule_csv(
        DATA_DIR / "schedule_airport.csv"
    )
    schedule = schedule_cleaner.clean_schedule()

    airports_cleaner = AirportDataCleaner.from_airports_csv(
        DATA_DIR / "airports-extended.csv"
    )
    airports = airports_cleaner.clean_airports()

    weather_cleaner = AirportDataCleaner.from_weather_csv(
        DATA_DIR / "weather_zurich_2019-2020.csv"
    )
    weather = weather_cleaner.clean_weather()

    merger = DataMerger(schedule)
    merger.create_delay_columns()
    merger.merge_airports(airports)
    merger.merge_weather(weather)
    merged = merger.get_df()

    model_data = DataMerger(merged).prepare_model_data()

    return {
        "schedule": schedule,
        "airports": airports,
        "weather": weather,
        "merged": merged,
        "model_data": model_data,
    }


datasets = load_datasets()

schedule = datasets["schedule"]
airports = datasets["airports"]
weather = datasets["weather"]
merged = datasets["merged"]
model_data = datasets["model_data"]
