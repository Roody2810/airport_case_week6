from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd


class DataMerger:
    """Prepare schedule data and combine it with airport and weather data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def get_df(self) -> pd.DataFrame:
        return self.df

    def create_delay_columns(
        self,
        *,
        date_column: str = "std",
        scheduled_time_column: str = "sta_std_ltc",
        actual_time_column: str = "ata_atd_ltc",
        output_column: str = "delay_minutes",
        delayed_column: str = "is_delayed",
        delayed_after_minutes: int = 15,
    ) -> pd.DataFrame:
        """Create scheduled/actual datetimes and a delay target.

        The schedule contains a date and time in separate columns. If an
        actual time is more than twelve hours earlier than the scheduled time,
        it is treated as an after-midnight time on the following day.
        """

        required = {
            date_column,
            scheduled_time_column,
            actual_time_column,
        }
        missing = required.difference(self.df.columns)
        if missing:
            raise KeyError(
                "Missing schedule columns: " + ", ".join(sorted(missing))
            )

        date_values = pd.to_datetime(
            self.df[date_column],
            errors="coerce",
        )
        scheduled = pd.to_timedelta(
            self.df[scheduled_time_column].astype("string"),
            errors="coerce",
        )
        actual = pd.to_timedelta(
            self.df[actual_time_column].astype("string"),
            errors="coerce",
        )

        scheduled_datetime = date_values + scheduled
        actual_datetime = date_values + actual
        overnight = (
            (actual_datetime < scheduled_datetime)
            & (
                scheduled_datetime - actual_datetime
                > pd.Timedelta(hours=12)
            )
        )
        actual_datetime = actual_datetime.where(
            ~overnight,
            actual_datetime + pd.Timedelta(days=1),
        )

        self.df["scheduled_datetime"] = scheduled_datetime
        self.df["actual_datetime"] = actual_datetime
        self.df[output_column] = (
            actual_datetime - scheduled_datetime
        ).dt.total_seconds() / 60
        self.df[delayed_column] = (
            self.df[output_column] > delayed_after_minutes
        )
        return self.df

    def merge_airports(
        self,
        airports: pd.DataFrame,
        *,
        schedule_code_column: str = "org/des",
        airport_code_column: str = "icao",
        airport_columns: Sequence[str] | None = None,
        validate: str | None = None,
    ) -> pd.DataFrame:
        """Add airport metadata using the schedule ICAO code.

        A left join preserves every flight, including flights whose airport
        code is absent from the reference data.
        """

        if schedule_code_column not in self.df.columns:
            raise KeyError(f"Missing schedule column: {schedule_code_column}")
        if airport_code_column not in airports.columns:
            raise KeyError(f"Missing airport column: {airport_code_column}")

        selected = list(airport_columns) if airport_columns else [
            "icao",
            "iata",
            "name",
            "city",
            "country",
            "latitude",
            "longitude",
            "altitude",
            "type",
        ]
        selected = [
            column for column in selected if column in airports.columns
        ]
        if airport_code_column not in selected:
            selected.insert(0, airport_code_column)

        lookup = airports[selected].copy()
        lookup[airport_code_column] = (
            lookup[airport_code_column].astype("string").str.upper()
        )
        lookup = lookup.drop_duplicates(subset=[airport_code_column])

        self.df[schedule_code_column] = (
            self.df[schedule_code_column].astype("string").str.upper()
        )
        self.df = self.df.merge(
            lookup,
            how="left",
            left_on=schedule_code_column,
            right_on=airport_code_column,
            suffixes=("", "_airport"),
            validate=validate,
        )
        if airport_code_column != schedule_code_column:
            self.df = self.df.drop(columns=[airport_code_column])
        return self.df

    def merge_weather(
        self,
        weather: pd.DataFrame,
        *,
        schedule_time_column: str = "scheduled_datetime",
        weather_time_column: str = "time",
        tolerance: str | pd.Timedelta | None = "1h",
    ) -> pd.DataFrame:
        """Merge the nearest weather observation onto each scheduled flight."""

        if schedule_time_column not in self.df.columns:
            raise KeyError(
                f"Missing schedule datetime column: {schedule_time_column}"
            )
        if weather_time_column not in weather.columns:
            raise KeyError(f"Missing weather column: {weather_time_column}")

        left = self.df.copy()
        right = weather.copy()
        left[schedule_time_column] = pd.to_datetime(
            left[schedule_time_column],
            errors="coerce",
        )
        right[weather_time_column] = pd.to_datetime(
            right[weather_time_column],
            errors="coerce",
        )
        left = left.sort_values(schedule_time_column)
        right = right.sort_values(weather_time_column)

        tolerance_value = (
            pd.Timedelta(tolerance) if tolerance is not None else None
        )
        self.df = pd.merge_asof(
            left,
            right,
            left_on=schedule_time_column,
            right_on=weather_time_column,
            direction="nearest",
            tolerance=tolerance_value,
            suffixes=("", "_weather"),
        )
        return self.df

    def prepare_model_data(
        self,
        *,
        drop_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Keep pre-flight features and remove outcome/leakage columns."""

        default_drop = {
            "actual_datetime",
            "ata_atd_ltc",
            "dl1",
            "dl2",
            "ix1",
            "ix2",
            "gat",
            "identifier",
        }
        columns_to_drop = default_drop.union(drop_columns or set())
        self.df = self.df.drop(
            columns=[
                column
                for column in columns_to_drop
                if column in self.df.columns
            ]
        )
        return self.df

    def relevant_airports(
        self,
        airports: pd.DataFrame,
        *,
        schedule_code_column: str = "org/des",
        airport_code_column: str = "icao",
    ) -> pd.DataFrame:
        """Return only airport reference rows used by the schedule."""

        if schedule_code_column not in self.df.columns:
            raise KeyError(f"Missing schedule column: {schedule_code_column}")
        if airport_code_column not in airports.columns:
            raise KeyError(f"Missing airport column: {airport_code_column}")

        codes = set(
            self.df[schedule_code_column]
            .dropna()
            .astype("string")
            .str.upper()
        )
        airport_codes = airports[airport_code_column].astype("string").str.upper()
        return airports.loc[airport_codes.isin(codes)].copy()
