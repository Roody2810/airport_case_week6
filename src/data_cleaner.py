from __future__ import annotations

from html import unescape
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class DataCleaner:
    """Reusable cleaning utility for tabular datasets."""

    DEFAULT_MISSING_VALUES = (
        "",
        " ",
        "-",
        r"\N",
        "#N/A",
        "N/A",
        "NA",
        "NULL",
        "null",
        "None",
    )

    def __init__(self, df: pd.DataFrame | None = None):
        self.df = df.copy() if df is not None else pd.DataFrame()
        self._issues: dict[str, object] = {}

    @classmethod
    def from_csv(cls, file_path: str, **kwargs) -> "DataCleaner":
        return cls(pd.read_csv(file_path, **kwargs))

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self._issues = {}

    def normalize_columns(self) -> pd.DataFrame:
        self.df.columns = [
            str(column).strip().lower().replace(" ", "_")
            for column in self.df.columns
        ]
        return self.df

    def standardize_empty_values(
        self,
        columns: Iterable[str] | None = None,
        *,
        missing_values: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Convert configured empty-value markers to pandas.NA."""

        columns = columns if columns is not None else self.df.columns
        markers = set(
            self.DEFAULT_MISSING_VALUES
            if missing_values is None
            else missing_values
        )

        for column in columns:
            if column not in self.df.columns:
                continue

            self.df[column] = self.df[column].replace(
                list(markers),
                np.nan,
            )

            if self.df[column].dtype == "object":
                self.df[column] = self.df[column].replace(
                    r"^\s*$",
                    np.nan,
                    regex=True,
                )

        return self.df

    def clean_text_columns(
        self,
        columns: Sequence[str] | None = None,
        *,
        strip: bool = True,
        lower: bool = False,
        replace_html: bool = True,
        collapse_whitespace: bool = True,
    ) -> pd.DataFrame:
        """Clean text without converting missing values to the string 'nan'."""

        target_columns = (
            columns
            if columns is not None
            else self.df.select_dtypes(include=["object", "string"]).columns
        )

        for column in target_columns:
            if column not in self.df.columns:
                continue

            def clean_value(value: object) -> object:
                if pd.isna(value):
                    return value

                value = str(value)

                if replace_html:
                    value = unescape(value)

                if collapse_whitespace:
                    value = " ".join(value.split())

                if strip:
                    value = value.strip()

                if lower:
                    value = value.lower()

                return value if value else np.nan

            self.df[column] = self.df[column].map(clean_value)

        return self.df

    def convert_numeric(
        self,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        columns = columns if columns is not None else self.df.columns

        for column in columns:
            if column in self.df.columns:
                self.df[column] = pd.to_numeric(
                    self.df[column],
                    errors="coerce",
                )

        return self.df

    def convert_datetime(
        self,
        columns: Sequence[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        columns = columns if columns is not None else self.df.columns

        for column in columns:
            if column in self.df.columns:
                self.df[column] = pd.to_datetime(
                    self.df[column],
                    errors="coerce",
                    **kwargs,
                )

        return self.df

    def drop_duplicates(
        self,
        subset: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        self.df = self.df.drop_duplicates(subset=subset)
        return self.df

    def remove_invalid_ranges(
        self,
        column: str,
        min_value: float | int,
        max_value: float | int,
    ) -> pd.DataFrame:
        if column in self.df.columns:
            valid = self.df[column].between(
                min_value,
                max_value,
                inclusive="both",
            )
            self.df = self.df[valid | self.df[column].isna()]

        return self.df

    def identify_null_rows(
        self,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        columns = columns if columns is not None else self.df.columns
        existing_columns = [
            column for column in columns if column in self.df.columns
        ]

        if not existing_columns:
            return self.df.iloc[0:0].copy()

        return self.df[self.df[existing_columns].isna().any(axis=1)]

    def quality_report(self) -> dict[str, object]:
        """Return collected issues and current missing-value statistics."""

        report = dict(self._issues)
        report["row_count"] = len(self.df)
        report["column_count"] = len(self.df.columns)
        report["missing_values"] = self.df.isna().sum()
        report["duplicate_rows"] = int(self.df.duplicated().sum())
        return report

    def clean(
        self,
        text_columns: Sequence[str] | None = None,
        numeric_columns: Sequence[str] | None = None,
        datetime_columns: Sequence[str] | None = None,
        *,
        lower_text: bool = False,
        drop_duplicate_rows: bool = True,
        missing_values: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        self.normalize_columns()
        self.standardize_empty_values(
            missing_values=missing_values,
        )
        self.clean_text_columns(
            text_columns,
            lower=lower_text,
        )

        if numeric_columns:
            self.convert_numeric(numeric_columns)

        if datetime_columns:
            self.convert_datetime(datetime_columns)

        if drop_duplicate_rows:
            self.drop_duplicates()

        return self.df

    def save(self, file_path: str, **kwargs) -> None:
        self.df.to_csv(file_path, index=False, **kwargs)

    def get_df(self) -> pd.DataFrame:
        return self.df


class AirportDataCleaner(DataCleaner):
    """Cleaner for airport reference, schedule, and weather datasets."""

    AIRPORT_COLUMNS = [
        "airport_id",
        "name",
        "city",
        "country",
        "iata",
        "icao",
        "latitude",
        "longitude",
        "altitude",
        "timezone",
        "dst",
        "tz_database_time_zone",
        "type",
        "source",
    ]

    AIRPORT_NUMERIC_COLUMNS = [
        "airport_id",
        "latitude",
        "longitude",
        "altitude",
        "timezone",
    ]

    WEATHER_NUMERIC_COLUMNS = [
        "tavg",
        "tmin",
        "tmax",
        "temp",
        "dwpt",
        "rhum",
        "prcp",
        "snow",
        "wdir",
        "wspd",
        "wpgt",
        "pres",
        "tsun",
        "coco",
    ]

    SCHEDULE_DATE_COLUMNS = ["std"]
    SCHEDULE_TIME_COLUMNS = [
        "sta_std_ltc",
        "ata_atd_ltc",
    ]

    @classmethod
    def from_airports_csv(
        cls,
        file_path: str,
        **kwargs,
    ) -> "AirportDataCleaner":
        """Load airports-extended.csv, which does not contain a header."""

        kwargs.setdefault("header", None)
        kwargs.setdefault("names", cls.AIRPORT_COLUMNS)
        kwargs.setdefault("keep_default_na", True)

        return cls(pd.read_csv(file_path, **kwargs))

    @classmethod
    def from_schedule_csv(
        cls,
        file_path: str,
        **kwargs,
    ) -> "AirportDataCleaner":
        return cls(pd.read_csv(file_path, **kwargs))

    @classmethod
    def from_weather_csv(
        cls,
        file_path: str,
        **kwargs,
    ) -> "AirportDataCleaner":
        return cls(pd.read_csv(file_path, **kwargs))

    def clean_airports(
        self,
        *,
        drop_duplicate_rows: bool = False,
    ) -> pd.DataFrame:
        """Clean and type the airport reference dataset."""

        self.normalize_columns()

        self.standardize_empty_values(
            missing_values=(
                "",
                "-",
                r"\N",
                "#N/A",
                "N/A",
                "NA",
                "NULL",
                "null",
            ),
        )

        self.clean_text_columns(
            columns=[
                "name",
                "city",
                "country",
                "iata",
                "icao",
                "dst",
                "tz_database_time_zone",
                "type",
                "source",
            ],
        )

        self.convert_numeric(self.AIRPORT_NUMERIC_COLUMNS)

        self._issues["airport_validation"] = self.validate_airports()

        if drop_duplicate_rows:
            self.drop_duplicates()

        return self.df

    def identify_missing_airport_codes(
        self,
        *,
        require_both: bool = False,
    ) -> pd.DataFrame:
        """Return rows with genuinely missing IATA and/or ICAO codes.

        Values such as ``AMS/N`` are kept as present values so malformed
        codes can still be found by ``validate_airports``. Only null values
        and known standalone missing markers are treated as missing.
        """

        required_columns = {"iata", "icao"}
        missing_columns = required_columns.difference(self.df.columns)
        if missing_columns:
            raise KeyError(
                "Missing airport code columns: "
                + ", ".join(sorted(missing_columns))
            )

        missing_markers = {
            "",
            "-",
            r"\N",
            "#N/A",
            "N/A",
            "NA",
            "NULL",
            "null",
            "None",
        }

        def is_missing(series: pd.Series) -> pd.Series:
            normalized = series.astype("string").str.strip()
            return series.isna() | normalized.isin(missing_markers)

        missing_iata = is_missing(self.df["iata"])
        missing_icao = is_missing(self.df["icao"])
        missing_codes = (
            missing_iata & missing_icao
            if require_both
            else missing_iata | missing_icao
        )

        result = self.df.loc[missing_codes].copy()
        result["missing_iata"] = missing_iata.loc[missing_codes]
        result["missing_icao"] = missing_icao.loc[missing_codes]
        self._issues["missing_airport_codes"] = result
        return result

    def validate_airports(self) -> dict[str, pd.DataFrame]:
        """Return invalid airport rows without deleting them."""

        issues: dict[str, pd.DataFrame] = {}

        def invalid_range(
            column: str,
            minimum: float,
            maximum: float,
        ) -> pd.DataFrame:
            if column not in self.df.columns:
                return self.df.iloc[0:0].copy()

            invalid = (
                self.df[column].notna()
                & ~self.df[column].between(minimum, maximum)
            )
            return self.df[invalid]

        issues["invalid_latitude"] = invalid_range(
            "latitude",
            -90,
            90,
        )
        issues["invalid_longitude"] = invalid_range(
            "longitude",
            -180,
            180,
        )
        issues["invalid_timezone"] = invalid_range(
            "timezone",
            -12,
            14,
        )

        if "airport_id" in self.df.columns:
            invalid_ids = (
                self.df["airport_id"].notna()
                & (self.df["airport_id"] <= 0)
            )
            issues["invalid_airport_id"] = self.df[invalid_ids]

        if "type" in self.df.columns:
            valid_types = {"airport", "station", "unknown", "port"}
            invalid_types = (
                self.df["type"].notna()
                & ~self.df["type"].isin(valid_types)
            )
            issues["invalid_type"] = self.df[invalid_types]

        if "source" in self.df.columns:
            valid_sources = {"OurAirports", "Legacy", "User"}
            invalid_sources = (
                self.df["source"].notna()
                & ~self.df["source"].isin(valid_sources)
            )
            issues["invalid_source"] = self.df[invalid_sources]

        if "iata" in self.df.columns:
            invalid_iata = (
                self.df["iata"].notna()
                & ~self.df["iata"].str.fullmatch(r"[A-Za-z0-9]{3}")
            )
            issues["invalid_iata"] = self.df[invalid_iata]

        if "icao" in self.df.columns:
            invalid_icao = (
                self.df["icao"].notna()
                & ~self.df["icao"].str.fullmatch(r"[A-Za-z0-9]{4}")
            )
            issues["invalid_icao"] = self.df[invalid_icao]

        return issues

    def validate_schedule(
        self,
        *,
        airport_codes: Iterable[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Validate schedule dates, times, and destination airport codes."""

        issues: dict[str, pd.DataFrame] = {}

        if "std" in self.df.columns:
            parsed_dates = pd.to_datetime(
                self.df["std"],
                format="%d/%m/%Y",
                errors="coerce",
            )
            issues["invalid_std_date"] = self.df[
                self.df["std"].notna() & parsed_dates.isna()
            ]

        for column in self.SCHEDULE_TIME_COLUMNS:
            if column not in self.df.columns:
                continue

            parsed_times = pd.to_datetime(
                self.df[column],
                format="%H:%M:%S",
                errors="coerce",
            )
            issues[f"invalid_{column}"] = self.df[
                self.df[column].notna() & parsed_times.isna()
            ]

        if "org/des" in self.df.columns and airport_codes is not None:
            valid_codes = set(airport_codes)
            unmatched = (
                self.df["org/des"].notna()
                & ~self.df["org/des"].isin(valid_codes)
            )
            issues["unmatched_airport_codes"] = self.df[unmatched]

        return issues

    def clean_schedule(
        self,
        *,
        airport_codes: Iterable[str] | None = None,
        drop_duplicate_rows: bool = False,
    ) -> pd.DataFrame:
        """Clean schedule data while preserving mixed categorical fields."""

        self.normalize_columns()
        self.standardize_empty_values(
            missing_values=(
                "",
                "-",
                "#N/A",
                "N/A",
                "NA",
                "NULL",
                "null",
            ),
        )
        self.clean_text_columns()

        if "std" in self.df.columns:
            self.df["std"] = pd.to_datetime(
                self.df["std"],
                format="%d/%m/%Y",
                errors="coerce",
            )

        for column in self.SCHEDULE_TIME_COLUMNS:
            if column in self.df.columns:
                self.df[column] = pd.to_datetime(
                    self.df[column],
                    format="%H:%M:%S",
                    errors="coerce",
                ).dt.time

        self._issues["schedule_validation"] = self.validate_schedule(
            airport_codes=airport_codes,
        )

        if drop_duplicate_rows:
            self.drop_duplicates()

        return self.df

    def combine_schedule_datetime(
        self,
        date_column: str = "std",
        time_column: str = "sta_std_ltc",
        output_column: str | None = None,
    ) -> pd.DataFrame:
        """Combine a schedule date and time into one datetime column."""

        if date_column not in self.df.columns:
            raise KeyError(f"Missing date column: {date_column}")

        if time_column not in self.df.columns:
            raise KeyError(f"Missing time column: {time_column}")

        output_column = output_column or f"{date_column}_{time_column}"

        date_values = pd.to_datetime(
            self.df[date_column],
            errors="coerce",
        )

        time_values = pd.to_timedelta(
            self.df[time_column].astype("string"),
            errors="coerce",
        )

        self.df[output_column] = date_values + time_values
        return self.df

    def validate_weather(
        self,
        *,
        expected_frequency: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Validate weather ranges and timestamp continuity."""

        issues: dict[str, pd.DataFrame] = {}

        ranges = {
            "rhum": (0, 100),
            "wdir": (0, 360),
            "wspd": (0, None),
            "wpgt": (0, None),
            "pres": (0, None),
            "tsun": (0, 1440),
            "coco": (1, 5),
        }

        temperature_ranges = {
            "tavg": (-90, 70),
            "tmin": (-90, 70),
            "tmax": (-90, 70),
            "temp": (-90, 70),
            "dwpt": (-100, 70),
        }
        ranges.update(temperature_ranges)

        for column, (minimum, maximum) in ranges.items():
            if column not in self.df.columns:
                continue

            values = self.df[column]
            invalid = values.notna()

            if minimum is not None:
                invalid &= values < minimum

            if maximum is not None:
                invalid |= values > maximum

            issues[f"invalid_{column}"] = self.df[invalid]

        weather_time_column = (
            "time" if "time" in self.df.columns
            else "date" if "date" in self.df.columns
            else None
        )
        if weather_time_column is not None:
            timestamps = pd.to_datetime(
                self.df[weather_time_column],
                errors="coerce",
            )

            issues["invalid_weather_time"] = self.df[
                self.df[weather_time_column].notna() & timestamps.isna()
            ]

            ordered = timestamps.dropna().sort_values()
            if len(ordered) > 1 and expected_frequency is not None:
                intervals = ordered.diff().dropna()
                expected_interval = pd.to_timedelta(
                    1,
                    unit=expected_frequency,
                )
                invalid_intervals = intervals[
                    intervals != expected_interval
                ]
                issues["non_expected_weather_intervals"] = pd.DataFrame(
                    {"interval": invalid_intervals}
                )

        return issues

    def clean_weather(
        self,
        *,
        drop_duplicate_rows: bool = False,
        date_column: str | None = None,
        expected_frequency: str | None = None,
    ) -> pd.DataFrame:
        """Clean and validate hourly or daily weather data."""

        self.normalize_columns()
        self.standardize_empty_values(
            missing_values=(
                "",
                "-",
                r"\N",
                "#N/A",
                "N/A",
                "NA",
                "NULL",
                "null",
            ),
        )

        datetime_columns = (
            [date_column]
            if date_column is not None
            else ["time"] if "time" in self.df.columns
            else ["date"] if "date" in self.df.columns
            else []
        )
        if expected_frequency is None and datetime_columns:
            expected_frequency = (
                "h" if datetime_columns[0] == "time" else "D"
            )
        self.convert_datetime(datetime_columns)
        self.convert_numeric(self.WEATHER_NUMERIC_COLUMNS)

        self._issues["weather_validation"] = self.validate_weather(
            expected_frequency=expected_frequency,
        )

        if drop_duplicate_rows:
            self.drop_duplicates()

        return self.df

    def validate_schedule_against_airports(
        self,
        airports: pd.DataFrame,
        schedule_code_column: str = "org/des",
        airport_code_column: str = "icao",
    ) -> pd.DataFrame:
        """Return schedule rows whose airport code is not in the reference data."""

        if schedule_code_column not in self.df.columns:
            raise KeyError(
                f"Missing schedule column: {schedule_code_column}"
            )

        if airport_code_column not in airports.columns:
            raise KeyError(
                f"Missing airport column: {airport_code_column}"
            )

        valid_codes = set(
            airports[airport_code_column]
            .dropna()
            .astype(str)
            .str.upper()
        )

        codes = self.df[schedule_code_column].astype("string").str.upper()
        unmatched = codes.notna() & ~codes.isin(valid_codes)

        result = self.df[unmatched].copy()
        self._issues["unmatched_schedule_airports"] = result
        return result



airportDataCleaner = AirportDataCleaner