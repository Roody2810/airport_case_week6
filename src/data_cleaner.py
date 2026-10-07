from __future__ import annotations

from html import unescape
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
""" Example use of data cleaner 
from src.data_cleaner import DataCleaner

cleaner = DataCleaner.from_csv("data/ufo_sighting_data.csv", low_memory=False)

df = cleaner.clean(
    text_columns=["city", "state/province", "country", "description"],
    numeric_columns=["length_of_encounter_seconds", "latitude", "longitude"],
    datetime_columns=["date_time", "date_documented"],
    lower_text=False
)

print(df.head())

"""

class DataCleaner:
    """Reusable cleaning utility for tabular datasets."""

    def __init__(self, df: pd.DataFrame | None = None):
        self.df = df.copy() if df is not None else pd.DataFrame()

    @classmethod
    def from_csv(cls, file_path: str, **kwargs) -> "DataCleaner":
        return cls(pd.read_csv(file_path, **kwargs))

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def normalize_columns(self) -> pd.DataFrame:
        self.df.columns = [str(col).strip().lower() for col in self.df.columns]
        return self.df

    def standardize_empty_values(self, columns: Iterable[str] | None = None) -> pd.DataFrame:
        cols = columns if columns is not None else self.df.columns
        for col in cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].replace(r"^\s*$", np.nan, regex=True)
        return self.df

    def clean_text_columns(
        self,
        columns: Sequence[str] | None = None,
        *,
        strip: bool = True,
        lower: bool = False,
        replace_html: bool = True,
    ) -> pd.DataFrame:
        target_cols = columns if columns is not None else self.df.select_dtypes(include=["object"]).columns

        for col in target_cols:
            if col not in self.df.columns:
                continue

            if replace_html:
                self.df[col] = self.df[col].map(
                    lambda value: unescape(value) if isinstance(value, str) else value
                )

            self.df[col] = self.df[col].astype(str)

            if strip:
                self.df[col] = self.df[col].str.strip()

            if lower:
                self.df[col] = self.df[col].str.lower()

            self.df[col] = self.df[col].replace({"nan": np.nan, "None": np.nan})
            self.df[col] = self.df[col].replace(r"\s+", " ", regex=True)

        return self.df

    def convert_numeric(self, columns: Sequence[str] | None = None) -> pd.DataFrame:
        cols = columns if columns is not None else self.df.columns
        for col in cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        return self.df

    def convert_datetime(self, columns: Sequence[str] | None = None, **kwargs) -> pd.DataFrame:
        cols = columns if columns is not None else self.df.columns
        for col in cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce", **kwargs)
        return self.df

    def drop_duplicates(self) -> pd.DataFrame:
        self.df = self.df.drop_duplicates()
        return self.df

    def remove_invalid_ranges(self, column: str, min_value: float | int, max_value: float | int) -> pd.DataFrame:
        if column in self.df.columns:
            self.df = self.df[(self.df[column].between(min_value, max_value)) | self.df[column].isna()]
        return self.df

    def identify_null_rows(self, columns: Sequence[str] | None = None) -> pd.DataFrame:
        cols = columns if columns is not None else self.df.columns
        return self.df[self.df[cols].isna().any(axis=1)]

    def clean(
        self,
        text_columns: Sequence[str] | None = None,
        numeric_columns: Sequence[str] | None = None,
        datetime_columns: Sequence[str] | None = None,
        *,
        lower_text: bool = False,
        drop_duplicate_rows: bool = True,
    ) -> pd.DataFrame:
        self.normalize_columns()
        self.standardize_empty_values()
        self.clean_text_columns(text_columns, lower=lower_text)

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


