"""
Correcties op de gedeelde pipeline in src/, zonder de groepscode aan te passen.

Gebruik in plaats van DataMerger:

    from src.data_fixes import FixedDataMerger

    merger = FixedDataMerger(schedule)
    merger.create_delay_columns()
    merger.merge_airports(airports)
    merger.merge_weather(weather)      # <- gerepareerde versie
    merged = merger.get_df()

FixedDataMerger erft alles van DataMerger. Alleen de methodes hieronder
zijn vervangen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_merger import DataMerger


class FixedDataMerger(DataMerger):
    """DataMerger met een correcte koppeling van het dagweer."""

    def merge_weather(
        self,
        weather: pd.DataFrame,
        *,
        schedule_time_column: str = "scheduled_datetime",
        weather_time_column: str = "date",
        **_ignored,
    ) -> pd.DataFrame:
        """Koppel het dagweer aan elke vlucht op de geplande kalenderdatum.

        Probleem in DataMerger.merge_weather: die gebruikt merge_asof met
        direction="nearest" op weer dat op 00:00 staat. Een vlucht na 12:00
        ligt dichter bij middernacht van de volgende dag en kreeg daardoor
        het weer van morgen (62,5% van alle vluchten). Vluchten op
        31-12-2020 na 12:00 kregen helemaal geen weer.

        Oplossing: beide kanten terugbrengen tot de datum en exact koppelen.
        Extra argumenten van de oude versie (zoals tolerance) worden genegeerd.
        """
        if schedule_time_column not in self.df.columns:
            raise KeyError(f"Missing schedule datetime column: {schedule_time_column}")
        if weather_time_column not in weather.columns:
            raise KeyError(f"Missing weather column: {weather_time_column}")

        left = self.df.copy()
        right = weather.copy()

        left["_flight_date"] = pd.to_datetime(left[schedule_time_column], errors="coerce").dt.normalize()
        right[weather_time_column] = pd.to_datetime(right[weather_time_column], errors="coerce").dt.normalize()
        right = right.drop_duplicates(subset=[weather_time_column])

        rows_before = len(left)
        merged = left.merge(
            right,
            how="left",
            left_on="_flight_date",
            right_on=weather_time_column,
            suffixes=("", "_weather"),
            validate="many_to_one",
        ).drop(columns="_flight_date")

        if len(merged) != rows_before:
            raise ValueError("Weather merge changed the number of rows")

        self.df = merged
        return self.df


# --------------------------------------------------------------------------
# Extra kolommen voor de analyse en de grafieken
# --------------------------------------------------------------------------
# Coordinaten van Zurich Airport (LSZH) uit data/airports-extended.csv
ZURICH_LAT, ZURICH_LON = 47.464698791504, 8.5491695404053
LANGE_AFSTAND_KM = 3500       # grens tussen Europa/omgeving en intercontinentaal


def afstand_km(lat1, lon1, lat2, lon2):
    """
    Afstand over de aardbol (grootcirkel) met de haversine-formule.
    Bron formule: https://en.wikipedia.org/wiki/Haversine_formula
    """
    straal_aarde = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * straal_aarde * np.arcsin(np.sqrt(a))


def add_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Voegt de kolommen toe die de kaart en de heatmap nodig hebben.
    Verwacht de output van FixedDataMerger (na create_delay_columns en merge_airports).

    Nieuwe kolommen: datum, jaar, maand, weekdag (0 = maandag), uur, richting,
    maatschappij, afstand_km (vanaf Zurich) en gebied.
    """
    df = df.copy()
    gepland = pd.to_datetime(df["scheduled_datetime"])
    df["datum"] = gepland.dt.normalize()
    df["jaar"] = gepland.dt.year.astype("Int16")
    df["maand"] = gepland.dt.month.astype("Int16")
    df["weekdag"] = gepland.dt.dayofweek.astype("Int16")
    df["uur"] = gepland.dt.hour.astype("Int16")
    df["richting"] = df["lsv"].map({"S": "Vertrek", "L": "Aankomst"})

    # Maatschappij uit het vluchtnummer: is het derde teken een letter, dan is het
    # een drieletterige code (EZY2191), anders een tweetekencode (LX092, A3850).
    vlucht = df["flt"].astype("string")
    derde_is_letter = vlucht.str[2].str.isalpha().fillna(False).astype(bool)
    df["maatschappij"] = vlucht.str[:2].where(~derde_is_letter, vlucht.str[:3])

    df["afstand_km"] = afstand_km(ZURICH_LAT, ZURICH_LON,
                                  df["latitude"].astype(float), df["longitude"].astype(float))
    df["gebied"] = np.where(df["afstand_km"] >= LANGE_AFSTAND_KM, "Intercontinentaal", "Europa en omgeving")
    df.loc[df["afstand_km"].isna(), "gebied"] = None
    return df
