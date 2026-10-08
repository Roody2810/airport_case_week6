from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_cleaner import AirportDataCleaner
from src.data_fixes import FixedDataMerger
from src.data_inspectie import bouw_inspectierapport, toon_data_inspectie
from src.dashboard_onderdelen import (
    pagina_stijl, toon_heatmap_dag_uur, toon_kaart, toon_kerncijfers, toon_kop, toon_vergelijking,
    toon_lijngrafiek_tijd, toon_voorspelmodel
)


DATA_DIR = Path(__file__).resolve().parent / "data"

# Moet de eerste Streamlit-opdracht zijn
st.set_page_config(page_title="Vertraging op Zürich Airport", page_icon="✈️", layout="wide")
pagina_stijl()


@st.cache_data(show_spinner="Data inlezen en combineren...")
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

    # FixedDataMerger = DataMerger, maar met correcte weerkoppeling (zie src/data_fixes.py)
    merger = FixedDataMerger(schedule)
    merger.create_delay_columns()
    merger.merge_airports(airports)
    merger.merge_weather(weather)
    merged = merger.get_df()

    model_data = FixedDataMerger(merged).prepare_model_data()

    return {
        "schedule": schedule,
        "airports": airports,
        "weather": weather,
        "merged": merged,
        "model_data": model_data,
    }


@st.cache_data(show_spinner="Data-inspectie voorbereiden...")
def load_inspectierapport() -> dict:
    """Cijfers voor de pagina Data-inspectie, berekend op de ruwe bestanden."""
    return bouw_inspectierapport(DATA_DIR)


datasets = load_datasets()

schedule = datasets["schedule"]
airports = datasets["airports"]
weather = datasets["weather"]
merged = datasets["merged"]
model_data = datasets["model_data"]


# --------------------------------------------------------------------------
# Zijbalk: navigatie en filters
# --------------------------------------------------------------------------
PAGINAS = ["📊  Dashboard", "🔍  Data-inspectie"]

st.sidebar.markdown("### ✈️ Zürich Airport")
st.sidebar.caption("Vluchten en vertraging, 2019–2020")
pagina = st.sidebar.radio("Pagina", PAGINAS, label_visibility="collapsed")
st.sidebar.markdown("---")

if pagina == PAGINAS[0]:
    jaar_keuze = st.sidebar.radio(
        "Jaar", ["2019", "2020", "Beide jaren"], index=2, horizontal=True,
        help="2019 is een normaal jaar, 2020 het coronajaar. Een gemiddelde over beide zegt weinig.",
    )
with st.sidebar.expander("Bronnen"):
    st.markdown(
        "- **Vluchten**: `schedule_airport.csv` (Brightspace)\n"
        "- **Luchthavens**: OpenFlights, via Kaggle\n"
        "- **Weer**: Meteostat, station 06670 Zürich-Kloten\n\n"
        "Vertraagd = meer dan 15 minuten later dan gepland."
    )

# --------------------------------------------------------------------------
# Pagina's
# --------------------------------------------------------------------------
if pagina == PAGINAS[1]:
    toon_data_inspectie(load_inspectierapport(), merged)
else:
    toon_kop()
    toon_kerncijfers(merged, jaar_keuze)
    st.write("")

    jaar_data = merged if jaar_keuze == "Beide jaren" else merged[merged["std"].dt.year == int(jaar_keuze)]

    tab_kaart, tab_tijd, tab_toestel, tab_model = st.tabs([
        "🗺️ Bestemmingen", 
        "🕒 Dag en uur", 
        "🛫 Toestellen en banen", 
        "🤖 Voorspelmodel"
    ])

    with tab_kaart:
        toon_kaart(merged, jaar_keuze)

    with tab_tijd:
        toon_lijngrafiek_tijd(jaar_data)
        st.write("---")
        toon_heatmap_dag_uur(jaar_data)

    with tab_toestel:
        toon_vergelijking(jaar_data)

    with tab_model:
        toon_voorspelmodel(model_data)
