"""
Pagina Data-inspectie: wat zit er in de drie bronnen, wat was er mis en wat hebben we gedaan?

    bouw_inspectierapport(data_dir)        rekent alle cijfers uit (een keer, daarna cachen)
    toon_data_inspectie(rapport, merged)   tekent de pagina

De cijfers worden berekend op de ruwe bestanden in data/, dus ze kloppen ook
als iemand de data of de opschoning aanpast.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_cleaner import AirportDataCleaner
from src.data_merger import DataMerger
from src.dashboard_onderdelen import BLAUW, INKT, ORANJE, nl, opmaak, tabel, toon
from src.data_fixes import FixedDataMerger

VERTRAAGD_NA_MIN = 15   # zelfde grens als DataMerger.create_delay_columns


# ==========================================================================
# 1. Cijfers berekenen
# ==========================================================================
def _inspecteer_kolommen(ruw: pd.DataFrame) -> pd.DataFrame:
    """Per kolom: unieke waarden, hoeveel 'lege' tekens en een voorbeeld."""
    rijen = []
    for kolom in ruw.columns:
        waarden = ruw[kolom]
        rijen.append({
            "Kolom": kolom,
            "Unieke waarden": waarden.nunique(),
            "Aantal '-'": int((waarden == "-").sum()),
            "Aantal '#N/A'": int((waarden == "#N/A").sum()),
            "Leeg (%)": round(100 * waarden.isin(["-", "#N/A", ""]).mean(), 1),
            "Voorbeeld": waarden.iloc[0],
        })
    return pd.DataFrame(rijen)


def bouw_inspectierapport(data_dir: Path) -> dict:
    """Leest de ruwe bronnen en verzamelt alles wat de pagina Data-inspectie laat zien."""
    data_dir = Path(data_dir)
    rapport: dict = {}

    # --- Vluchten, precies zoals ze binnenkomen (alles als tekst) ----------
    ruw = pd.read_csv(data_dir / "schedule_airport.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    rapport["vlucht_rijen"], rapport["vlucht_kolommen"] = ruw.shape
    rapport["vlucht_voorbeeld"] = ruw.head(8)
    rapport["vlucht_per_kolom"] = _inspecteer_kolommen(ruw)
    rapport["dubbele_rijen"] = int(ruw.duplicated().sum())
    rapport["dubbele_identifier"] = int(ruw["Identifier"].duplicated().sum())
    rapport["zonder_code"] = int((ruw["Org/Des"] == "#N/A").sum())

    # Middernacht: zonder correctie lijkt een vlucht van 23:55 die om 00:10 vertrekt bijna 24 uur te vroeg
    datum = pd.to_datetime(ruw["STD"], format="%d/%m/%Y")
    ruwe_vertraging = ((datum + pd.to_timedelta(ruw["ATA_ATD_ltc"]))
                       - (datum + pd.to_timedelta(ruw["STA_STD_ltc"]))).dt.total_seconds() / 60
    over_middernacht = ruwe_vertraging < -12 * 60
    rapport["over_middernacht"] = int(over_middernacht.sum())
    rapport["min_voor_correctie"] = float(ruwe_vertraging.min())
    nacht = ruw.loc[over_middernacht, ["STD", "FLT", "STA_STD_ltc", "ATA_ATD_ltc"]].copy()
    nacht["Zonder correctie (min)"] = ruwe_vertraging[over_middernacht].round(1)
    nacht["Na correctie (min)"] = (ruwe_vertraging[over_middernacht] + 24 * 60).round(1)
    rapport["middernacht_voorbeelden"] = nacht.rename(columns={
        "STD": "Datum", "FLT": "Vlucht", "STA_STD_ltc": "Gepland", "ATA_ATD_ltc": "Werkelijk"})

    # --- Luchthavens ------------------------------------------------------
    havens = AirportDataCleaner.from_airports_csv(data_dir / "airports-extended.csv").clean_airports()
    rapport["haven_rijen"] = len(havens)
    rapport["haven_soorten"] = havens["type"].value_counts()
    rapport["haven_zonder_icao"] = int(havens["icao"].isna().sum())

    codes = pd.Series(ruw.loc[ruw["Org/Des"] != "#N/A", "Org/Des"].unique())
    icao, iata = set(havens["icao"].dropna()), set(havens["iata"].dropna())
    rapport["bestemmingen"] = len(codes)
    rapport["via_icao"] = int(codes.isin(icao).sum())
    rapport["via_iata"] = int(codes.isin(iata).sum())
    onbekend = ruw.loc[ruw["Org/Des"].isin(codes[~codes.isin(icao)]), "Org/Des"].value_counts()
    rapport["onbekende_codes"] = onbekend
    rapport["vluchten_zonder_haven"] = int(onbekend.sum())

    # --- Weer -------------------------------------------------------------
    weer_ruw = pd.read_csv(data_dir / "weather_zurich_2019-2020.csv", encoding="utf-8-sig")
    rapport["weer_dagen"] = len(weer_ruw)
    rapport["weer_periode"] = (str(weer_ruw["date"].min())[:10], str(weer_ruw["date"].max())[:10])
    rapport["weer_leeg"] = weer_ruw.drop(columns="date").isna().sum()
    rapport["weer_bereik"] = weer_ruw.drop(columns="date").describe().T[["min", "max"]]
    rapport["weer_ruw"] = weer_ruw.assign(date=pd.to_datetime(weer_ruw["date"]))

    # --- Weerkoppeling: oude versie tegenover de gerepareerde --------------
    schema = AirportDataCleaner.from_schedule_csv(data_dir / "schedule_airport.csv").clean_schedule()
    weer = AirportDataCleaner.from_weather_csv(data_dir / "weather_zurich_2019-2020.csv").clean_weather()
    for naam, klasse in [("oud", DataMerger), ("nieuw", FixedDataMerger)]:
        merger = klasse(schema)
        merger.create_delay_columns()
        merger.merge_weather(weer)
        df = merger.get_df()
        andere_dag = df["date"].notna() & (df["date"].dt.date != df["scheduled_datetime"].dt.date)
        rapport[f"weer_{naam}_andere_dag"] = int(andere_dag.sum())
        rapport[f"weer_{naam}_geen_weer"] = int(df["tavg"].isna().sum())
    rapport["weer_vluchten"] = len(schema)
    return rapport


# ==========================================================================
# 2. De pagina
# ==========================================================================
def _tabblad_vluchten(r: dict, merged: pd.DataFrame) -> None:
    a, b, c = st.columns(3)
    a.metric("Rijen", nl(r["vlucht_rijen"]))
    b.metric("Kolommen", r["vlucht_kolommen"])
    c.metric("Periode", "2019 – 2020")

    st.markdown("##### Zo komt het bestand binnen")
    tabel(r["vlucht_voorbeeld"])

    st.markdown("##### Per kolom")
    st.caption("Er staat geen enkele echt lege cel in het bestand. 'Leeg' is hier een streepje `-` of `#N/A`.")
    tabel(r["vlucht_per_kolom"], height=35 * (len(r["vlucht_per_kolom"]) + 1) + 3)

    st.markdown("##### Wat viel op, en wat hebben we gedaan")
    stappen = pd.DataFrame([
        ["Streepje '-' betekent 'geen waarde' (gate, vertragingscode, baanconcept)",
         "Omgezet naar een echte lege waarde", nl(int(r["vlucht_per_kolom"]["Aantal '-'"].sum())) + " cellen"],
        ["'#N/A' als herkomst of bestemming", "Omgezet naar leeg; deze vluchten komen niet op de kaart",
         nl(r["zonder_code"]) + " vluchten"],
        ["Exact dubbele rijen", "Gecontroleerd", nl(r["dubbele_rijen"])],
        ["Zelfde datum, tijd en vluchtnummer twee keer", "Laten staan: niet te zien welke klopt",
         nl(r["dubbele_identifier"]) + " keer"],
        ["Datum en tijden staan als tekst in losse kolommen", "Samengevoegd tot geplande en werkelijke datum-tijd",
         nl(r["vlucht_rijen"]) + " rijen"],
        ["Werkelijke tijd na middernacht, maar er is alleen een geplande datum",
         "Een dag opgeteld bij de werkelijke tijd", nl(r["over_middernacht"]) + " vluchten"],
        ["Geen kolom voor vertraging", "Zelf gemaakt: werkelijk − gepland, in minuten", "nieuwe kolom"],
        ["Vertragingscodes (DL1–IX2) en werkelijke gate zijn pas na de vlucht bekend",
         "Weggelaten uit de modeldata, anders 'spiekt' een model", "6 kolommen"],
    ], columns=["Wat viel op", "Wat we deden", "Hoe vaak"])
    tabel(stappen)

    st.markdown("##### Het middernachtprobleem")
    m1, m2 = st.columns([1, 2])
    m1.metric("Kleinste 'vertraging' zonder correctie", nl(r["min_voor_correctie"]) + " min")
    m1.metric("Kleinste vertraging na correctie", nl(merged["delay_minutes"].min(), 1) + " min")
    with m2:
        st.caption("Een vlucht gepland om 23:55 die om 00:10 vertrekt is 15 minuten te laat, niet bijna 24 uur te vroeg. "
                   "Ligt de werkelijke tijd meer dan 12 uur vóór de geplande tijd, dan tellen we er een dag bij op.")
        tabel(r["middernacht_voorbeelden"])

    st.markdown("##### De nieuwe kolom: vertraging in minuten")
    # Zelf tellen per 5 minuten: sneller dan 323.000 losse punten naar de browser sturen
    randen = np.arange(-60, 125, 5)
    aantallen, _ = np.histogram(merged["delay_minutes"].dropna().clip(-60, 119.9), bins=randen)
    fig = go.Figure(go.Bar(
        x=randen[:-1] + 2.5, y=aantallen, width=5, marker=dict(color=BLAUW, line=dict(color="white", width=1)),
        customdata=np.stack([randen[:-1], randen[1:]], axis=-1),
        hovertemplate="%{customdata[0]} tot %{customdata[1]} min<br>%{y:,} vluchten<extra></extra>"))
    opmaak(fig, x_titel="Vertraging in minuten (negatief = te vroeg; alles boven 120 staat in de laatste staaf)",
           y_titel="Aantal vluchten", hoogte=320, legenda=False)
    for x, tekst in [(0, "op tijd"), (VERTRAAGD_NA_MIN, "vanaf hier 'vertraagd'")]:
        fig.add_shape(type="line", x0=x, x1=x, yref="paper", y0=0, y1=1, line=dict(color=INKT, width=1, dash="dot"))
        fig.add_annotation(x=x, yref="paper", y=1, text=tekst, showarrow=False, xanchor="left", xshift=4,
                           yanchor="top", font=dict(size=12, color=INKT))
    toon(fig)
    v = merged["delay_minutes"]
    st.caption(
        f"De helft van de vluchten zit tussen {nl(v.quantile(0.25))} en {nl(v.quantile(0.75))} minuten. "
        f"De staart naar rechts is lang: 1% heeft meer dan {nl(v.quantile(0.99))} minuten vertraging, de grootste is "
        f"{nl(v.max() / 60, 1)} uur. Daarom rekenen we vooral met het aandeel vertraagde vluchten en de mediaan, "
        "niet alleen met het gemiddelde."
    )


def _tabblad_luchthavens(r: dict) -> None:
    a, b, c = st.columns(3)
    a.metric("Locaties in het bestand", nl(r["haven_rijen"]))
    b.metric("Bestemmingen in de vluchtdata", r["bestemmingen"])
    c.metric("Vluchten met een luchthaven",
             nl(100 * (1 - (r["vluchten_zonder_haven"] + r["zonder_code"]) / r["vlucht_rijen"]), 2) + "%")
    st.markdown("Het bestand heeft **geen kopregel** en gebruikt `\\N` voor een ontbrekende waarde. "
                "De kolomnamen zijn zelf toegevoegd. Het bevat ook niet alleen vliegvelden:")
    soorten = r["haven_soorten"].rename_axis("Soort").reset_index(name="Aantal")
    soorten["Soort"] = soorten["Soort"].map({"airport": "Vliegveld", "station": "Treinstation", "port": "Haven",
                                             "unknown": "Onbekend"}).fillna(soorten["Soort"])
    tabel(soorten)

    st.markdown("##### Koppelen: ICAO of IATA?")
    st.markdown(
        f"`Org/Des` bevat codes van vier letters. Koppelen op de **ICAO**-kolom vindt **{r['via_icao']}** van de "
        f"{r['bestemmingen']} bestemmingen; koppelen op de **IATA**-kolom maar **{r['via_iata']}**. "
        "Dat is het verschil tussen een kaart en een lege kaart. We koppelen daarom op ICAO."
    )
    onbekend = r["onbekende_codes"].rename_axis("Code").reset_index(name="Vluchten")
    st.caption(f"Niet gevonden op ICAO: {len(onbekend)} codes, samen {nl(r['vluchten_zonder_haven'])} vluchten. "
               "Een deel is een code van drie letters (zoals `BER`); deze vluchten laten we buiten de kaart.")
    tabel(onbekend)


def _tabblad_weer(r: dict) -> None:
    a, b, c = st.columns(3)
    a.metric("Dagen", r["weer_dagen"])
    b.metric("Eerste dag", r["weer_periode"][0])
    c.metric("Laatste dag", r["weer_periode"][1])
    st.markdown("Dagwaarden van Meteostat, station 06670 (Zürich-Kloten), voor precies de periode van de vluchten. "
                "Elke dag van 2019 en 2020 is aanwezig.")

    leeg = r["weer_leeg"].rename_axis("Kolom").reset_index(name="Lege dagen")
    leeg["Leeg (%)"] = (100 * leeg["Lege dagen"] / r["weer_dagen"]).round(1)
    bereik = r["weer_bereik"].reset_index().rename(columns={"index": "Kolom", "min": "Laagste", "max": "Hoogste"})
    leeg = leeg.merge(bereik, on="Kolom", how="left")
    leeg["Wat we deden"] = np.where(leeg["Leeg (%)"] == 100, "Niet bruikbaar (helemaal leeg)",
                                    np.where(leeg["Lege dagen"] > 0, "Leeg gelaten, niet op 0 gezet", ""))
    tabel(leeg)
    st.caption("Sneeuw, windrichting en zonuren zijn in deze twee jaar helemaal leeg. Ontbrekende neerslag laten we leeg: "
               "'onbekend' is iets anders dan 'droog'. De bereiken van temperatuur, wind en luchtdruk zijn realistisch.")

    st.markdown("##### Weer koppelen aan vluchten: hier ging het mis")
    st.markdown(
        "Het weer staat per dag op 00:00. De eerste versie koppelde elke vlucht aan het **dichtstbijzijnde** tijdstip. "
        "Een vlucht na 12:00 ligt dichter bij middernacht van de **volgende** dag, en kreeg daardoor het weer van morgen."
    )
    tabel(pd.DataFrame([
        ["Eerste versie (dichtstbijzijnde tijdstip)",
         nl(r["weer_oud_andere_dag"]), nl(100 * r["weer_oud_andere_dag"] / r["weer_vluchten"], 1) + "%",
         nl(r["weer_oud_geen_weer"])],
        ["Nu (zelfde kalenderdatum)",
         nl(r["weer_nieuw_andere_dag"]), nl(100 * r["weer_nieuw_andere_dag"] / r["weer_vluchten"], 1) + "%",
         nl(r["weer_nieuw_geen_weer"])],
    ], columns=["Koppeling", "Vluchten met weer van een andere dag", "Aandeel", "Vluchten zonder weer"]))
    st.caption("De vluchten zonder weer in de eerste versie waren vluchten op 31 december 2020 na 12:00: "
               "voor 'morgen' stond er geen weer in het bestand. De reparatie staat in `src/data_fixes.py`.")

    st.markdown("##### Neerslag per dag")
    w = r["weer_ruw"]
    fig = go.Figure(go.Scatter(x=w["date"], y=w["prcp"], mode="lines", line=dict(color=BLAUW, width=1.5),
                               name="Neerslag", hovertemplate="%{y:,.1f} mm<extra></extra>", connectgaps=False))
    leeg_dagen = w.loc[w["prcp"].isna(), "date"]
    if len(leeg_dagen):
        fig.add_trace(go.Scatter(x=leeg_dagen, y=[0] * len(leeg_dagen), mode="markers", name="Geen meting",
                                 marker=dict(color=ORANJE, size=7, symbol="line-ns-open", line=dict(width=2)),
                                 hovertemplate="%{x|%d-%m-%Y}: geen meting<extra></extra>"))
    opmaak(fig, y_titel="Neerslag per dag (mm)", hoogte=300)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(hoverformat="%d-%m-%Y", dtick="M3", tickformat="%m-%Y")
    toon(fig)
    st.caption(f"Oranje streepjes: de {len(leeg_dagen)} dagen zonder neerslagmeting.")


def _tabblad_resultaat(merged: pd.DataFrame) -> None:
    a, b, c = st.columns(3)
    a.metric("Rijen", nl(len(merged)), "geen vlucht weggegooid", delta_color="off")
    b.metric("Kolommen", merged.shape[1])
    c.metric("Bronnen gecombineerd", 3)
    st.markdown("##### Wat nog leeg is, en waarom")
    leeg = merged.isna().sum()
    leeg = leeg[leeg > 0].rename("Lege waarden").rename_axis("Kolom").reset_index()
    uitleg = {
        "dl1": "Vertragingscode: alleen ingevuld als er een oorzaak is geregistreerd", "ix1": "Idem",
        "dl2": "Idem (tweede oorzaak)", "ix2": "Idem", "tar": "Geen geplande gate bekend",
        "gat": "Geen werkelijke gate bekend", "rwc": "Geen baanconcept bekend", "org/des": "#N/A in de bron",
        "prcp": "Geen neerslagmeting die dag", "snow": "Kolom is leeg in de bron", "wdir": "Kolom is leeg in de bron",
        "tsun": "Kolom is leeg in de bron", "iata": "Luchthaven heeft geen IATA-code",
    }
    leeg["Reden"] = leeg["Kolom"].map(uitleg).fillna("Vlucht zonder gekoppelde luchthaven")
    leeg["Leeg (%)"] = (100 * leeg["Lege waarden"] / len(merged)).round(2)
    tabel(leeg)
    st.markdown("##### De gecombineerde dataset (eerste 50 rijen)")
    tabel(merged.head(50))


def toon_data_inspectie(rapport: dict, merged: pd.DataFrame) -> None:
    """Tekent de pagina Data-inspectie."""
    st.markdown('<div class="kop-label">Data-inspectie</div>', unsafe_allow_html=True)
    st.title("Wat zit er in de data?")
    st.markdown(
        '<div class="kop-intro">Eerst kijken wat er in de drie bronnen zit, dan pas conclusies trekken. '
        "Per bron: hoe het binnenkomt, wat er mis was en wat we eraan hebben gedaan. "
        "Deze pagina gebruikt altijd alle vluchten uit 2019 en 2020.</div>",
        unsafe_allow_html=True,
    )
    tab_v, tab_l, tab_w, tab_r = st.tabs(["✈️  Vluchten", "📍  Luchthavens", "🌦️  Weer", "✅  Resultaat"])
    with tab_v:
        _tabblad_vluchten(rapport, merged)
    with tab_l:
        _tabblad_luchthavens(rapport)
    with tab_w:
        _tabblad_weer(rapport)
    with tab_r:
        _tabblad_resultaat(merged)
