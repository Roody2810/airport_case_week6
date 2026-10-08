"""
Onderdelen voor het dashboard, aangepast aan de kolomnamen van de groepspipeline
(src/data_cleaner.py + src/data_fixes.py).

    toon_kaart(df, jaar_keuze)     kaart met bestemmingen, ranglijst en verloop per bestemming
    toon_heatmap_dag_uur(df)       heatmap: aandeel vertraagd per dag van de week en uur

Gebruik in app.py:

    from src.dashboard_onderdelen import toon_kaart, toon_heatmap_dag_uur
    toon_kaart(merged)
    toon_heatmap_dag_uur(merged)

Gebruikte documentatie (code is aangepast aan onze eigen data):
    Plotly kaarten         https://plotly.com/python/scatter-plots-on-maps/
    Plotly lijnen op kaart https://plotly.com/python/lines-on-maps/
    Plotly heatmap         https://plotly.com/python/heatmaps/
    Streamlit widgets      https://docs.streamlit.io/develop/api-reference/widgets
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_fixes import LANGE_AFSTAND_KM, ZURICH_LAT, ZURICH_LON, add_analysis_columns

# --------------------------------------------------------------------------
# Kleuren en vaste waarden
# --------------------------------------------------------------------------
BLAUW, ORANJE, ROOD = "#2a78d6", "#eb6834", "#e34948"
INKT, INKT_ZACHT, RASTER, LAND = "#0b0b0b", "#52514e", "#e6e5e1", "#eeede9"
BLAUW_MIDDEN, BLAUW_DONKER = "#3987e5", "#0d366b"
SCHAAL_BLAUW = [[0, "#cde2fb"], [0.5, BLAUW_MIDDEN], [1, BLAUW_DONKER]]
SCHAAL_ROOD_BLAUW = [[0, ROOD], [0.5, "#f0efec"], [1, BLAUW]]

LOCKDOWN = "2020-03-16"      # Zwitserland kondigt de 'buitengewone situatie' af
WEEKDAGEN = ["Maandag", "Dinsdag", "Woensdag", "Donderdag", "Vrijdag", "Zaterdag", "Zondag"]

# Kolommen die add_analysis_columns toevoegt
_ANALYSE_KOLOMMEN = {"datum", "jaar", "weekdag", "uur", "afstand_km", "gebied", "maatschappij"}


# --------------------------------------------------------------------------
# Hulpfuncties
# --------------------------------------------------------------------------
def _met_analysekolommen(df: pd.DataFrame) -> pd.DataFrame:
    """Voegt de extra kolommen toe als ze er nog niet zijn."""
    if _ANALYSE_KOLOMMEN.issubset(df.columns):
        return df
    return add_analysis_columns(df)


def nl(getal, decimalen=0):
    """Getal met Nederlandse notatie: 323.461 en 8,3"""
    if getal is None or (isinstance(getal, float) and np.isnan(getal)):
        return "–"
    tekst = f"{getal:,.{decimalen}f}"
    return tekst.replace(",", "X").replace(".", ",").replace("X", ".")


def _volle_breedte(functie):
    """Streamlit heeft de naam van de breedte-optie veranderd; dit werkt in oud en nieuw."""
    parameters = inspect.signature(functie).parameters
    if "width" in parameters and parameters["width"].default == "stretch":
        return {}
    return {"use_container_width": True}


def toon(fig):
    st.plotly_chart(fig, theme=None, config={"displaylogo": False}, **_volle_breedte(st.plotly_chart))


def tabel(df, **opties):
    st.dataframe(df, hide_index=True, **_volle_breedte(st.dataframe), **opties)


def opmaak(fig, x_titel=None, y_titel=None, hoogte=380, legenda=True, marge_rechts=20):
    """Dezelfde rustige opmaak voor elke grafiek: dun raster, legenda bovenaan."""
    fig.update_layout(
        template="plotly_white", height=hoogte, separators=",.",
        font=dict(family="Source Sans Pro, Segoe UI, sans-serif", size=13, color=INKT_ZACHT),
        margin=dict(l=10, r=marge_rechts, t=40 if legenda else 15, b=10),
        showlegend=legenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=dict(text="")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
    )
    fig.update_xaxes(title_text=x_titel, gridcolor=RASTER, linecolor="#b5b4ae", zeroline=False)
    fig.update_yaxes(title_text=y_titel, gridcolor=RASTER, zerolinecolor="#b5b4ae", rangemode="tozero")
    return fig


def hoverdata(data, kolommen):
    """Extra kolommen voor de hovertekst, met tekst en getallen door elkaar."""
    return data[kolommen].astype(object).to_numpy()


def samenvatting(data, groep):
    """Aantal vluchten, gemiddelde, mediaan en aandeel vertraagd per groep."""
    uit = (
        data.groupby(groep, observed=True)
        .agg(vluchten=("delay_minutes", "size"), gem=("delay_minutes", "mean"),
             mediaan=("delay_minutes", "median"), aandeel=("is_delayed", "mean"))
        .reset_index()
    )
    uit["aandeel"] = uit["aandeel"] * 100
    return uit


def lockdown_lijn(fig, tekst="16 mrt 2020: lockdown"):
    """Verticale lijn op een datum-as. Los getekend als vorm + tekst."""
    fig.add_shape(type="line", x0=LOCKDOWN, x1=LOCKDOWN, yref="paper", y0=0, y1=1,
                  line=dict(color=INKT_ZACHT, width=1, dash="dot"))
    fig.add_annotation(x=LOCKDOWN, yref="paper", y=1, text=tekst, showarrow=False,
                       xanchor="left", xshift=6, yanchor="top", font=dict(size=12, color=INKT_ZACHT))


# ==========================================================================
# BESTEMMINGEN (KAART)
# ==========================================================================
def toon_kaart(df: pd.DataFrame, jaar_keuze: str = "Beide jaren") -> None:
    """
    Kaart met bestemmingen vanaf Zurich, plus een ranglijst en het verloop per maand.

    df          gecombineerde vluchtdata (beide jaren; andere filters mag je vooraf toepassen)
    jaar_keuze  "2019", "2020" of "Beide jaren"
    """
    zonder_jaar = _met_analysekolommen(df)
    selectie = zonder_jaar if jaar_keuze == "Beide jaren" else zonder_jaar[zonder_jaar["jaar"] == int(jaar_keuze)]
    if len(selectie) < 50:
        st.warning("Met deze filters blijven er te weinig vluchten over.")
        return

    st.subheader("Waarheen, en waar loopt het uit?")
    c1, c2, c3 = st.columns([1.2, 1.5, 1.3])
    gebied = c1.radio("Gebied", ["Europa en omgeving", "Intercontinentaal"], key="kaart_gebied",
                      help=f"Grens: {nl(LANGE_AFSTAND_KM)} km vanaf Zürich.")
    kleur_keuze = c2.radio("Kleur toont", ["Aandeel vertraagd (%)", "Gemiddelde vertraging (min)",
                                           "Verandering aantal vluchten 2020 t.o.v. 2019 (%)"], key="kaart_kleur")
    minimum = c3.slider("Minimaal aantal vluchten per bestemming", 10, 1000,
                        200 if gebied == "Europa en omgeving" else 50, step=10, key="kaart_minimum",
                        help="Bestemmingen met weinig vluchten geven toevallige uitschieters.")

    met_gebied = zonder_jaar["gebied"].notna()
    st.caption(
        "**Waarom twee kaarten?** 295 bestemmingen op één wereldkaart wordt een wolk stippen. "
        f"In Europa en omgeving zit {nl((zonder_jaar['gebied'] == 'Europa en omgeving').sum() / max(met_gebied.sum(), 1) * 100)}% "
        "van de vluchten en zie je de dichtheid; intercontinentaal zijn het er weinig, maar elke lijn is een verbinding."
    )

    info = zonder_jaar.drop_duplicates("org/des")[
        ["org/des", "name", "city", "country", "latitude", "longitude", "afstand_km", "gebied"]].copy()
    info["org/des"] = info["org/des"].astype(str)

    verandering = kleur_keuze.startswith("Verandering")
    if verandering:
        # Voor de vergelijking tussen de jaren negeren we het jaarfilter
        per_jaar = zonder_jaar.groupby(["org/des", "jaar"], observed=True).size().unstack("jaar", fill_value=0)
        per_jaar = per_jaar.reindex(columns=[2019, 2020], fill_value=0)
        per_jaar.index = per_jaar.index.astype(str)
        stats = samenvatting(zonder_jaar, "org/des")
        stats["org/des"] = stats["org/des"].astype(str)
        stats["v2019"] = stats["org/des"].map(per_jaar[2019])
        stats["v2020"] = stats["org/des"].map(per_jaar[2020])
        stats = stats[stats["v2019"] >= minimum]
        stats["waarde"] = (stats["v2020"] - stats["v2019"]) / stats["v2019"] * 100
        stats["grootte"] = stats["v2019"]
        schaal, legenda_titel = SCHAAL_ROOD_BLAUW, "Verandering (%)"
        kleur_min, kleur_max = -100, 100
    else:
        stats = samenvatting(selectie, "org/des")
        stats["org/des"] = stats["org/des"].astype(str)
        stats = stats[stats["vluchten"] >= minimum]
        stats["waarde"] = stats["aandeel"] if kleur_keuze.startswith("Aandeel") else stats["gem"]
        stats["grootte"] = stats["vluchten"]
        schaal = SCHAAL_BLAUW
        legenda_titel = "% vertraagd" if kleur_keuze.startswith("Aandeel") else "Gem. vertraging (min)"
        kleur_min, kleur_max = None, None

    best = stats.merge(info, on="org/des", how="inner")
    best = best[(best["gebied"] == gebied) & best["latitude"].notna()].sort_values("grootte", ascending=False)
    if len(best) == 0:
        st.warning("Geen bestemmingen bij deze instellingen. Zet het minimum aantal vluchten lager.")
        return
    if kleur_min is None:
        kleur_min = float(np.floor(best["waarde"].quantile(0.05)))
        kleur_max = float(np.ceil(best["waarde"].quantile(0.95)))
        if kleur_min == kleur_max:
            kleur_max = kleur_min + 1

    fig = go.Figure()
    if gebied == "Intercontinentaal":
        # Een lijn per verbinding; None scheidt de lijnen binnen een trace
        lons, lats = [], []
        for _, rij in best.iterrows():
            lons += [ZURICH_LON, rij["longitude"], None]
            lats += [ZURICH_LAT, rij["latitude"], None]
        fig.add_trace(go.Scattergeo(lon=lons, lat=lats, mode="lines", line=dict(width=0.7, color="rgba(82,81,78,0.35)"),
                                    hoverinfo="skip", showlegend=False))
    grootste_stip = 34 if gebied == "Europa en omgeving" else 26
    fig.add_trace(go.Scattergeo(
        lon=best["longitude"], lat=best["latitude"], mode="markers", showlegend=False,
        marker=dict(
            size=best["grootte"], sizemode="area", sizeref=2.0 * best["grootte"].max() / grootste_stip ** 2, sizemin=4,
            color=best["waarde"], colorscale=schaal, cmin=kleur_min, cmax=kleur_max,
            colorbar=dict(title=legenda_titel, thickness=12, len=0.6), line=dict(color=BLAUW_DONKER, width=0.6),
            opacity=0.92,
        ),
        customdata=hoverdata(best, ["city", "name", "country", "vluchten", "aandeel", "gem", "afstand_km", "waarde"]),
        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>%{customdata[2]}, %{customdata[6]:,.0f} km<br>"
                      "%{customdata[3]:,} vluchten<br>%{customdata[4]:,.1f}% vertraagd · gemiddeld %{customdata[5]:,.1f} min"
                      + ("<br>Verandering 2020: %{customdata[7]:+,.0f}%" if verandering else "") + "<extra></extra>",
    ))
    # Alleen de grootste bestemmingen krijgen een naam, en alleen als er nog geen naam vlakbij staat.
    ruimte = 5.0 if gebied == "Europa en omgeving" else 14.0       # in graden
    met_naam = []
    for _, rij in best.iterrows():                                  # best is gesorteerd op grootte
        bij_zurich = abs(rij["latitude"] - ZURICH_LAT) < ruimte * 0.5 and abs(rij["longitude"] - ZURICH_LON) < ruimte * 0.8
        vrij = all(abs(rij["latitude"] - ander["latitude"]) > ruimte * 0.5
                   or abs(rij["longitude"] - ander["longitude"]) > ruimte for ander in met_naam)
        if vrij and not bij_zurich:
            met_naam.append(rij)
        if len(met_naam) == 14:
            break
    top = pd.DataFrame(met_naam) if met_naam else best.head(0)
    fig.add_trace(go.Scattergeo(lon=top["longitude"], lat=top["latitude"], mode="text", text=top["city"].astype(str),
                                textposition="top center", textfont=dict(size=11, color=INKT), hoverinfo="skip",
                                showlegend=False))
    fig.add_trace(go.Scattergeo(lon=[ZURICH_LON], lat=[ZURICH_LAT], mode="markers+text", text=["Zürich"],
                                textposition="bottom center", textfont=dict(size=12, color=INKT),
                                marker=dict(size=11, color=INKT, symbol="diamond", line=dict(color="white", width=1.5)),
                                hovertemplate="<b>Zürich Airport</b><extra></extra>", showlegend=False))
    opmaak(fig, hoogte=600 if gebied == "Europa en omgeving" else 470, legenda=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_geos(showland=True, landcolor=LAND, showcountries=True, countrycolor="white", coastlinecolor="#c9c7c0",
                    showocean=True, oceancolor="#f8fafc", showframe=False, resolution=50)
    if gebied == "Europa en omgeving":
        fig.update_geos(projection_type="mercator", lonaxis_range=[-32, 58], lataxis_range=[26, 67])
    else:
        fig.update_geos(projection_type="natural earth", lataxis_range=[-45, 75])
    toon(fig)
    uitleg_grootte = "vluchten in 2019" if verandering else "vluchten in de selectie"
    st.caption(f"Elke stip is een bestemming; hoe groter de stip, hoe meer {uitleg_grootte}. "
               f"Getoond: {len(best)} bestemmingen met minstens {minimum} vluchten. Zweef over een stip voor de cijfers.")

    # --- Onder de kaart: ranglijst en detail van een bestemming --------------
    links, rechts = st.columns([1, 1.2])
    with links:
        st.markdown("##### Uitschieters op de kaart")
        volgorde = st.radio("Sorteer", ["Hoogste eerst", "Laagste eerst"], horizontal=True,
                            label_visibility="collapsed", key="kaart_sorteer")
        lijst = best.sort_values("waarde", ascending=volgorde == "Laagste eerst").head(10)
        kolommen = ["city", "country", "vluchten", "aandeel", "gem"] + (["v2020", "waarde"] if verandering else [])
        lijst = lijst[kolommen].rename(columns={
            "city": "Stad", "country": "Land", "vluchten": "Vluchten", "aandeel": "% vertraagd", "gem": "Gem. min",
            "v2020": "Vluchten 2020", "waarde": "Verandering (%)"})
        if verandering:
            lijst = lijst.rename(columns={"Vluchten": "Vluchten 2019 + 2020"})
        tabel(lijst.round(1))
        st.caption(f"Gesorteerd op: {kleur_keuze.lower()}.")
    with rechts:
        st.markdown("##### Eén bestemming door de tijd")
        opties = best["org/des"].tolist()
        labels = dict(zip(best["org/des"], best["city"].astype(str) + " · " + best["name"].astype(str)))
        code = st.selectbox("Bestemming", opties, format_func=lambda c: labels[c],
                            label_visibility="collapsed", key="kaart_bestemming")
        route = zonder_jaar[zonder_jaar["org/des"].astype(str) == code]
        per_maand = route.groupby(pd.Grouper(key="datum", freq="MS")).agg(
            vluchten=("is_delayed", "size"), aandeel=("is_delayed", "mean"))
        per_maand = per_maand.reindex(pd.date_range("2019-01-01", "2020-12-01", freq="MS"), fill_value=0)
        fig = go.Figure(go.Scatter(x=per_maand.index, y=per_maand["vluchten"], mode="lines+markers",
                                   line=dict(color=BLAUW, width=2), marker=dict(size=6),
                                   hovertemplate="%{y:,} vluchten<extra></extra>"))
        opmaak(fig, y_titel="Vluchten per maand", hoogte=260, legenda=False)
        fig.update_layout(hovermode="x unified")
        fig.update_xaxes(hoverformat="%m-%Y", dtick="M6", tickformat="%m-%Y")
        lockdown_lijn(fig, "lockdown")
        toon(fig)
        rij = best[best["org/des"] == code].iloc[0]
        st.caption(f"{labels[code]}: {nl(rij['afstand_km'])} km, {nl(rij['vluchten'])} vluchten, "
                   f"{nl(rij['aandeel'], 1)}% vertraagd, gemiddeld {nl(rij['gem'], 1)} minuten.")


# ==========================================================================
# DAG VAN DE WEEK x UUR (HEATMAP)
# ==========================================================================
def toon_heatmap_dag_uur(df: pd.DataFrame) -> None:
    """Heatmap: aandeel vertraagde vluchten per dag van de week en gepland uur (06:00-22:00)."""
    selectie = _met_analysekolommen(df)

    st.markdown("#### Welke dag, welk uur?")
    per_cel = samenvatting(selectie, ["weekdag", "uur"])
    per_cel = per_cel[(per_cel["vluchten"] >= 30) & per_cel["uur"].between(6, 22)]
    raster = per_cel.pivot(index="weekdag", columns="uur", values="aandeel").reindex(index=range(7), columns=range(6, 23))
    aantallen = per_cel.pivot(index="weekdag", columns="uur", values="vluchten").reindex(index=range(7), columns=range(6, 23))
    fig = go.Figure(go.Heatmap(
        z=raster.to_numpy(dtype=float), x=[f"{u}:00" for u in raster.columns], y=WEEKDAGEN,
        customdata=aantallen.to_numpy(dtype=float),
        colorscale=SCHAAL_BLAUW, xgap=2, ygap=2, colorbar=dict(title="% vertraagd", thickness=12),
        hovertemplate="%{y} %{x}<br>%{z:,.1f}% vertraagd<br>%{customdata:,} vluchten<extra></extra>",
    ))
    opmaak(fig, hoogte=320, legenda=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    toon(fig)
    per_dag = samenvatting(selectie, "weekdag")
    best, slechtst = per_dag.loc[per_dag["aandeel"].idxmin()], per_dag.loc[per_dag["aandeel"].idxmax()]
    st.caption(
        f"Rustigste dag: {WEEKDAGEN[int(best['weekdag'])].lower()} ({nl(best['aandeel'], 1)}% vertraagd); "
        f"slechtste dag: {WEEKDAGEN[int(slechtst['weekdag'])].lower()} ({nl(slechtst['aandeel'], 1)}%). "
        "Vakjes met minder dan 30 vluchten blijven leeg."
    )


# ==========================================================================
# VERSCHILLEN TUSSEN TOESTELLEN, MAATSCHAPPIJEN EN BANEN
# ==========================================================================
def toon_vergelijking(df: pd.DataFrame) -> None:
    """Liggende staven: aandeel vertraagd per baan, vliegtuigtype of maatschappij."""
    selectie = _met_analysekolommen(df)

    st.markdown("#### Verschillen tussen toestellen, maatschappijen en banen")
    c1, c2 = st.columns([1, 2])
    soort = c1.selectbox("Vergelijk op", ["Baan", "Vliegtuigtype", "Maatschappij"], key="verg_soort")
    aantal = c2.slider("Toon de grootste", 5, 20, 12, key="verg_aantal", disabled=soort == "Baan",
                       help="Het aantal groepen met de meeste vluchten. Bij banen worden ze altijd allemaal getoond.")
    kolomnaam = {"Baan": "rwy", "Vliegtuigtype": "act", "Maatschappij": "maatschappij"}[soort]

    data = selectie.dropna(subset=[kolomnaam]).copy()
    if pd.api.types.is_numeric_dtype(data[kolomnaam]):
        # Banen worden als getal ingelezen (10.0); toon ze als baannummer (10)
        data[kolomnaam] = data[kolomnaam].round().astype(int).astype(str)
    per_groep = samenvatting(data, kolomnaam)
    per_groep = per_groep[per_groep["vluchten"] >= 100]
    if soort != "Baan":
        per_groep = per_groep.nlargest(aantal, "vluchten")
    per_groep = per_groep.sort_values("aandeel")
    if len(per_groep) == 0:
        st.info("Geen groepen met minstens 100 vluchten in deze selectie.")
        return

    fig = go.Figure(go.Bar(
        x=per_groep["aandeel"], y=per_groep[kolomnaam].astype(str), orientation="h",
        marker=dict(color=BLAUW), text=[nl(w, 1) + "%" for w in per_groep["aandeel"]], textposition="outside",
        cliponaxis=False, customdata=np.stack([per_groep["vluchten"], per_groep["gem"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>%{x:,.1f}% vertraagd<br>%{customdata[0]:,} vluchten<br>"
                      "gemiddeld %{customdata[1]:,.1f} min<extra></extra>",
    ))
    opmaak(fig, x_titel="Aandeel vertraagd (%)", hoogte=max(260, 30 * len(per_groep) + 80), legenda=False,
           marge_rechts=50)
    fig.update_yaxes(type="category", showgrid=False)
    toon(fig)

    hoogste, laagste = per_groep.iloc[-1], per_groep.iloc[0]
    st.caption(
        f"Hoogste: **{hoogste[kolomnaam]}** ({nl(hoogste['aandeel'], 1)}% vertraagd, {nl(hoogste['vluchten'])} vluchten); "
        f"laagste: **{laagste[kolomnaam]}** ({nl(laagste['aandeel'], 1)}%). Groepen met minder dan 100 vluchten zijn weggelaten. "
        "Let op: dit zijn samenhangen, geen oorzaken. Grote toestellen vliegen ver en vertrekken in de drukke middaggolf; "
        "welke baan in gebruik is hangt vooral af van de wind."
    )


# ==========================================================================
# PAGINA-OPMAAK: stijl, kop en kerncijfers
# ==========================================================================
# Kleine CSS-aanvulling op het thema in .streamlit/config.toml.
# Selectors (data-testid) uit de Streamlit-elementen; idee:
# https://docs.streamlit.io/develop/api-reference/text/st.markdown (unsafe_allow_html)
_STIJL = """
<style>
.block-container, [data-testid="stMainBlockContainer"] {padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1320px;}
h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.2rem;}
h3, h4 {letter-spacing: -0.01em;}
.kop-label {color: #2a78d6; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.08em;
            text-transform: uppercase; margin-bottom: 0.1rem;}
.kop-intro {color: #52596a; font-size: 1.02rem; max-width: 62rem; margin-bottom: 1.2rem;}
[data-testid="stMetric"] {background: #f3f5f8; border: 1px solid #e3e7ee; border-radius: 10px;
                          padding: 14px 16px;}
[data-testid="stMetricLabel"] p {color: #52596a; font-size: 0.85rem;}
[data-testid="stMetricValue"] {font-size: 1.7rem; font-weight: 600;}
[data-testid="stSidebar"] {border-right: 1px solid #e3e7ee;}
.stTabs [data-baseweb="tab-list"] {gap: 1.5rem; border-bottom: 1px solid #e3e7ee;}
.stTabs [data-baseweb="tab"] {font-size: 1rem; padding-left: 0; padding-right: 0;}
</style>
"""


def pagina_stijl() -> None:
    """Zet de extra opmaak op de pagina. Roep dit een keer aan, bovenaan app.py."""
    st.markdown(_STIJL, unsafe_allow_html=True)


def toon_kop() -> None:
    """Titel en korte uitleg bovenaan het dashboard."""
    st.markdown('<div class="kop-label">Case 3 · Zürich Airport · 2019–2020</div>', unsafe_allow_html=True)
    st.title("Wat voorspelt vertraging op Zürich Airport?")
    st.markdown(
        '<div class="kop-intro">Ruim 323.000 vluchten van en naar Zürich, gecombineerd met de '
        '<b>luchthavens</b> waar ze vandaan komen of naartoe gaan en met het <b>weer</b> van die dag. '
        'Geen van die bronnen laat op zichzelf zien waar vertraging vandaan komt; samen wel.</div>',
        unsafe_allow_html=True,
    )


def toon_kerncijfers(df: pd.DataFrame, jaar_keuze: str = "Beide jaren") -> None:
    """Rij met kerncijfers. Bij een gekozen jaar staat het verschil met het andere jaar eronder."""
    data = _met_analysekolommen(df)
    selectie = data if jaar_keuze == "Beide jaren" else data[data["jaar"] == int(jaar_keuze)]
    ander, ander_jaar = None, None
    if jaar_keuze != "Beide jaren":
        ander_jaar = 2020 if jaar_keuze == "2019" else 2019
        ander = data[data["jaar"] == ander_jaar]

    def verschil(nu, toen, eenheid="", decimalen=1):
        if ander is None or len(ander) == 0:
            return None
        plus = "+" if nu - toen > 0 else ""
        return f"{plus}{nl(nu - toen, decimalen)}{eenheid} t.o.v. {ander_jaar}"

    def waarde(deel, functie):
        return functie(deel) if deel is not None and len(deel) else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Vluchten", nl(len(selectie)),
              verschil(len(selectie), waarde(ander, len), "", 0), delta_color="off")
    k2.metric("Vertraagd (> 15 min)", nl(selectie["is_delayed"].mean() * 100, 1) + "%",
              verschil(selectie["is_delayed"].mean() * 100, waarde(ander, lambda d: d["is_delayed"].mean() * 100), " %-punt"),
              delta_color="inverse")
    k3.metric("Gemiddelde vertraging", nl(selectie["delay_minutes"].mean(), 1) + " min",
              verschil(selectie["delay_minutes"].mean(), waarde(ander, lambda d: d["delay_minutes"].mean()), " min"),
              delta_color="inverse")
    k4.metric("Bestemmingen", nl(selectie["org/des"].nunique()),
              verschil(selectie["org/des"].nunique(), waarde(ander, lambda d: d["org/des"].nunique()), "", 0),
              delta_color="off")

# ==========================================================================
# LIJNGRAFIEK OVER DE TIJD (2019 VS 2020)
# ==========================================================================
def toon_lijngrafiek_tijd(df: pd.DataFrame) -> None:
    """Lijngrafiek met het verloop van de vertraging per maand (2019 vs 2020)."""
    selectie = _met_analysekolommen(df)
    
    st.markdown("#### Vertraging over de tijd: 2019 vs 2020")
    
    # Per maand en jaar aggregeren
    maand_stats = selectie.groupby(["jaar", "maand"], observed=True).agg(
        aandeel=("is_delayed", "mean"),
        gem_min=("delay_minutes", "mean"),
        vluchten=("is_delayed", "size")
    ).reset_index()
    maand_stats["aandeel"] = maand_stats["aandeel"] * 100
    
    fig = go.Figure()
    kleuren = {2019: BLAUW, 2020: ORANJE}
    
    maanden_namen = ["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dec"]
    
    for yr in [2019, 2020]:
        sub = maand_stats[maand_stats["jaar"] == yr]
        fig.add_trace(go.Scatter(
            x=sub["maand"], y=sub["aandeel"],
            mode="lines+markers",
            name=str(yr),
            line=dict(color=kleuren.get(yr, BLAUW), width=2.5),
            marker=dict(size=7),
            customdata=np.stack([sub["vluchten"], sub["gem_min"]], axis=-1),
            hovertemplate=f"<b>{yr} - %{{x}}</b><br>%{{y:,.1f}}% vertraagd<br>%{{customdata[0]:,}} vluchten<br>Gem. %{{customdata[1]:,.1f}} min<extra></extra>"
        ))
        
    opmaak(fig, x_titel="Maand", y_titel="% Vertraagd (>15 min)", hoogte=350)
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)), ticktext=maanden_namen)
    toon(fig)
    st.caption("Vergelijking van het aandeel vertraagde vluchten per maand in het normale jaar 2019 t.o.v. het coronajaar 2020.")


# ==========================================================================
# VOORSPELMODEL (FEATURE IMPORTANCE & EVALUATIE)
# ==========================================================================
def toon_voorspelmodel(model_data: pd.DataFrame) -> None:
    """Traint een Random Forest model en toont feature importances en scores."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    st.markdown("#### Machine Learning: Wat veroorzaakt vertraging?")
    st.caption("Een Random Forest Classifier voorspelt of een vlucht meer dan 15 minuten vertraging oploopt op basis van geplande kenmerken en het weer.")
    
    df = _met_analysekolommen(model_data).copy()
    
    # Gebruik de daadwerkelijk aanwezige kolommen in de dataset (tavg en prcp i.p.v. temp/rhum)
    mogelijke_features = ["maand", "weekdag", "uur", "afstand_km", "tavg", "prcp", "wspd", "pres"]
    feature_cols = [col for col in mogelijke_features if col in df.columns]
    
    if "is_delayed" not in df.columns or not feature_cols:
        st.error("De vereiste kolommen voor het voorspelmodel ontbreken in de dataset.")
        return

    # Alleen rijen gebruiken waar de geselecteerde features compleet zijn
    schoon = df.dropna(subset=feature_cols + ["is_delayed"]).copy()
    
    if len(schoon) < 1000:
        st.warning("Te weinig data beschikbaar om het voorspelmodel betrouwbaar te trainen.")
        return

    X = schoon[feature_cols]
    y = schoon["is_delayed"].astype(int)
    
    @st.cache_resource(show_spinner="Model trainen op vlucht- en weerdata...")
    def train_model(X_train, y_train):
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        return rf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = train_model(X_train, y_train)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    # Top metrieken
    m1, m2, m3 = st.columns(3)
    m1.metric("Model Nauwkeurigheid (Accuracy)", f"{acc * 100:.1f}%")
    m2.metric("ROC-AUC Score", f"{auc:.3f}")
    m3.metric("Aantal gebruikte vluchten", f"{len(schoon):,}".replace(",", "."))
    
    st.write("")
    st.markdown("##### Welke factoren wegen het zwaarst? (Feature Importance)")
    
    # Nederlandse namen voor de weergave
    naam_mapping = {
        "maand": "Maand",
        "weekdag": "Weekdag",
        "uur": "Uur van vertrek",
        "afstand_km": "Afstand (km)",
        "tavg": "Gem. Temperatuur (°C)",
        "prcp": "Neerslag (mm)",
        "wspd": "Windsnelheid",
        "pres": "Luchtdruk"
    }
    
    importances = pd.DataFrame({
        "Feature": [naam_mapping.get(col, col) for col in feature_cols],
        "Belangrijkheid": model.feature_importances_
    }).sort_values("Belangrijkheid", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=importances["Belangrijkheid"],
        y=importances["Feature"],
        orientation="h",
        marker=dict(color=BLAUW)
    ))
    opmaak(fig, x_titel="Relatief belang van de feature in de voorspelling", hoogte=320, legenda=False)
    toon(fig)
    
    st.caption(
        "**Conclusie model:** Bovenstaande factoren tonen aan welk gewicht het model toekent aan vluchtkenmerken en weersomstandigheden bij het voorspellen van vertraging."
    )