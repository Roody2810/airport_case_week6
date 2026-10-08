
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==================================================
# 1. PAGINA-INSTELLINGEN
# ==================================================

st.set_page_config(
    page_title="Zürich Airport | Vluchten en vertraging",
    layout="wide"
)

st.title("Vluchten en vertraging op Zürich Airport")
st.write(
    "Wanneer ontstaat vertraging en welke factoren "
    "hangen ermee samen? | 2019–2020"
)


# ==================================================
# 2. DATA INLADEN
# ==================================================

@st.cache_data
def laad_data():
    df = pd.read_csv("schedule_airport.csv")

    df["gepland"] = pd.to_datetime(
        df["STD"].astype(str) + " " +
        df["STA_STD_ltc"].astype(str),
        dayfirst=True,
        errors="coerce"
    )

    df["werkelijk"] = pd.to_datetime(
        df["STD"].astype(str) + " " +
        df["ATA_ATD_ltc"].astype(str),
        dayfirst=True,
        errors="coerce"
    )

    verschil_uren = (
        df["werkelijk"] - df["gepland"]
    ).dt.total_seconds() / 3600

    # Correctie voor vluchten rond middernacht
    df.loc[
        verschil_uren <= -12, "werkelijk"
    ] += pd.Timedelta(days=1)

    df["vertraging_minuten"] = (
        df["werkelijk"] - df["gepland"]
    ).dt.total_seconds() / 60

    df["jaar"] = df["gepland"].dt.year
    df["maand"] = df["gepland"].dt.month
    df["uur"] = df["gepland"].dt.hour

    df["dagdeel"] = pd.cut(
        df["uur"],
        bins=[-1, 11, 17, 23],
        labels=["Ochtend", "Middag", "Avond"]
    )

    return df


@st.cache_data
def laad_luchthavens():
    luchthavens = pd.read_csv(
        "airports-extended.csv",
        header=None
    )

    luchthavens.columns = [
        "Airport_ID", "Naam", "Stad", "Land",
        "IATA", "ICAO", "Latitude", "Longitude",
        "Altitude", "Timezone", "DST",
        "Tz_database", "Type", "Source"
    ]

    luchthavens["Latitude"] = pd.to_numeric(
        luchthavens["Latitude"], errors="coerce"
    )

    luchthavens["Longitude"] = pd.to_numeric(
        luchthavens["Longitude"], errors="coerce"
    )

    return luchthavens.drop_duplicates(subset=["ICAO"])


vluchten = laad_data()
luchthavens = laad_luchthavens()


# ==================================================
# 3. FILTERS
# ==================================================

maand_namen = {
    1: "Januari",
    2: "Februari",
    3: "Maart",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Augustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "December"
}

with st.sidebar:
    st.header("Filters")

    jaar_keuze = st.selectbox(
        "Jaar",
        ["Alle jaren", 2019, 2020]
    )

    maand_keuze = st.selectbox(
        "Maand",
        ["Alle maanden"] + list(maand_namen.values())
    )

    type_keuze = st.selectbox(
        "Type vlucht",
        ["Alle vluchten", "Aankomst (L)", "Vertrek (S)"]
    )

    st.caption(
        "Filters gelden voor de grafieken, kaart en "
        "dynamische conclusies. De voorspellingsanalyse "
        "gebruikt altijd de volledige periode."
    )


# ==================================================
# 4. DATA FILTEREN
# ==================================================

gefilterde_data = vluchten.copy()

if jaar_keuze != "Alle jaren":
    gefilterde_data = gefilterde_data[
        gefilterde_data["jaar"] == jaar_keuze
    ]

if maand_keuze != "Alle maanden":
    gekozen_maand = next(
        nummer
        for nummer, naam in maand_namen.items()
        if naam == maand_keuze
    )

    gefilterde_data = gefilterde_data[
        gefilterde_data["maand"] == gekozen_maand
    ]

if type_keuze == "Aankomst (L)":
    gefilterde_data = gefilterde_data[
        gefilterde_data["LSV"] == "L"
    ]

elif type_keuze == "Vertrek (S)":
    gefilterde_data = gefilterde_data[
        gefilterde_data["LSV"] == "S"
    ]

if gefilterde_data.empty:
    st.warning(
        "Geen vluchten gevonden. Pas je filters aan."
    )
    st.stop()


# ==================================================
# 5. KPI'S
# ==================================================

gemiddelde_filter = (
    gefilterde_data["vertraging_minuten"].mean()
)

aandeel_vertraagd = (
    gefilterde_data["vertraging_minuten"] > 15
).mean()

kolom1, kolom2, kolom3 = st.columns(3)

with kolom1:
    st.metric(
        "Aantal vluchten",
        f"{len(gefilterde_data):,}"
    )

with kolom2:
    st.metric(
        "Gemiddeld tijdsverschil",
        f"{gemiddelde_filter:.1f} min"
    )

with kolom3:
    st.metric(
        "Meer dan 15 minuten later",
        f"{aandeel_vertraagd:.1%}"
    )

st.caption(
    "Positief tijdsverschil = later dan gepland. "
    "Negatief tijdsverschil = eerder dan gepland."
)


# ==================================================
# 6. TABBLADEN
# ==================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Vertraging",
    "🕒 Tijdstip & type",
    "🌍 Bestemmingen",
    "🔎 Datakwaliteit & conclusies"
])


# ==================================================
# TAB 1 - VERTRAGING
# ==================================================

with tab1:

    st.subheader("Hoe verandert vertraging door de tijd?")

    # Gemiddelde vertraging per maand
    vertraging_per_maand = (
        gefilterde_data
        .groupby(["jaar", "maand"])["vertraging_minuten"]
        .mean()
        .reset_index(name="gemiddelde_vertraging")
    )

    vertraging_per_maand["jaar"] = (
        vertraging_per_maand["jaar"].astype(str)
    )

    fig_maand = px.line(
        vertraging_per_maand,
        x="maand",
        y="gemiddelde_vertraging",
        color="jaar",
        markers=True,
        title="Gemiddeld tijdsverschil per maand",
        labels={
            "maand": "Maand",
            "gemiddelde_vertraging": "Minuten",
            "jaar": "Jaar"
        }
    )

    fig_maand.update_xaxes(dtick=1)

    st.plotly_chart(
        fig_maand,
        width="stretch",
        key="maand_vertraging"
    )

    # Dynamische conclusie
    if not vertraging_per_maand.empty:
        hoogste_maand = vertraging_per_maand.loc[
            vertraging_per_maand[
                "gemiddelde_vertraging"
            ].idxmax()
        ]

        st.info(
            f"Binnen de huidige selectie was "
            f"{maand_namen[int(hoogste_maand['maand'])]} "
            f"{hoogste_maand['jaar']} de maand met het "
            f"hoogste gemiddelde tijdsverschil: "
            f"{hoogste_maand['gemiddelde_vertraging']:.1f} minuten."
        )

    # --------------------------------------------------
    # DRUKTE VERSUS VERTRAGING
    # --------------------------------------------------

    drukte_vertraging = (
        gefilterde_data
        .groupby(["jaar", "maand"])
        .agg(
            aantal_vluchten=("LSV", "size"),
            gemiddelde_vertraging=(
                "vertraging_minuten", "mean"
            )
        )
        .reset_index()
    )

    drukte_vertraging["jaar"] = (
        drukte_vertraging["jaar"].astype(str)
    )

    fig_drukte = px.scatter(
        drukte_vertraging,
        x="aantal_vluchten",
        y="gemiddelde_vertraging",
        color="jaar",
        hover_data=["maand"],
        title="Aantal vluchten versus gemiddeld tijdsverschil",
        labels={
            "aantal_vluchten": "Aantal vluchten per maand",
            "gemiddelde_vertraging": "Minuten",
            "jaar": "Jaar"
        }
    )

    st.plotly_chart(
        fig_drukte,
        width="stretch",
        key="drukte_vertraging"
    )

    st.caption(
        "Iedere stip is één maand. Een verband tussen "
        "drukte en vertraging betekent niet automatisch "
        "dat drukte de vertraging veroorzaakt."
    )

    # --------------------------------------------------
    # VOORSPELLING 2020
    # --------------------------------------------------

    st.subheader("Voorspelling voor 2020 op basis van 2019")

    trend_data = (
        vluchten
        .groupby(["jaar", "maand"])["vertraging_minuten"]
        .mean()
        .reset_index(name="gemiddelde_vertraging")
    )

    trend_data = trend_data.sort_values(
        ["jaar", "maand"]
    ).copy()

    # Doorlopende tijdas:
    # januari 2019 = 1, december 2019 = 12
    # januari 2020 = 13, december 2020 = 24
    trend_data["tijd_index"] = (
        (trend_data["jaar"] - 2019) * 12
        + trend_data["maand"]
    )

    trend_2019 = trend_data[
        trend_data["jaar"] == 2019
    ].copy()

    trend_2020 = trend_data[
        trend_data["jaar"] == 2020
    ].copy()

    if len(trend_2019) >= 2 and not trend_2020.empty:

        helling, startwaarde = np.polyfit(
            trend_2019["tijd_index"],
            trend_2019["gemiddelde_vertraging"],
            1
        )

        trend_2020["voorspelde_vertraging"] = (
            helling * trend_2020["tijd_index"]
            + startwaarde
        )

        vergelijking_2020 = trend_2020[
            [
                "maand",
                "gemiddelde_vertraging",
                "voorspelde_vertraging"
            ]
        ].melt(
            id_vars="maand",
            var_name="type",
            value_name="vertraging"
        )

        vergelijking_2020["type"] = (
            vergelijking_2020["type"].replace({
                "gemiddelde_vertraging":
                    "Werkelijk 2020",
                "voorspelde_vertraging":
                    "Voorspeld op basis van 2019"
            })
        )

        fig_voorspelling = px.line(
            vergelijking_2020,
            x="maand",
            y="vertraging",
            color="type",
            markers=True,
            title="Werkelijke versus voorspelde vertraging in 2020",
            labels={
                "maand": "Maand",
                "vertraging": "Gemiddelde vertraging (minuten)",
                "type": ""
            }
        )

        fig_voorspelling.update_xaxes(dtick=1)

        st.plotly_chart(
            fig_voorspelling,
            width="stretch",
            key="voorspelling_2020"
        )

        mae = (
            trend_2020["gemiddelde_vertraging"]
            - trend_2020["voorspelde_vertraging"]
        ).abs().mean()

        st.metric(
            "Gemiddelde voorspellingsfout (MAE)",
            f"{mae:.2f} minuten"
        )

        st.info(
            "Het model is getraind op de twaalf "
            "maandgemiddelden van 2019. De lineaire trend "
            "wordt doorgetrokken naar de maanden van 2020. "
            "De MAE geeft aan hoeveel minuten de voorspelling "
            "gemiddeld afwijkt van het werkelijke maandgemiddelde. "
            "Het model houdt geen rekening met de "
            "coronapandemie of seizoenseffecten."
        )

        st.caption(
            "Let op: deze versie gebruikt een doorlopende "
            "tijdas van maand 1 tot en met 24. Daardoor kan "
            "de MAE afwijken van je eerdere berekening van "
            "9,32 minuten, die de maandnummers 1–12 hergebruikte."
        )


# ==================================================
# TAB 2 - TIJDSTIP & TYPE
# ==================================================

with tab2:

    st.subheader("Wanneer ontstaat de meeste vertraging?")

    # --------------------------------------------------
    # AANKOMST VERSUS VERTREK
    # --------------------------------------------------

    vertraging_per_type = (
        gefilterde_data
        .groupby(["jaar", "LSV"])["vertraging_minuten"]
        .mean()
        .reset_index(name="gemiddelde_vertraging")
    )

    vertraging_per_type["jaar"] = (
        vertraging_per_type["jaar"].astype(str)
    )

    fig_type = px.bar(
        vertraging_per_type,
        x="LSV",
        y="gemiddelde_vertraging",
        color="jaar",
        barmode="group",
        title="Gemiddeld tijdsverschil bij aankomst en vertrek",
        labels={
            "LSV": "L = aankomst, S = vertrek",
            "gemiddelde_vertraging": "Minuten",
            "jaar": "Jaar"
        }
    )

    st.plotly_chart(
        fig_type,
        width="stretch",
        key="aankomst_vertrek"
    )

    # --------------------------------------------------
    # DAGDEEL
    # --------------------------------------------------

    vertraging_per_dagdeel = (
        gefilterde_data
        .groupby(
            ["jaar", "dagdeel"],
            observed=True
        )["vertraging_minuten"]
        .mean()
        .reset_index(name="gemiddelde_vertraging")
    )

    vertraging_per_dagdeel["jaar"] = (
        vertraging_per_dagdeel["jaar"].astype(str)
    )

    fig_dagdeel = px.bar(
        vertraging_per_dagdeel,
        x="dagdeel",
        y="gemiddelde_vertraging",
        color="jaar",
        barmode="group",
        category_orders={
            "dagdeel": [
                "Ochtend",
                "Middag",
                "Avond"
            ]
        },
        title="Gemiddeld tijdsverschil per dagdeel",
        labels={
            "dagdeel": "Dagdeel",
            "gemiddelde_vertraging": "Minuten",
            "jaar": "Jaar"
        }
    )

    st.plotly_chart(
        fig_dagdeel,
        width="stretch",
        key="dagdeel_vertraging"
    )

    st.caption(
        "Ochtend: 00:00–11:59 | "
        "Middag: 12:00–17:59 | "
        "Avond: 18:00–23:59"
    )

    # Dynamische conclusie
    if not vertraging_per_dagdeel.empty:
        hoogste_dagdeel = vertraging_per_dagdeel.loc[
            vertraging_per_dagdeel[
                "gemiddelde_vertraging"
            ].idxmax()
        ]

        st.info(
            f"Binnen de huidige selectie had "
            f"{hoogste_dagdeel['dagdeel']} "
            f"in {hoogste_dagdeel['jaar']} "
            f"het hoogste gemiddelde tijdsverschil: "
            f"{hoogste_dagdeel['gemiddelde_vertraging']:.1f} minuten."
        )


# ==================================================
# TAB 3 - BESTEMMINGEN
# ==================================================

with tab3:

    st.subheader("Waar ontstaan verschillen tussen luchthavens?")

    # --------------------------------------------------
    # VLIEGRICHTING BEPALEN
    # --------------------------------------------------

    if type_keuze == "Aankomst (L)":
        kaart_vluchten = gefilterde_data[
            gefilterde_data["LSV"] == "L"
        ].copy()

        kaart_titel = "Herkomstluchthavens van vluchten naar Zürich"
        richting_uitleg = "aankomende vluchten"

    elif type_keuze == "Vertrek (S)":
        kaart_vluchten = gefilterde_data[
            gefilterde_data["LSV"] == "S"
        ].copy()

        kaart_titel = "Bestemmingen van vluchten vanuit Zürich"
        richting_uitleg = "vertrekkende vluchten"

    else:
        richting_keuze = st.radio(
            "Welke vliegrichting wil je op de kaart zien?",
            ["Vertrek vanuit Zürich", "Aankomst in Zürich"],
            horizontal=True
        )

        if richting_keuze == "Vertrek vanuit Zürich":
            kaart_vluchten = gefilterde_data[
                gefilterde_data["LSV"] == "S"
            ].copy()

            kaart_titel = "Bestemmingen van vluchten vanuit Zürich"
            richting_uitleg = "vertrekkende vluchten"

        else:
            kaart_vluchten = gefilterde_data[
                gefilterde_data["LSV"] == "L"
            ].copy()

            kaart_titel = "Herkomstluchthavens van vluchten naar Zürich"
            richting_uitleg = "aankomende vluchten"

    st.write(
        f"De kaart toont **{richting_uitleg}**. "
        "De grootte van de bolletjes geeft het aantal "
        "vluchten aan. De kleur geeft het gemiddelde "
        "tijdsverschil in minuten aan."
    )

    # --------------------------------------------------
    # VLUCHTEN KOPPELEN AAN LUCHTHAVENS
    # --------------------------------------------------

    vluchten_met_luchthaven = kaart_vluchten.merge(
        luchthavens[
            [
                "ICAO",
                "Naam",
                "Stad",
                "Land",
                "Latitude",
                "Longitude"
            ]
        ],
        left_on="Org/Des",
        right_on="ICAO",
        how="left"
    )

    # --------------------------------------------------
    # GEGEVENS PER LUCHTHAVEN
    # --------------------------------------------------

    bestemmingen = (
        vluchten_met_luchthaven
        .dropna(subset=["Latitude", "Longitude"])
        .groupby(
            [
                "ICAO",
                "Naam",
                "Stad",
                "Land",
                "Latitude",
                "Longitude"
            ]
        )
        .agg(
            aantal_vluchten=("LSV", "size"),
            gemiddelde_vertraging=(
                "vertraging_minuten", "mean"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------
    # KAARTFILTERS
    # --------------------------------------------------

    min_vluchten = st.slider(
        "Minimaal aantal vluchten per luchthaven",
        min_value=1,
        max_value=1000,
        value=100,
        step=25
    )

    kaartgebied = st.radio(
        "Kaartgebied",
        ["Europa", "Wereld"],
        horizontal=True
    )

    kaart_data = bestemmingen[
        bestemmingen["aantal_vluchten"] >= min_vluchten
    ].copy()

    if kaartgebied == "Europa":
        kaart_data = kaart_data[
            kaart_data["Latitude"].between(34, 72)
            & kaart_data["Longitude"].between(-12, 45)
        ]

    # --------------------------------------------------
    # INTERACTIEVE KAART
    # --------------------------------------------------

    if kaart_data.empty:
        st.info(
            "Geen luchthavens gevonden met deze instellingen. "
            "Verlaag het minimumaantal vluchten."
        )

    else:
        fig_kaart = px.scatter_map(
            kaart_data,
            lat="Latitude",
            lon="Longitude",
            size="aantal_vluchten",
            color="gemiddelde_vertraging",
            color_continuous_scale="RdYlGn_r",
            size_max=28,
            hover_name="Naam",
            hover_data={
                "Stad": True,
                "Land": True,
                "aantal_vluchten": True,
                "gemiddelde_vertraging": ":.1f",
                "Latitude": False,
                "Longitude": False
            },
            zoom=3 if kaartgebied == "Europa" else 1,
            center=(
                {"lat": 49, "lon": 12}
                if kaartgebied == "Europa"
                else {"lat": 25, "lon": 0}
            ),
            title=kaart_titel,
            labels={
                "aantal_vluchten": "Aantal vluchten",
                "gemiddelde_vertraging":
                    "Gem. tijdsverschil (min)"
            }
        )

        fig_kaart.update_layout(
            height=550,
            coloraxis_colorbar=dict(
                title="Minuten"
            ),
            margin=dict(
                l=0,
                r=0,
                t=50,
                b=0
            )
        )

        st.plotly_chart(
            fig_kaart,
            width="stretch",
            key="bestemmingen_kaart"
        )

        st.caption(
            "Grotere bol = meer vluchten. "
            "Groen = lager gemiddeld tijdsverschil. "
            "Rood = hoger gemiddeld tijdsverschil. "
            "Beweeg over een bolletje voor details."
        )

    # --------------------------------------------------
    # TOP 10 LUCHTHAVENS
    # --------------------------------------------------

    top10_bestemmingen = (
        bestemmingen[
            bestemmingen["aantal_vluchten"] >= 500
        ]
        .nlargest(
            10,
            "gemiddelde_vertraging"
        )
        .sort_values("gemiddelde_vertraging")
    )

    if not top10_bestemmingen.empty:

        fig_top10 = px.bar(
            top10_bestemmingen,
            x="gemiddelde_vertraging",
            y="Naam",
            orientation="h",
            hover_data=[
                "Stad",
                "Land",
                "aantal_vluchten"
            ],
            title=(
                "Top 10 luchthavens met het hoogste "
                "gemiddelde tijdsverschil"
            ),
            labels={
                "gemiddelde_vertraging": "Minuten",
                "Naam": "Luchthaven"
            }
        )

        fig_top10.update_layout(height=500)

        st.plotly_chart(
            fig_top10,
            width="stretch",
            key="top10_bestemmingen"
        )

        # Dynamische conclusie
        hoogste_bestemming = (
            top10_bestemmingen.iloc[-1]
        )

        st.info(
            f"Bij de geselecteerde {richting_uitleg} "
            f"had {hoogste_bestemming['Naam']} "
            f"het hoogste gemiddelde tijdsverschil "
            f"van de luchthavens met minimaal "
            f"500 vluchten: "
            f"{hoogste_bestemming['gemiddelde_vertraging']:.1f} "
            f"minuten."
        )

    else:
        st.info(
            "Er zijn geen luchthavens met minimaal "
            "500 vluchten binnen deze selectie."
        )

    st.caption(
        "De top 10 gebruikt minimaal 500 vluchten "
        "per luchthaven. De kaartfilter verandert "
        "de top 10 niet."
    )


# ==================================================
# TAB 4 - DATAKWALITEIT & CONCLUSIES
# ==================================================

with tab4:

    st.subheader("Datakwaliteit")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Ontbrekende geplande tijden",
            f"{vluchten['gepland'].isna().sum():,}"
        )

    with c2:
        st.metric(
            "Ontbrekende werkelijke tijden",
            f"{vluchten['werkelijk'].isna().sum():,}"
        )

    with c3:
        extreme_waarden = (
            vluchten["vertraging_minuten"].abs() > 180
        ).sum()

        st.metric(
            "Extreme tijdsverschillen (>180 min)",
            f"{extreme_waarden:,}"
        )

    st.write("Ontbrekende waarden per kolom:")

    ontbrekende_waarden = (
        vluchten
        .isna()
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.dataframe(
        ontbrekende_waarden.rename("Aantal ontbrekend")
    )

    st.warning(
        "Extreme waarden zijn niet automatisch fouten. "
        "De middernachtcorrectie is een aanname die "
        "bij uitzonderlijke vluchten gecontroleerd "
        "moet worden."
    )

    # --------------------------------------------------
    # DYNAMISCHE CONCLUSIES
    # --------------------------------------------------

    st.subheader("Conclusies voor jouw selectie")

    st.write(
        f"**Geselecteerde periode:** {jaar_keuze}, "
        f"{maand_keuze.lower()}, {type_keuze.lower()}."
    )

    st.write(
        f"Er zijn **{len(gefilterde_data):,} vluchten** "
        f"geselecteerd. Het gemiddelde tijdsverschil "
        f"is **{gemiddelde_filter:.1f} minuten**."
    )

    st.write(
        f"Van deze vluchten was **{aandeel_vertraagd:.1%}** "
        f"meer dan 15 minuten later dan gepland."
    )

    # Vergelijk aankomst en vertrek binnen de selectie
    type_gemiddelden = (
        gefilterde_data
        .groupby("LSV")["vertraging_minuten"]
        .mean()
    )

    if "L" in type_gemiddelden and "S" in type_gemiddelden:

        aankomst_gem = type_gemiddelden["L"]
        vertrek_gem = type_gemiddelden["S"]

        st.write(
            f"**Aankomst versus vertrek:** aankomende "
            f"vluchten hebben gemiddeld "
            f"{aankomst_gem:.1f} minuten tijdsverschil. "
            f"Bij vertrekkende vluchten is dit "
            f"{vertrek_gem:.1f} minuten."
        )

        if vertrek_gem > aankomst_gem:
            st.write(
                "Binnen deze selectie ligt het "
                "gemiddelde bij vertrekken hoger."
            )
        elif aankomst_gem > vertrek_gem:
            st.write(
                "Binnen deze selectie ligt het "
                "gemiddelde bij aankomsten hoger."
            )
        else:
            st.write(
                "Binnen deze selectie zijn de "
                "gemiddelden gelijk."
            )

    # Hoogste dagdeel
    dagdeel_gemiddelden = (
        gefilterde_data
        .groupby("dagdeel", observed=True)[
            "vertraging_minuten"
        ]
        .mean()
        .dropna()
    )

    if not dagdeel_gemiddelden.empty:

        drukste_dagdeel = dagdeel_gemiddelden.idxmax()
        hoogste_dagdeel_vertraging = (
            dagdeel_gemiddelden.max()
        )

        st.write(
            f"**Dagdeel:** {drukste_dagdeel} heeft "
            f"binnen de selectie het hoogste "
            f"gemiddelde tijdsverschil: "
            f"{hoogste_dagdeel_vertraging:.1f} minuten."
        )

    st.subheader("Algemene conclusie")

    st.write(
        "De analyse laat zien dat tijdstip, "
        "vluchtrichting en luchthaven samenhangen "
        "met verschillen in gemiddelde vertraging. "
        "Ook zijn er duidelijke verschillen tussen "
        "2019 en 2020."
    )

    st.write(
        "De eenvoudige voorspelling op basis van "
        "2019 kan de uitzonderlijke omstandigheden "
        "van 2020 niet volledig verklaren."
    )

    st.caption(
        "De gevonden verbanden bewijzen geen "
        "oorzakelijke relaties."
    )

    st.caption(
        "Bronnen: schedule_airport.csv en "
        "airports-extended.csv."
    )




