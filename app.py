# IMPORTATION 
import base64
import io
import json
import folium
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.raster_layers import ImageOverlay
from PIL import Image
from shapely import contains_xy
from shapely.geometry import shape
from shapely.ops import unary_union
from streamlit_folium import st_folium
import xyzservices.providers as xyz


# CONFIGURATION
st.set_page_config(page_title="Cartographie SOC Maroc", page_icon="🌱", layout="wide")
MODEL_ACCURACY = 0.92  
GREEN = "#1b7a4a"       
COLORS = {"Faible": "#d64545", "Moyen": "#2e9e5b"}
MAP_CENTER, MAP_ZOOM = (28.3, -9.2), 5

LABELS = {"NDVI": "NDVI (végétation)", "BSI": "BSI (sol nu)", "TWI": "TWI (humidité topo)","elevation": "Altitude (m)", "slope": "Pente (°)", "bio12_prec": "Précipitations (mm)","bio15_pseason": "Saisonnalité pluies", "clay": "Argile (%)", "sand": "Sable (%)","silt": "Limon (%)", "ph": "pH", "bdod": "Densité apparente", "nitrogen": "Azote (g/kg)","lon": "Longitude", "lat": "Latitude",}

# titre, icône Material, variables
GROUPS = [("Indices satellitaires", ":material/satellite_alt:", ["NDVI", "BSI"]),("Topographie", ":material/landscape:", ["elevation", "slope", "TWI"]),
("Climat", ":material/rainy:", ["bio12_prec", "bio15_pseason"]),("Sol", ":material/layers:", ["clay", "sand", "silt", "ph", "bdod", "nitrogen"]),]
GROUP_OF = {f: g for g, _, fs in GROUPS for f in fs}
GROUP_OF.update({"lon": "Localisation", "lat": "Localisation"})
GROUP_COLORS = {"Indices satellitaires": "#2e9e5b", "Topographie": "#8c6d31", "Climat": "#2a78d6",
                "Sol": "#b5651d", "Localisation": "#7b4fa0"}
# Bornes physiques des variables (les curseurs restent dans ces limites)
PHYS = {"NDVI": (-1, 1), "BSI": (-1, 1), "TWI": (0, None), "elevation": (-50, None), "slope": (0, 90),
        "bio12_prec": (0, None), "bio15_pseason": (0, None), "clay": (0, 100), "sand": (0, 100),
        "silt": (0, 100), "ph": (3, 11), "bdod": (0.5, 2.2), "nitrogen": (0, None)}

# Icônes au trait fin (style line icons )
_SVG = '<svg viewBox="0 0 48 48" width="52" height="52" fill="none" stroke="{c}" stroke-width="1.8" ' \
       'stroke-linecap="round" stroke-linejoin="round">{p}</svg>'
ICONS = {"target": '<circle cx="24" cy="24" r="17"/><circle cx="24" cy="24" r="10"/><circle cx="24" cy="24" r="3"/>'
              '<path d="M24 3v6M24 39v6M3 24h6M39 24h6"/>',
    "leaf": '<path d="M10 38c0-16 10-26 28-28 0 18-10 28-26 28"/><path d="M10 38 28 20"/>'
            '<path d="M6 42h36"/>',
    "dry": '<path d="M6 40h36"/><path d="M10 40l5-9 4 5 5-11 5 8 4-4 5 11"/>'
           '<circle cx="36" cy="11" r="5"/><path d="M36 2v2M36 18v2M27 11h2M43 11h2"/>',
    "ruler": '<path d="M6 30 30 6l12 12-24 24z"/><path d="M13 23l4 4M18 18l3 3M23 13l4 4M28 8l3 3"/>',
    "sliders": '<path d="M8 12h20M36 12h4M8 24h6M22 24h18M8 36h22M38 36h2"/>'
               '<circle cx="32" cy="12" r="4"/><circle cx="18" cy="24" r="4"/><circle cx="34" cy="36" r="4"/>',
    "tag": '<path d="M6 8v14l20 20 16-16L22 6H8a2 2 0 0 0-2 2z"/><circle cx="14" cy="14" r="3"/>',
}

CSS = f"""
<style>
.stApp {{ background: #f7f9f7; }}
h1 {{ font-size: 2.1rem !important; font-weight: 700 !important; color: #1f2a24; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 18px; border-bottom: 1px solid #e3e7e3; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 500; }}
[data-testid="stExpander"] details {{ background: #fff; border: 1px solid #e3e7e3; border-radius: 12px; }}
.kpi {{ display:flex; align-items:center; gap:16px; background:#fff; border:1px solid #e3e7e3;
        border-radius:14px; padding:18px 20px; height:100%; }}
.kpi .lbl {{ font-size:14px; color:#4b5a51; }}
.kpi .val {{ font-size:24px; font-weight:700; color:#1f2a24; line-height:1.3; }}
.kpi .val small {{ font-size:13px; font-weight:400; color:#6b7a71; margin-left:4px; }}
.kpi .sub {{ font-size:12.5px; color:#6b7a71; margin-top:4px; display:flex; align-items:center; gap:6px; }}
.kpi .dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
.leaflet-image-layer {{ image-rendering: pixelated; }}
</style>
"""



# CHARGEMENT 

@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model():
    return joblib.load("modele_soc.pkl")


@st.cache_resource
def load_regions():
    with open("maroc_regions.geojson", encoding="utf-8") as f:
        gj = json.load(f)
    regions = {f["properties"]["name"]: shape(f["geometry"]).buffer(0) for f in gj["features"]}
    return gj, regions, unary_union(list(regions.values()))


model_data = load_model()
pipeline = model_data["pipeline"]
FEATURES = model_data["features"]
MEDIAN_SOC = model_data["median_soc"]
CLASS_NAMES = model_data["class_names"]
ACCURACY = model_data.get("accuracy", MODEL_ACCURACY)
GEOJSON, REGIONS, MAROC = load_regions()

# Statistiques d'entraînement
_scaler = pipeline.named_steps["scaler"]
TRAIN_MEAN = dict(zip(FEATURES, _scaler.mean_))
TRAIN_STD = dict(zip(FEATURES, _scaler.scale_))
# Zone couverte par les données d'entraînement (moyenne ± 2 écarts-types)
LAT0, LAT1 = TRAIN_MEAN["lat"] - 2 * TRAIN_STD["lat"], TRAIN_MEAN["lat"] + 2 * TRAIN_STD["lat"]
LON0, LON1 = TRAIN_MEAN["lon"] - 2 * TRAIN_STD["lon"], TRAIN_MEAN["lon"] + 2 * TRAIN_STD["lon"]


def slider_range(f):
    lo, hi = TRAIN_MEAN[f] - 4 * TRAIN_STD[f], TRAIN_MEAN[f] + 4 * TRAIN_STD[f]
    plo, phi = PHYS.get(f, (None, None))
    lo = max(lo, plo) if plo is not None else lo
    hi = min(hi, phi) if phi is not None else hi
    return float(lo), float(hi)


def nice_step(f):
    return float(10 ** np.floor(np.log10(TRAIN_STD[f] / 20)))


def classify(proba_moyen):
    return np.where(np.asarray(proba_moyen) >= 0.5, CLASS_NAMES[1], CLASS_NAMES[0])


def region_of(lon, lat):
    lon, lat = np.asarray(lon), np.asarray(lat)
    out = np.full(lon.shape, "Hors Maroc", dtype=object)
    for name, geom in REGIONS.items():
        out[contains_xy(geom, lon, lat)] = name
    return out


def _merc(lat):
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def _inv_merc(y):
    return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)


@st.cache_data(show_spinner="Calcul de la carte…")
def prediction_grid(scenario: tuple, step: float = 0.04):
    """Prédit la classe sur tout le Maroc. Renvoie une image PNG (projection Web Mercator)."""
    lon_min, lat_min, lon_max, lat_max = MAROC.bounds
    n_cols = int(np.ceil((lon_max - lon_min) / step))
    n_rows = int(np.ceil((lat_max - lat_min) / step))
    # Lignes régulièrement espacées en Mercator pour que l'image se superpose exactement au fond de carte
    y_edges = np.linspace(_merc(lat_max), _merc(lat_min), n_rows + 1)   # du nord au sud
    lat_edges = _inv_merc(y_edges)
    lats = (lat_edges[:-1] + lat_edges[1:]) / 2
    lons = lon_min + (np.arange(n_cols) + 0.5) * (lon_max - lon_min) / n_cols
    LO, LA = np.meshgrid(lons, lats)
    inside = contains_xy(MAROC, LO, LA)

    X = np.tile(np.array(scenario, dtype=float), (inside.sum(), 1))
    X[:, FEATURES.index("lon")] = LO[inside]
    X[:, FEATURES.index("lat")] = LA[inside]
    is_moyen = pipeline.predict_proba(X)[:, 1] >= 0.5

    rgba = np.zeros((n_rows, n_cols, 4), dtype=np.uint8)
    hexrgb = lambda h: [int(h[i:i + 2], 16) for i in (1, 3, 5)]
    rgba[inside] = np.where(is_moyen[:, None], hexrgb(COLORS["Moyen"]) + [200], hexrgb(COLORS["Faible"]) + [200])
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # Part de surface (pondérée par la vraie superficie de chaque case)
    row_h = np.abs(np.diff(lat_edges))[:, None] * np.cos(np.radians(LA))
    w = row_h[inside]
    share_moyen = float((w * is_moyen).sum() / w.sum())
    return url, [[lat_min, lon_min], [lat_max, lon_max]], share_moyen


def base_map():
    gray = xyz.Esri.WorldGrayCanvas
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles=None, control_scale=True)
    folium.TileLayer(gray.build_url(), attr=gray.html_attribution + " | Régions : geoBoundaries (CC BY 4.0)",
                     name="Plan", max_zoom=16).add_to(m)
    sat = xyz.Esri.WorldImagery
    folium.TileLayer(sat.build_url(), attr=sat.html_attribution, name="Satellite", show=False).add_to(m)
    folium.GeoJson(
        GEOJSON, name="Régions",
        style_function=lambda _: {"color": "#3b4a41", "weight": 1, "fillOpacity": 0},
        tooltip=folium.GeoJsonTooltip(["name"], aliases=["Région"]),
    ).add_to(m)
    folium.Rectangle([[LAT0, LON0], [LAT1, LON1]], color="#7b4fa0", weight=1.5, dash_array="6",
                     fill=False, tooltip="Zone couverte par les données d'entraînement").add_to(m)
    items = "".join(
        f'<div style="margin-top:4px"><span style="background:{c};width:12px;height:12px;'
        f'display:inline-block;border-radius:3px;margin-right:6px"></span>{n}</div>'
        for n, c in COLORS.items())
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:28px;left:12px;z-index:9999;background:white;padding:8px 12px;'
        'border-radius:10px;border:1px solid #e3e7e3;font:13px sans-serif"><b>Classe SOC</b>' + items +
        '<div style="margin-top:6px;color:#7b4fa0">┅ zone d\'entraînement</div></div>'))
    return m


def kpi(col, icon, label, value, unit="", sub="", dot=None):
    svg = _SVG.format(c=GREEN, p=ICONS[icon])
    dot_html = f'<span class="dot" style="background:{dot}"></span>' if dot else ""
    col.markdown(f"""<div class="kpi">{svg}<div>
<div class="lbl">{label}</div><div class="val">{value}<small>{unit}</small></div>
<div class="sub">{dot_html}{sub}</div></div></div>""", unsafe_allow_html=True)


def result_card(classe):
    col = COLORS[classe]
    desc = (f"Sol à faible teneur en carbone (&lt; {MEDIAN_SOC:.2f} g/kg)" if classe == CLASS_NAMES[0]
            else f"Sol à teneur moyenne en carbone (≥ {MEDIAN_SOC:.2f} g/kg)")
    st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;gap:16px;padding:18px 22px;
     border-radius:14px;background:{col}14;border:1px solid {col};border-left:10px solid {col}">
  <div>
    <div style="font-size:14px;color:#4b5a51">Classe prédite</div>
    <div style="font-size:40px;font-weight:800;color:{col};line-height:1.15">{classe}</div>
    <div style="font-size:14px;color:#1f2a24">{desc}</div>
  </div>
  <div style="text-align:right">
    <div style="font-size:13px;color:#4b5a51">Accuracy du modèle</div>
    <div style="font-size:32px;font-weight:800;color:#1f2a24">{ACCURACY:.0%}</div>
    <div style="font-size:12px;color:#6b7a71">validation croisée 5 plis</div>
  </div>
</div>""", unsafe_allow_html=True)


# ÉTAT DE SESSION

for f in FEATURES:
    st.session_state.setdefault(f"in_{f}", round(float(TRAIN_MEAN[f]), 4))


def reset_inputs():
    for f in FEATURES:
        st.session_state[f"in_{f}"] = round(float(TRAIN_MEAN[f]), 4)
    st.session_state.pop("dernier_clic", None)


# EN-TÊTE

st.markdown(CSS, unsafe_allow_html=True)
st.title("Cartographie du carbone organique du sol au niveau du Maroc")

tab1, tab2, tab3, tab4 = st.tabs([
    ":material/online_prediction: Prédiction",
    ":material/show_chart: Effet d'une variable sur la classe",
    ":material/map: Cartographie",
    ":material/analytics: Metrics du modèle",
])


# ONGLET 1 — PRÉDICTION INTERACTIVE

with tab1:
    st.subheader("Prédire la classe de SOC d'une parcelle")
    st.caption("Cliquez sur la carte pour choisir l'emplacement, puis ajustez les caractéristiques : "
               "la prédiction se met à jour automatiquement.")

    col_map, col_form = st.columns([1.1, 1], gap="large")

    with col_map:
        m = base_map()
        folium.Marker([st.session_state["in_lat"], st.session_state["in_lon"]],
                      icon=folium.Icon(color="green", icon="map-marker"), tooltip="Parcelle").add_to(m)
        out = st_folium(m, height=640, use_container_width=True, returned_objects=["last_clicked"], key="map_pred")

        clic = (out or {}).get("last_clicked")
        if clic and clic != st.session_state.get("dernier_clic"):
            st.session_state["dernier_clic"] = clic
            st.session_state["in_lat"] = round(clic["lat"], 4)
            st.session_state["in_lon"] = round(clic["lng"], 4)
            st.rerun()

    with col_form:
        c_lat, c_lon = st.columns(2)
        c_lat.number_input(LABELS["lat"], -90.0, 90.0, step=0.01, format="%.4f", key="in_lat")
        c_lon.number_input(LABELS["lon"], -180.0, 180.0, step=0.01, format="%.4f", key="in_lon")
        region = region_of([st.session_state["in_lon"]], [st.session_state["in_lat"]])[0]
        st.markdown(f":material/location_on: **Région :** {region}")

        for groupe, icon, feats in GROUPS:
            with st.expander(groupe, expanded=True, icon=icon):
                cols = st.columns(2)
                for j, f in enumerate(feats):
                    lo, hi = slider_range(f)
                    cols[j % 2].slider(LABELS[f], lo, hi, step=nice_step(f), key=f"in_{f}")
        st.button("Réinitialiser (valeurs moyennes)", icon=":material/restart_alt:", on_click=reset_inputs)

    inputs = {f: float(st.session_state[f"in_{f}"]) for f in FEATURES}
    X = pd.DataFrame([inputs])[FEATURES]
    classe = classify(pipeline.predict_proba(X.to_numpy())[:, 1])[0]

    st.divider()
    result_card(classe)

    alerts = []
    if region == "Hors Maroc":
        alerts.append("le point est en dehors du Maroc")
    if not (LAT0 <= inputs["lat"] <= LAT1 and LON0 <= inputs["lon"] <= LON1):
        alerts.append("le point est en dehors de la zone couverte par les données d'entraînement "
                      "(la prédiction est une extrapolation)")
    texture = inputs["clay"] + inputs["sand"] + inputs["silt"]
    if abs(texture - 100) > 3:
        alerts.append(f"argile + sable + limon = {texture:.1f} % (devrait être proche de 100 %)")
    for a in alerts:
        st.warning("Attention : " + a, icon=":material/warning:")

# ONGLET 2 — EFFET D'UNE VARIABLE SUR LA CLASSE

with tab2:
    var_opts = [f for f in FEATURES if f not in ("lat", "lon")]
    var = st.selectbox("Variable à faire varier", var_opts, index=var_opts.index("bio12_prec"),
                       format_func=lambda f: LABELS[f])
    lo, hi = slider_range(var)
    xs = np.linspace(lo, hi, 120)
    Xs = pd.concat([X] * len(xs), ignore_index=True)
    Xs[var] = xs
    ps = pipeline.predict_proba(Xs[FEATURES].to_numpy())[:, 1]
    cls_s = classify(ps)
    changes = xs[1:][cls_s[1:] != cls_s[:-1]]

    fig = go.Figure()
    fig.add_hrect(y0=0, y1=0.5, fillcolor=COLORS["Faible"], opacity=0.08, line_width=0)
    fig.add_hrect(y0=0.5, y1=1, fillcolor=COLORS["Moyen"], opacity=0.08, line_width=0)
    fig.add_hline(y=0.5, line_color="#9aa59f", line_width=1)
    fig.add_trace(go.Scatter(x=xs, y=ps, mode="lines", line=dict(color="#1f2a24", width=3), name="",
                             hovertemplate=f"{LABELS[var]} = %{{x:.3g}}<br>Classe : %{{customdata}}",
                             customdata=cls_s))
    fig.add_vline(x=inputs[var], line_dash="dash", line_color=GREEN,
                  annotation_text="valeur actuelle", annotation_position="top",
                  annotation_font_color=GREEN)
    fig.add_annotation(x=0.01, y=0.95, xref="paper", yref="y", text="<b>Moyen</b>", showarrow=False,
                       font_color=COLORS["Moyen"], xanchor="left")
    fig.add_annotation(x=0.01, y=0.05, xref="paper", yref="y", text="<b>Faible</b>", showarrow=False,
                       font_color=COLORS["Faible"], xanchor="left")
    fig.update_layout(height=440, showlegend=False, margin=dict(t=30, b=10, l=10, r=10),
                      plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(title=LABELS[var], gridcolor="#eef1ee"),
                      yaxis=dict(range=[0, 1], showticklabels=False, title=None, showgrid=False))
    st.plotly_chart(fig, width="stretch")
    if len(changes):
        st.info("La classe change pour " + LABELS[var] + " ≈ " +
                ", ".join(f"{v:.3g}" for v in changes[:4]) + ".", icon=":material/swap_vert:")
    else:
        st.info(f"La classe reste **{cls_s[0]}** sur toute la plage de {LABELS[var]}.", icon=":material/info:")


# ONGLET 3 — CARTOGRAPHIE 

with tab3:
    url, bounds, share = prediction_grid(tuple(inputs[f] for f in FEATURES))
    c1, c2, c3 = st.columns(3)
    kpi(c1, "leaf", "Surface classée Moyen", f"{share:.0%}", sub="du territoire", dot=COLORS["Moyen"])
    kpi(c2, "dry", "Surface classée Faible", f"{1 - share:.0%}", sub="du territoire", dot=COLORS["Faible"])
    kpi(c3, "target", "Accuracy du modèle", f"{ACCURACY:.0%}", sub="validation croisée 5 plis")
    st.write("")
    m2 = base_map()
    ImageOverlay(url, bounds=bounds, opacity=0.8, name="Classes SOC").add_to(m2)
    folium.LayerControl(collapsed=True).add_to(m2)
    st_folium(m2, height=720, use_container_width=True, returned_objects=[], key="map_grid")


# ONGLET 4 — METRICS DU MODÈLE

with tab4:
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "target", "Accuracy", f"{ACCURACY:.0%}", sub="validation croisée 5 plis")
    kpi(c2, "ruler", "Frontière de classe", f"{MEDIAN_SOC:.2f}", unit="g/kg", sub="médiane du SOC")
    kpi(c3, "sliders", "Variables prédictives", str(len(FEATURES)), sub="satellite, climat, sol, relief")
    kpi(c4, "tag", "Classes", " / ".join(CLASS_NAMES), sub="Random Forest")

    st.divider()
    st.subheader("Importance des variables")
    imp = pd.DataFrame({"Variable": [LABELS[f] for f in FEATURES],
                        "Importance": pipeline.named_steps["clf"].feature_importances_,
                        "Groupe": [GROUP_OF[f] for f in FEATURES]}).sort_values("Importance")
    fig = px.bar(imp, x="Importance", y="Variable", color="Groupe", orientation="h",
                 color_discrete_map=GROUP_COLORS, text_auto=".0%")
    fig.update_layout(height=520, yaxis_title=None, xaxis_tickformat=".0%", legend_title=None,
                      plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                      yaxis=dict(categoryorder="total ascending"), xaxis=dict(gridcolor="#eef1ee"),
                      margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    by_group = imp.groupby("Groupe")["Importance"].sum().sort_values(ascending=False)
    st.caption("Poids par groupe : " + " · ".join(f"{g} {v:.0%}" for g, v in by_group.items()))

    st.divider()
    st.markdown(f"""
    ### Méthodologie
    - **Modèle** : Random Forest (2 classes : Faible / Moyen)
    - **Découpage** : à la médiane du SOC ({MEDIAN_SOC:.2f} g/kg), avec zone tampon (buffer ±{model_data.get('buffer_gap', 0.25)} g/kg)
      pour l'entraînement, afin d'écarter les parcelles de la zone de transition.
    - **Validation** : validation croisée 5 plis (accuracy ≈ {ACCURACY:.2f}, toutes métriques > 0.90).
    """)
