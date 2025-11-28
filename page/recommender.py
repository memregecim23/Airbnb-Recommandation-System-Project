import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import re, io, zipfile

# ---------- Sayfa ----------
st.set_page_config(page_title="Airbnb Akıllı Öneri Haritası", layout="wide")
st.title("Airbnb Akıllı Öneri Haritası")
st.caption("Seçtiğin sütunlara göre filtrele, haritada göster, hover ile ayrıntıları gör.")

# ---------- Sabit harita ayarları ----------
POINT_RADIUS = 60
OPACITY = 0.80

# ---------- Yardımcı Fonksiyonlar ----------
def canon_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())

@st.cache_data(show_spinner=False)
def read_csv_any(uploaded_file) -> pd.DataFrame:
    """CSV veya ZIP içinden CSV okur (latin1 fallback, sıkıştırma otomatik)."""
    name = (uploaded_file.name or "").lower()
    data = uploaded_file.getvalue()

    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("ZIP içinde CSV bulunamadı.")
            with z.open(csv_names[0]) as fh:
                try:
                    return pd.read_csv(fh, low_memory=False)
                except UnicodeDecodeError:
                    fh.seek(0)
                    return pd.read_csv(fh, low_memory=False, encoding="latin1")
    else:
        # Düz .csv veya .csv.gz
        bio = io.BytesIO(data)
        try:
            return pd.read_csv(bio, low_memory=False, compression="infer")
        except UnicodeDecodeError:
            bio.seek(0)
            return pd.read_csv(bio, low_memory=False, encoding="latin1", compression="infer")

def detect_lat_lon_cols(df):
    lat_candidates = ["latitude", "lat", "Latitude", "LATITUDE", "centroid_lat"]
    lon_candidates = ["longitude", "lon", "lng", "long", "Longitude", "LONGITUDE", "centroid_lon"]
    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    return lat_col, lon_col

def to_int_safe(x):
    try:
        return int(float(str(x).replace(",", ".").strip()))
    except:
        return None

def to_float_safe(x):
    try:
        s = str(x)
        s = re.sub(r"[^\d\.\-]", "", s)
        return float(s)
    except:
        return None

def price_display(v):
    try:
        f = to_float_safe(v)
        if f is None or np.isnan(f):
            return str(v)
        return f"{int(f)}" if abs(f - int(f)) < 1e-9 else f"{f:.2f}"
    except:
        return str(v)

# ---------- Veri Yükleme ----------
file = st.file_uploader("CSV/ZIP yükle", type=["csv", "zip", "gz"])
if not file:
    st.stop()

df = read_csv_any(file)
if df.empty:
    st.error("Yüklenen veri boş görünüyor.")
    st.stop()

# minimum_nights her zaman 1, filtrede görünmez (21)
if "minimum_nights" in df.columns:
    df["minimum_nights"] = 1

# maximum_nights 0–1125 integer; filtrelerde varsayılan seçilmesin (22,28)
if "maximum_nights" in df.columns:
    df["maximum_nights"] = pd.to_numeric(df["maximum_nights"], errors="coerce").fillna(0).clip(0, 1125).astype(int)

# Lat/Lon (31)
lat_col, lon_col = detect_lat_lon_cols(df)
if lat_col is None or lon_col is None:
    st.error("Lat/Lon sütunları bulunamadı.")
    st.stop()

# ---------- Coğrafi + Tür Filtreleri ----------
st.subheader("🔎 Filtreler")
country_col = next((c for c in ["country", "Country", "COUNTRY"] if c in df.columns), None)
city_col    = next((c for c in ["city", "City", "CITY"] if c in df.columns), None)
neigh_col   = next((c for c in ["neighbourhood", "neighborhood", "neighbourhood_cleansed", "neighborhood_cleansed"] if c in df.columns), None)
room_type_col     = next((c for c in ["room_type", "roomtype", "Room Type", "RoomType"] if c in df.columns), None)
property_type_col = next((c for c in ["property_type", "property type", "Property Type", "PropertyType"] if c in df.columns), None)

mask = pd.Series(True, index=df.index)
any_filter_used = False          # genel
other_filter_used = False        # "Diğer filtrelenecek sütunlar" için

geo1, geo2, geo3 = st.columns(3)
sel_countries, sel_cities, sel_neighs = [], [], []

# Country → City → Neighbourhood zinciri (2–3)
if country_col:
    opts = sorted(df[country_col].dropna().astype(str).unique().tolist())
    sel_countries = geo1.multiselect("Country", options=opts)
    if sel_countries:
        mask &= df[country_col].astype(str).isin(sel_countries); any_filter_used=True

base_city = df[df[country_col].astype(str).isin(sel_countries)] if (country_col and sel_countries) else df
if city_col:
    opts = sorted(base_city[city_col].dropna().astype(str).unique().tolist())
    sel_cities = geo2.multiselect("City", options=opts)
    if sel_cities:
        mask &= df[city_col].astype(str).isin(sel_cities); any_filter_used=True

base_neigh = base_city[base_city[city_col].astype(str).isin(sel_cities)] if (city_col and sel_cities) else base_city
if neigh_col:
    opts = sorted(base_neigh[neigh_col].dropna().astype(str).unique().tolist())
    sel_neighs = geo3.multiselect("Neighbourhood", options=opts)
    if sel_neighs:
        mask &= df[neigh_col].astype(str).isin(sel_neighs); any_filter_used=True

# Oda & Mülk tipi (4)
rt_col, pt_col = st.columns(2)
if room_type_col:
    rt_opts = sorted(df[room_type_col].dropna().astype(str).unique().tolist())
    sel_room_types = rt_col.multiselect("room_type", options=rt_opts)
    if sel_room_types:
        mask &= df[room_type_col].astype(str).isin(sel_room_types); any_filter_used=True

if property_type_col:
    pt_opts = sorted(df[property_type_col].dropna().astype(str).unique().tolist())
    sel_property_types = pt_col.multiselect("property_type", options=pt_opts)
    if sel_property_types:
        mask &= df[property_type_col].astype(str).isin(sel_property_types); any_filter_used=True

# ---------- Diğer Filtreler ----------
HIDE_FROM_CHOOSER = {"name", "host_since", "listing_id", "id", "minimum_nights"}  # (6,21)

# (19) filtre listesinden çıkarılacaklar + (23,24,25)
EXCLUDE_CANON = {
    "hosthasprofilepic", "hostidentityverified", "hostissuperhost", "hostissuperhostt",
    "reviewscorescheckin", "reviewscorescommunication", "reviewscoreslocation",
    "reviewscoresrating", "reviewscoresvalue", "reviewscoresaccuracy",
    "hosttotallistingscount", "hosttotallistingcount", "hostlistingscount",
    "customscore", "reviewcleanlinessscore"
}

exclude = {country_col, city_col, neigh_col, room_type_col, property_type_col, lat_col, lon_col}
exclude |= HIDE_FROM_CHOOSER

filterable_cols = []
for c in df.columns:
    if c is None or c in exclude:
        continue
    if canon_name(c) in EXCLUDE_CANON:
        continue
    filterable_cols.append(c)

# Başlangıçta hiçbir şey seçili olmasın (18)
chosen_filters = st.multiselect("Diğer filtrelenecek sütunlar", options=sorted(filterable_cols), default=[])

col1, col2 = st.columns(2)
for i, c in enumerate(chosen_filters):
    tgt = col1 if i % 2 == 0 else col2
    cn = canon_name(c)

    # (29) instant_bookable → checkbox (seçiliyse=1)
    if cn == "instantbookable":
        raw = df[c].astype(str).str.strip().str.lower()
        mapping = {"t":1,"true":1,"yes":1,"y":1,"1":1, "f":0,"false":0,"no":0,"n":0,"0":0}
        s = raw.map(mapping)
        s = s.where(s.notna(), pd.to_numeric(df[c], errors="coerce")).fillna(0).astype(int)
        chk = tgt.checkbox(c, value=False)
        mask &= (s == (1 if chk else 0)); any_filter_used=True; other_filter_used=True
        continue

    # -------- (1) Range slider'lar (etiketsiz açıklama) --------
    if cn == "bedrooms":
        s = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        lo, hi = tgt.slider(c, min_value=0, max_value=30, value=(0, 30), step=1)
        mask &= s.between(lo, hi); any_filter_used=True; other_filter_used=True
        continue

    if cn in {"staycount", "staycount_"}:
        s = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(0, 20).astype(int)
        lo, hi = tgt.slider(c, min_value=0, max_value=20, value=(0, 20), step=1)
        mask &= s.between(lo, hi); any_filter_used=True; other_filter_used=True
        continue

    if cn == "accommodates":
        s = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        amin, amax = int(s.min()), int(s.max())
        lo, hi = tgt.slider(c, min_value=amin, max_value=amax, value=(amin, amax), step=1)
        mask &= s.between(lo, hi); any_filter_used=True; other_filter_used=True
        continue

    if cn == "price":
        s_float = pd.to_numeric(df[c].apply(to_float_safe), errors="coerce")
        pmin = int(np.nanmin(s_float)) if s_float.notna().any() else 0
        pmax = int(np.nanmax(s_float)) if s_float.notna().any() else 0
        lo, hi = tgt.slider(c, min_value=pmin, max_value=pmax, value=(pmin, pmax), step=1)
        mask &= s_float.between(lo, hi); any_filter_used=True; other_filter_used=True
        continue
    # -----------------------------------------------------------

    # (16,26) 0/1 kolonlar → Var/Yok
    if set(pd.to_numeric(df[c], errors="coerce").dropna().unique().tolist()).issubset({0,1}):
        s = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        choice = tgt.radio(c, options=["Var", "Yok"], horizontal=True, index=0)
        mask &= (s == (1 if choice == "Var" else 0)); any_filter_used=True; other_filter_used=True
        continue

    # bool kolonlar → Var/Yok
    if pd.api.types.is_bool_dtype(df[c]):
        s = df[c].astype("boolean")
        choice = tgt.radio(c, options=["Var", "Yok"], horizontal=True, index=0)
        mask &= (s == (True if choice == "Var" else False)); any_filter_used=True; other_filter_used=True
        continue

    # Genel sayısal (tek değer, ≤) — diğerleri
    if pd.api.types.is_numeric_dtype(df[c]):
        s = pd.to_numeric(df[c], errors="coerce")
        dmin, dmax = s.min(), s.max()
        if pd.isna(dmin) or pd.isna(dmax):
            tgt.info(f"'{c}' için sayısal aralık tespit edilemedi.")
            continue
        dmin_i, dmax_i = int(np.floor(dmin)), int(np.ceil(dmax))
        val = tgt.number_input(c, min_value=dmin_i, max_value=dmax_i, value=dmax_i, step=1)
        mask &= (s <= val); any_filter_used=True; other_filter_used=True
        continue

    # Kategorik — combobox (multiselect)
    uniq = pd.Series(df[c].dropna().unique())
    if len(uniq) > 2000:
        tgt.info(f"'{c}' çok fazla farklı değer içeriyor ({len(uniq)}); atlandı.")
        continue
    opts = sorted(uniq.astype(str).tolist())
    choice = tgt.multiselect(c, options=opts)
    if choice:
        mask &= df[c].astype(str).isin(choice); any_filter_used=True; other_filter_used=True

# ---------- Filtrelerin altında GÖRÜNTÜLE butonu (2) ----------
if "recs_go" not in st.session_state:
    st.session_state["recs_go"] = False

clicked = st.button("Görüntüle")
if clicked:
    # "Diğer filtrelenecek sütunlar"dan en az bir seçim yapılmışsa etkinleştir
    st.session_state["recs_go"] = len(chosen_filters) > 0

# ---------- Filtrelenmiş Veri ----------
filtered = df[mask].copy()
filtered = filtered.dropna(subset=[lat_col, lon_col])
filtered[lat_col] = pd.to_numeric(filtered[lat_col], errors="coerce")
filtered[lon_col] = pd.to_numeric(filtered[lon_col], errors="coerce")
filtered = filtered.dropna(subset=[lat_col, lon_col])

if len(filtered) == 0:
    st.warning("Aradığınız özellikte ev bulunmamaktadır.")
    st.stop()

# ---------- Tooltip ----------
forced_tooltip_cols = [c for c in ["room_type", "property_type"] if c in df.columns]
allowed_tooltip_cols = [c for c in [
    "name","host_has_profile_pic","host_identity_verified","review_scores_accuracy",
    "review_scores_cleanliness","review_cleanliness_score","review_scores_checkin",
    "review_scores_communication","review_scores_location","host_is_superhost_t",
    "custom_score"
] if c in df.columns]

with st.sidebar:
    st.subheader("Tooltip Alanları")
    hover_extra = st.multiselect(
        "Hover'da gösterilecek ek sütunlar",
        options=allowed_tooltip_cols,
        default=["name"] if "name" in allowed_tooltip_cols else []
    )

tooltip_cols = list(dict.fromkeys(forced_tooltip_cols + hover_extra))

# Mevki etiketi (city+neighbourhood) — sadece tooltip için birleştir
location_col_for_tooltip = None
if (city_col or neigh_col):
    join_name = "__location_join__"
    cols_to_join = [c for c in [city_col, neigh_col] if c]
    if cols_to_join:
        filtered[join_name] = filtered[cols_to_join].astype(str).agg(" / ".join, axis=1)
        location_col_for_tooltip = join_name

def build_tooltip_html(loc_col, hover_cols):
    lines = []
    if loc_col: lines.append(f"<b>Konum:</b> {{{loc_col}}}")
    for c in hover_cols: lines.append(f"<b>{c}:</b> {{{c}}}")
    return "<br/>".join(lines)

tooltip_html = build_tooltip_html(location_col_for_tooltip, tooltip_cols)

# ---------- Öneriler: custom_score top-5 (numaralı) ----------
show_recs = False
rec_df = pd.DataFrame()
if "custom_score" in filtered.columns:
    scores = pd.to_numeric(filtered["custom_score"], errors="coerce")
    tmp = filtered.copy()
    tmp["__custom_score_num__"] = scores
    tmp = tmp.dropna(subset=["__custom_score_num__"]).sort_values("__custom_score_num__", ascending=False)
    rec_df = tmp.head(5).reset_index(drop=True)
    if st.session_state.get("recs_go", False) and len(chosen_filters) > 0 and not rec_df.empty:
        show_recs = True
        rec_df["__rank__"] = np.arange(1, len(rec_df)+1)  # 1..5

# ---- Beğenebileceğiniz evler (3: kartta 'Evi görüntüle' butonu) ----
if show_recs:
    st.markdown("### Beğenebileceğiniz evler")

    st.markdown(
        """
        <style>
          .like-card {border:1px solid #e6e6e6; border-radius: 18px; padding: 12px 16px; margin: 10px 0; background:#fff; color:#000;}
          .like-title {font-weight: 700; font-size: 18px; margin-bottom: 6px; color:#000;}
          .like-meta {font-size: 13px; color:#000; margin-bottom: 8px;}
          .like-badges {display:flex; gap:8px; flex-wrap:wrap; margin-bottom: 8px;}
          .like-badge {padding: 6px 10px; border-radius: 999px; background:#f5f7ff; border:1px solid #e7eaff; font-size:12px; color:#000;}
          .like-scores {margin: 6px 0 0 0; padding: 0; list-style: none; font-size: 13px; color:#000;}
          .like-scores li {margin: 2px 0; color:#000;}
          .rank-pill {display:inline-block; padding:2px 8px; border-radius:999px; background:#0a5ff2; color:#fff; font-weight:700; margin-right:8px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    review_fields = [
        "review_scores_rating",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value",
    ]

    for idx, row in rec_df.iterrows():
        header_col, btn_col = st.columns([1, 0.18])

        with header_col:
            name = row.get("name") if "name" in rec_df.columns and pd.notna(row.get("name")) else "Listing"
            price_txt = price_display(row.get("price")) if "price" in rec_df.columns and pd.notna(row.get("price")) else "-"
            cs_txt = f"{float(row['__custom_score_num__']):.2f}" if pd.notna(row.get("__custom_score_num__")) else "-"
            badges = []
            if "room_type" in rec_df.columns and pd.notna(row.get("room_type")): badges.append(str(row.get("room_type")))
            if "property_type" in rec_df.columns and pd.notna(row.get("property_type")): badges.append(str(row.get("property_type")))
            badges_html = "".join([f"<span class='like-badge'>{b}</span>" for b in badges])

            score_lines = []
            for f in review_fields:
                if f in rec_df.columns and pd.notna(row.get(f)):
                    score_lines.append(f"<li>{f}: <b>{row.get(f)}</b></li>")

            rank = int(row["__rank__"]) if "__rank__" in row else (idx+1)
            card_html = f"""
            <div class='like-card'>
                <div class='like-title'><span class='rank-pill'>{rank}</span>{name}</div>
                <div class='like-meta'>price: <b>{price_txt}</b> • custom_score: <b>{cs_txt}</b></div>
                <div class='like-badges'>{badges_html}</div>
                <ul class='like-scores'>{''.join(score_lines)}</ul>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

        with btn_col:
            if st.button("Evi görüntüle", key=f"view_{idx+1}"):
                st.session_state["focus_lat"] = float(row[lat_col])
                st.session_state["focus_lon"] = float(row[lon_col])
                st.session_state["focus_zoom"] = 16.0

# ---------- Harita Merkezi ----------
center_lat = float(filtered[lat_col].mean())
center_lon = float(filtered[lon_col].mean())
zoom = 10
if len(sel_neighs) > 0:
    zoom = 13
elif len(sel_cities) > 0:
    zoom = 10
elif len(sel_countries) > 0:
    zoom = 6

# "Evi görüntüle" tıklandıysa o ilana odakla
if "focus_lat" in st.session_state and "focus_lon" in st.session_state:
    center_lat = st.session_state["focus_lat"]
    center_lon = st.session_state["focus_lon"]
    zoom = st.session_state.get("focus_zoom", 16.0)

# ---------- Harita ----------
map_cols = [lon_col, lat_col]
if location_col_for_tooltip: map_cols.append(location_col_for_tooltip)
map_cols += tooltip_cols
map_cols = list(dict.fromkeys(map_cols))
map_data = filtered[map_cols].copy()
map_data["emoji"] = "🏡"

layers = [
    pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position=[lon_col, lat_col],
        get_radius=POINT_RADIUS,
        get_fill_color=[255, 99, 64, int(255*OPACITY)],
        pickable=True,
        auto_highlight=True
    ),
    pdk.Layer(
        "TextLayer",
        data=map_data,
        get_position=[lon_col, lat_col],
        get_text="emoji",
        get_size=16,
        get_color=[20,20,20,220],
        get_alignment_baseline="'bottom'",
        pickable=True
    )
]

# Önerileri haritada mavi nokta + ortasında numara; highlight kapalı, ek halka yok (4)
if show_recs:
    rec_map_cols = [lon_col, lat_col]
    if location_col_for_tooltip and location_col_for_tooltip in rec_df.columns:
        rec_map_cols.append(location_col_for_tooltip)
    for c in tooltip_cols:
        if c in rec_df.columns and c not in rec_map_cols:
            rec_map_cols.append(c)
    rec_map = rec_df[rec_map_cols].copy()
    rec_map["rank_label"] = rec_df["__rank__"].astype(str).values

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=rec_map,
            get_position=[lon_col, lat_col],
            get_radius=int(POINT_RADIUS*1.6),
            get_fill_color=[0,122,255,180],
            pickable=True,
            auto_highlight=False  # hover/click büyütme yok
        )
    )
    layers.append(
        pdk.Layer(
            "TextLayer",
            data=rec_map,
            get_position=[lon_col, lat_col],
            get_text="rank_label",
            get_size=14,
            get_color=[255,255,255,255],
            get_text_anchor="'middle'",
            get_alignment_baseline="'center'",
            pickable=False,
        )
    )

deck = pdk.Deck(
    map_provider="carto",
    map_style="light",
    initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom),
    layers=layers,
    tooltip={"html": tooltip_html, "style": {"backgroundColor": "white", "color": "black"}}
)

st.pydeck_chart(deck, use_container_width=True)
