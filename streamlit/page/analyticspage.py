import streamlit as st
import pandas as pd
import plotly.express as px  # ileride kullanırsan dursun
import numpy as np
import altair as alt
import io, zipfile
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_datetime64_any_dtype

st.set_page_config(page_title="Veriseti", layout="wide")
st.title("Veriseti")

# ------------------ OTURUM DURUMU ------------------
if "stage" not in st.session_state:
    # aşamalar: "filter" -> "line" -> "bar"
    st.session_state.stage = "filter"
if "line_drawn" not in st.session_state:
    st.session_state.line_drawn = False
if "file_sig" not in st.session_state:
    st.session_state.file_sig = None

# ------------------ YÜKLEME: CSV veya ZIP ------------------
st.subheader("Veri Yükleme")
uploaded = st.file_uploader("CSV veya ZIP dosyası yükleyin", type=["csv", "zip"])

@st.cache_data(show_spinner=False)
def read_uploaded(name: str, data_bytes: bytes) -> pd.DataFrame:
    name_l = (name or "").lower()
    # CSV
    if name_l.endswith(".csv"):
        bio = io.BytesIO(data_bytes)
        try:
            return pd.read_csv(bio, low_memory=False)
        except UnicodeDecodeError:
            bio.seek(0)
            return pd.read_csv(bio, low_memory=False, encoding="latin1")
    # ZIP
    elif name_l.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data_bytes)) as z:
            csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("ZIP içinde .csv bulunamadı.")
            # ilk .csv dosyasını al
            target = csv_names[0]
            with z.open(target) as f:
                try:
                    return pd.read_csv(f, low_memory=False)
                except UnicodeDecodeError:
                    f.seek(0)
                    data = f.read()
                    return pd.read_csv(io.BytesIO(data), low_memory=False, encoding="latin1")
    else:
        raise ValueError("Desteklenmeyen dosya türü.")

if not uploaded:
    st.info("Lütfen bir CSV veya ZIP dosyası yükleyin.")
    st.stop()

# Dosya değiştiyse aşamaları sıfırla
file_sig = (uploaded.name, uploaded.size if hasattr(uploaded, "size") else len(uploaded.getvalue()))
if file_sig != st.session_state.file_sig:
    st.session_state.stage = "filter"
    st.session_state.line_drawn = False
    st.session_state.file_sig = file_sig

# Veri oku
try:
    df = read_uploaded(uploaded.name, uploaded.getvalue())
    st.success("Veri yüklendi.")
    st.dataframe(df, use_container_width=True)
except Exception as e:
    st.error(f"Veri okunamadı: {e}")
    st.stop()

# ------------------ METRİKLER ------------------
st.subheader("Metrikler")

def mean_or_nan(frame: pd.DataFrame, col: str):
    if col in frame.columns:
        return float(pd.to_numeric(frame[col], errors="coerce").mean())
    return np.nan

avg_price = mean_or_nan(df, "price")
avg_rsr   = mean_or_nan(df, "review_scores_rating")
avg_rsa   = mean_or_nan(df, "review_scores_accuracy")
avg_rsc   = mean_or_nan(df, "review_scores_cleanliness")
avg_rsl   = mean_or_nan(df, "review_scores_location")

# custom_score: varsa kullan; yoksa mevcut puan kolonlarının ortalamasıyla üret
score_candidates = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_location",
]
if "custom_score" in df.columns:
    df["custom_score"] = pd.to_numeric(df["custom_score"], errors="coerce")
else:
    cols_present = [c for c in score_candidates if c in df.columns]
    if cols_present:
        df["custom_score"] = df[cols_present].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    else:
        df["custom_score"] = np.nan

avg_custom = float(pd.to_numeric(df["custom_score"], errors="coerce").mean())

col1, col2, col3, col4, col5, col6 = st.columns(6)

def fmt(v, suffix=""):
    return ("-" if pd.isna(v) else f"{v:.2f} {suffix}").strip()

col1.metric("avg_price", fmt(avg_price, "PRC"))
col2.metric("avg_rsr",   fmt(avg_rsr, "RSR"))
col3.metric("avg_rsa",   fmt(avg_rsa, "RSA"))
col4.metric("avg_rsc",   fmt(avg_rsc, "RSC"))
col5.metric("avg_rsl",   fmt(avg_rsl, "RSL"))
col6.metric("avg_custom", fmt(avg_custom, "CS"))

# ------------------ VERİ FİLTRELEME ------------------
st.subheader("Veri Filtreleme")

columns = df.columns.tolist()
selected_column = st.selectbox("Kolon Seçiniz", columns, index=0)
filter_value = st.text_input("Bir değer giriniz")

# Filtre uygulanmış veri (yoksa orijinal)
filtered_df = df.copy()
if selected_column and filter_value:
    col_dtype = df[selected_column].dtype
    try:
        if pd.api.types.is_numeric_dtype(col_dtype):
            filter_value_casted = float(filter_value)
            filtered_df = df[pd.to_numeric(df[selected_column], errors="coerce") == filter_value_casted]
        else:
            filtered_df = df[df[selected_column].astype(str).str.lower() == str(filter_value).lower()]
        st.write(filtered_df)
    except ValueError:
        st.warning("Girilen değer bu kolonun veri tipine dönüştürülemedi.")

# Aşama ilerletme butonu: Filtrelemeyi bitirince Line Chart aç
proceed_line = st.button("Line Chart", type="primary", use_container_width=True)
if proceed_line:
    st.session_state.stage = "line"
    st.session_state.line_drawn = False  # line grafiği yeniden çizdirilsin

# ------------------ LINE CHART (yalnızca stage >= line) ------------------
if st.session_state.stage in ("line", "bar"):
    st.subheader("Line Chart")
    columns = filtered_df.columns.tolist()
    x2_column = st.selectbox("X ekseni için bir kolon seçiniz", columns, key="line_x")
    y2_column = st.selectbox("Y ekseni için bir kolon seçiniz", columns, key="line_y")

    agg_func = st.selectbox("Y için toplama yöntemi", ["mean", "sum", "median", "count"], index=0, key="line_agg")
    scale_mode = st.selectbox("Değer ölçeklendirme", ["yüzdelik%", "Ham", "kısaltma"], index=0, key="line_scale")

    btn = st.button("Grafiği Çiz", key="line_btn")

    if btn:
        plot_df = filtered_df[[x2_column, y2_column]].copy()

        keep_original = ["neighbourhood", "neighborhood", "neighboorhood",
                        "city", "country", "room_type", "property_type"]

        for c in [x2_column, y2_column]:
            s = plot_df[c]
            if is_object_dtype(s) and c not in keep_original:
                parsed_dt = pd.to_datetime(s, errors="coerce")
                parsed_num = pd.to_numeric(s, errors="coerce")
                if parsed_dt.notna().any():
                    plot_df[c] = parsed_dt
                elif parsed_num.notna().any():
                    plot_df[c] = parsed_num
                else:
                    plot_df[c] = s.astype(str)

        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x2_column, y2_column])

        if not is_numeric_dtype(plot_df[y2_column]):
            st.warning("Y ekseni sayısal olmalı. Lütfen sayısal bir sütun seçin.")
            st.stop()

        def do_agg(gb):
            if agg_func == "mean":
                return gb.mean()
            elif agg_func == "sum":
                return gb.sum()
            elif agg_func == "median":
                return gb.median()
            else:
                return gb.count()

        is_dt = is_datetime64_any_dtype(plot_df[x2_column])
        is_num_x = is_numeric_dtype(plot_df[x2_column])

        if is_dt:
            plot_df = plot_df.set_index(x2_column).sort_index()
            span_days = (plot_df.index.max() - plot_df.index.min()).days if len(plot_df) else 0
            rule = "M" if span_days > 365 else ("W" if span_days > 90 else "D")
            s = do_agg(plot_df[y2_column].resample(rule))
            chart_df = s.reset_index().rename(columns={x2_column: "x", y2_column: "value"})
            x_type = "T"
        elif is_num_x:
            tmp = plot_df.groupby(x2_column, as_index=False)[y2_column].agg(do_agg)
            tmp = tmp.sort_values(by=x2_column)
            chart_df = tmp.rename(columns={x2_column: "x", y2_column: "value"})
            x_type = "Q"
        else:
            tmp = plot_df.groupby(x2_column, as_index=False)[y2_column].agg(do_agg)
            chart_df = tmp.rename(columns={x2_column: "x", y2_column: "value"})
            x_type = "N"

        def scale_zero_to_max(s: pd.Series) -> pd.Series:
            vmax = s.max()
            vmin = s.min()
            if pd.isna(vmax):
                return s * 0
            if vmin < 0 and vmax > vmin:
                return (s - vmin) / (vmax - vmin)
            return (s / vmax) if vmax != 0 else s * 0

        y_field = "value"
        y_axis = alt.Axis(title=y2_column)
        tooltips = [alt.Tooltip("x:" + x_type, title=x2_column),
                    alt.Tooltip("value:Q", title=f"{y2_column} (ham)", format=".2f")]

        if scale_mode == "yüzdelik%":
            chart_df["value_scaled"] = scale_zero_to_max(chart_df["value"])
            y_field = "value_scaled"
            y_axis = alt.Axis(title=f"{y2_column} (0–100%)", format=".0%")
            tooltips.append(alt.Tooltip("value_scaled:Q", title="Oran", format=".0%"))
        elif scale_mode == "kısaltma":
            y_axis = alt.Axis(title=y2_column, format="s")
            tooltips[-1] = alt.Tooltip("value:Q", title=y2_column, format="s")

        chart = (
            alt.Chart(chart_df)
            .mark_line(point=True)
            .encode(
                x=alt.X(f"x:{x_type}", title=x2_column, sort=None),
                y=alt.Y(f"{y_field}:Q", axis=y_axis),
                tooltip=tooltips
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)
        st.session_state.line_drawn = True

    # Line sonrası ilerleme butonu (grafik çizildiyse)
    if st.session_state.line_drawn:
        proceed_bar = st.button("Bar Chart", type="primary", use_container_width=True)
        if proceed_bar:
            st.session_state.stage = "bar"

# ------------------ BAR CHART (yalnızca stage == bar) ------------------
if st.session_state.stage == "bar":
    st.subheader("Bar Chart")
    columns = filtered_df.columns.tolist()
    x2_column = st.selectbox("X ekseni için bir kolon seçiniz", columns, key="bar_x")
    y2_column = st.selectbox("Y ekseni için bir kolon seçiniz", columns, key="bar_y")

    agg_func = st.selectbox("Y için toplama yöntemi", ["mean", "sum", "median", "count"], index=0, key="bar_agg")
    scale_mode = st.selectbox("Değer ölçeklendirme", ["yüzdelik%", "Ham", "kısaltma"], index=0, key="bar_scale")

    btn = st.button("Grafiği Çiz", key="bar_btn")

    if btn:
        plot_df = filtered_df[[x2_column, y2_column]].copy()

        keep_original = ["neighbourhood", "neighborhood", "neighboorhood",
                        "city", "country", "room_type", "property_type"]

        for c in [x2_column, y2_column]:
            s = plot_df[c]
            if is_object_dtype(s) and c not in keep_original:
                parsed_dt = pd.to_datetime(s, errors="coerce")
                if parsed_dt.notna().any():
                    plot_df[c] = parsed_dt
                else:
                    parsed_num = pd.to_numeric(s, errors="coerce")
                    if parsed_num.notna().any():
                        plot_df[c] = parsed_num
                    else:
                        plot_df[c] = pd.Categorical(s.astype(str)).codes

        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x2_column, y2_column])

        if not is_numeric_dtype(plot_df[y2_column]):
            st.warning("Y ekseni sayısal olmalı. Lütfen sayısal bir sütun seçin.")
            st.stop()

        if is_numeric_dtype(plot_df[x2_column]):
            nbins = min(20, max(5, int(max(5, plot_df[x2_column].nunique() // 10))))
            plot_df["x_bin"] = pd.cut(plot_df[x2_column], bins=nbins)
            group_key = "x_bin"
        else:
            group_key = x2_column

        if agg_func == "mean":
            agg = plot_df.groupby(group_key, dropna=False)[y2_column].mean()
        elif agg_func == "sum":
            agg = plot_df.groupby(group_key, dropna=False)[y2_column].sum()
        elif agg_func == "median":
            agg = plot_df.groupby(group_key, dropna=False)[y2_column].median()
        else:
            agg = plot_df.groupby(group_key, dropna=False)[y2_column].count()

        chart_df = agg.reset_index().rename(columns={y2_column: "value", group_key: "x"})
        chart_df["x"] = chart_df["x"].astype(str)

        def scale_zero_to_max(s: pd.Series) -> pd.Series:
            vmax = s.max()
            vmin = s.min()
            if pd.isna(vmax):
                return s * 0
            if vmin < 0 and vmax > vmin:
                return (s - vmin) / (vmax - vmin)
            if vmax > 0:
                return s / vmax
            return s * 0

        y_field = "value"
        y_axis = alt.Axis(title=y2_column)

        if scale_mode == "yüzdelik%":
            chart_df["value_scaled"] = scale_zero_to_max(chart_df["value"])
            y_field = "value_scaled"
            y_axis = alt.Axis(title=f"{y2_column} (0–100%)", format=".0%")
        elif scale_mode == "kısaltma":
            y_axis = alt.Axis(title=y2_column, format="s")

        tooltip_fields = [
            alt.Tooltip("x:N", title=x2_column),
            alt.Tooltip("value:Q", title=f"{y2_column} (ham)", format=".2f")
        ]
        if scale_mode == "yüzdelik%":
            tooltip_fields.append(alt.Tooltip("value_scaled:Q", title="Oran", format=".0%"))

        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("x:N", title=x2_column, sort=None),
                y=alt.Y(f"{y_field}:Q", axis=y_axis),
                tooltip=tooltip_fields
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)
