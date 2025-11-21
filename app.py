import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime, time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import hashlib
import json
import copy

# ============================================================
# 0) GLOBAL CONFIG
# ============================================================
TS_CONFIG = {
    "max_points_per_param": 7200,
    "rolling_window": 10,
    "spike_threshold_factor": 2.0,
    "gap_minutes": 20,  # for plotting (visual breaks)
}

gap_threshold = pd.Timedelta(minutes=TS_CONFIG["gap_minutes"])

# ============================================================
# 1) APP CONFIGURATION
# ============================================================
st.set_page_config(layout="wide")

# ------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------
if "timeline_graphs" not in st.session_state:
    st.session_state.timeline_graphs = [{"id": 1, "params": []}]
    st.session_state.next_timeline_graph_id = 2

if "active_filter_config" not in st.session_state:
    st.session_state.active_filter_config = None

if "applied_value_filters" not in st.session_state:
    st.session_state.applied_value_filters = []

# ============================================================
# SIDEBAR — DATA MODE & FILE UPLOAD
# ============================================================
st.sidebar.header("Datenquelle")

data_mode = st.sidebar.radio(
    "Datenformat auswählen",
    ["Hydra", "SPS"],
    horizontal=False,
)

uploaded_files = st.sidebar.file_uploader(
    "Dateien hochladen",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
)

st.sidebar.markdown("---")

# ============================================================
# TITLE
# ============================================================
st.title(f"📊 Prozessparameteranalyse ({data_mode}-Daten)")

if not uploaded_files:
    st.info("Bitte laden Sie mindestens eine Datei hoch, um zu beginnen.")
    st.stop()

# ============================================================
# 3) DATA LOADING
# ============================================================
@st.cache_data
def load_data(uploaded_files, mode):
    frames = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name.lower()

        if mode == "Hydra":
            expected_cols = [
                "Characteristic designation",
                "Measured value",
                "Measured value from",
            ]

            df = pd.read_excel(
                uploaded_file,
                skiprows=1,
                usecols=lambda col: col in expected_cols,
            )

        else:  # mode == "SPS"
            sps_cols = ["VarName", "VarValue", "TimeString"]

            try:
                df = pd.read_csv(
                    uploaded_file,
                    sep=";",
                    decimal=",",
                    encoding="latin1",
                    usecols=[0, 1, 2],
                    header=0,
                )
            except UnicodeDecodeError:
                df = pd.read_csv(
                    uploaded_file,
                    sep=";",
                    decimal=",",
                    encoding="cp1252",
                    usecols=sps_cols,
                )

            df = df.rename(columns={
                "VarName": "Characteristic designation",
                "VarValue": "Measured value",
                "TimeString": "Measured value from",
            })

        if "Measured value" in df.columns:
            df["Measured value"] = pd.to_numeric(
                df["Measured value"], errors="coerce", downcast="float"
            )

        if "Measured value from" in df.columns:
            df["Measured value from"] = pd.to_datetime(
                df["Measured value from"],
                dayfirst=True,
                errors="coerce",
                cache=True,
            )

        required = ["Characteristic designation", "Measured value", "Measured value from"]
        df = df.dropna(subset=required)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(
        subset=["Characteristic designation", "Measured value from"],
        keep="last",
    )
    data["Characteristic designation"] = data["Characteristic designation"].astype("category")
    data = data.sort_values("Measured value from").reset_index(drop=True)

    return data


# ============================================================
# TIME MASK (weekly window)
# ============================================================
def build_time_mask_for_series(
    timestamps: pd.Series,
    enable_weekly_window: bool,
    weekly_conf: dict | None,
):
    if timestamps.empty:
        return pd.Series([], dtype=bool, index=timestamps.index)

    mask = pd.Series(True, index=timestamps.index)

    if enable_weekly_window and weekly_conf is not None:
        start_idx = weekly_conf["start_idx"]
        end_idx = weekly_conf["end_idx"]
        start_time_of_day = weekly_conf["start_time"]
        end_time_of_day = weekly_conf["end_time"]

        def to_fraction(day, t):
            return day + (t.hour + t.minute / 60 + t.second / 3600) / 24.0

        start_f = to_fraction(start_idx, start_time_of_day)
        end_f = to_fraction(end_idx, end_time_of_day)

        ts_f = (
            timestamps.dt.weekday +
            timestamps.dt.hour / 24 +
            timestamps.dt.minute / (24 * 60) +
            timestamps.dt.second / (24 * 3600)
        )

        if start_f <= end_f:
            week_mask = (ts_f >= start_f) & (ts_f <= end_f)
        else:
            week_mask = (ts_f >= start_f) | (ts_f <= end_f)

        mask &= week_mask

    return mask


# ============================================================
# VALUE-BASED DOWNSAMPLING PER PARAMETER (NO NEW TIMESTAMPS)
# ============================================================
def select_kept_timestamps_for_param(
    df_param_raw: pd.DataFrame,
    max_points: int,
    rolling_window: int = 10,
    spike_threshold_factor: float = 2.0,
    eps_dynamic: float = 0.01,
    eps_static: float = 0.05,
    max_flat_duration_minutes: int = 20,
):
    if df_param_raw.empty:
        return np.array([], dtype="datetime64[ns]")

    df = df_param_raw.sort_values("Measured value from").copy()

    v = pd.to_numeric(df["Measured value"], errors="coerce").to_numpy()
    t = df["Measured value from"].to_numpy()

    mask_valid = ~np.isnan(v)
    v = v[mask_valid]
    t = t[mask_valid]

    n = len(v)
    if n <= max_points:
        return t

    # -----------------------------
    # 1) Spike detection (existing)
    # -----------------------------
    diff = np.abs(np.diff(v, prepend=v[0]))
    rolling_std = pd.Series(v).rolling(window=rolling_window, min_periods=5).std().to_numpy()
    threshold = spike_threshold_factor * np.nan_to_num(rolling_std, nan=0.0)
    spike_mask = diff > threshold

    must_keep = spike_mask.copy()
    must_keep[0] = True
    must_keep[-1] = True

    # -----------------------------------------------
    # 2) Adaptive linear-deviation redundancy removal
    # -----------------------------------------------
    keep_mask = must_keep.copy()

    for i in range(1, n - 1):
        if must_keep[i]:
            continue  # spikes always kept

        t0, t1, t2 = t[i-1], t[i], t[i+1]
        v0, v1, v2 = v[i-1], v[i], v[i+1]

        dt_01 = (t1 - t0).astype('timedelta64[s]').astype(int)
        dt_02 = (t2 - t0).astype('timedelta64[s]').astype(int)

        # Prevent division by zero
        if dt_02 == 0:
            continue

        slope = (v2 - v0) / dt_02
        v_pred = v0 + slope * dt_01

        deviation = abs(v1 - v_pred)

        tolerance = abs(slope) * dt_01 * eps_dynamic + eps_static

        if deviation <= tolerance:
            keep_mask[i] = False  # redundant → drop

    # -------------------------------------------------------------
    # 3) Enforce anchor points for long flat segments (> max_flat)
    # -------------------------------------------------------------
    min_anchor_step = max_flat_duration_minutes * 60  # seconds
    last_kept_idx = np.where(keep_mask)[0][0]

    for i in range(1, n):
        if keep_mask[i]:
            last_kept_idx = i
            continue

        dt_since_last = (t[i] - t[last_kept_idx]).astype('timedelta64[s]').astype(int)

        if dt_since_last >= min_anchor_step:
            keep_mask[i] = True
            last_kept_idx = i

    # -------------------------------------------
    # 4) Build index list BEFORE sampling
    # -------------------------------------------
    idx_kept = np.where(keep_mask)[0]
    n_kept = len(idx_kept)

    if n_kept <= max_points:
        return t[idx_kept]

    # -------------------------------------------
    # 5) Sampling fallback (unchanged)
    # -------------------------------------------
    step = int(np.ceil(n_kept / max_points))
    return t[idx_kept[::step]]


# ============================================================
# BUILD ALIGNED DOWNSAMPLED PIVOT
# ============================================================
@st.cache_data
def build_aligned_pivot(
    data: pd.DataFrame,
    applied_parameters: tuple,
    ts_config: dict,
):
    """
    1) For each parameter: select subset of original timestamps (value-based downsampling)
    2) Build unified timestamp index (union of all kept timestamps)
    3) For each param: read values from RAW at those timestamps (no fill)
    4) Return:
       - aligned_pivot: DataFrame index=timestamp, columns=params
       - downsample_stats: dict with raw_count, down_count, reduction
    """
    aligned_pivot = pd.DataFrame()

    if data.empty or not applied_parameters:
        return aligned_pivot

    kept_ts_per_param = {}

    # --- 1) select kept timestamps per param ---
    for param in applied_parameters:
        df_param_raw = data[data["Characteristic designation"] == param][
            ["Measured value from", "Measured value"]
        ]


        kept_ts = select_kept_timestamps_for_param(
            df_param_raw,
            max_points=ts_config["max_points_per_param"],
            rolling_window=ts_config["rolling_window"],
            spike_threshold_factor=ts_config["spike_threshold_factor"],
        )

        kept_ts_per_param[param] = kept_ts



    # --- 2) build unified timestamps ---
    all_ts = [ts_arr for ts_arr in kept_ts_per_param.values() if ts_arr.size > 0]
    if not all_ts:
        return aligned_pivot

    unified_timestamps = np.unique(np.concatenate(all_ts))
    unified_timestamps = np.sort(unified_timestamps)

    # --- 3) for each param: reindex raw series on unified timestamps (no fill) ---
    cols = {}
    for param in applied_parameters:
        df_param_raw = data[data["Characteristic designation"] == param][
            ["Measured value from", "Measured value"]
        ].copy()

        if df_param_raw.empty:
            continue

        df_param_raw = df_param_raw.set_index("Measured value from")
        series_aligned = df_param_raw["Measured value"].reindex(unified_timestamps)

        cols[param] = series_aligned

    if not cols:
        return aligned_pivot

    aligned_pivot = pd.DataFrame(cols)
    # ensure datetime index
    aligned_pivot.index = pd.to_datetime(aligned_pivot.index)

    return aligned_pivot


# ============================================================
# APPLY FILTERS TO ALIGNED PIVOT
# ============================================================
def apply_filters_to_pivot(
    aligned_pivot: pd.DataFrame,
    enable_weekly_window: bool,
    weekly_conf: dict | None,
    value_filters: list,
):
    """
    Filters:
    - weekly window on index (timestamps)
    - value filters per parameter (global timestamp removal if param out-of-range)
    """
    if aligned_pivot.empty:
        return aligned_pivot

    idx_series = aligned_pivot.index.to_series()

    # Weekly window mask
    time_mask = build_time_mask_for_series(
        idx_series,
        enable_weekly_window=enable_weekly_window,
        weekly_conf=weekly_conf,
    )

    mask = time_mask.copy()

    # Value filters (global mask)
    if value_filters:
        for f in value_filters:
            param = f.get("param")
            if not param or param not in aligned_pivot.columns:
                continue

            try:
                min_val = float(f.get("min", "")) if f.get("min", "") != "" else None
            except Exception:
                min_val = None

            try:
                max_val = float(f.get("max", "")) if f.get("max", "") != "" else None
            except Exception:
                max_val = None

            series = aligned_pivot[param]

            # Start with all True for this param
            cond = pd.Series(True, index=aligned_pivot.index)

            if min_val is not None:
                cond &= (series >= min_val) | series.isna()
            if max_val is not None:
                cond &= (series <= max_val) | series.isna()

            mask &= cond

    return aligned_pivot[mask]


# ============================================================
# CORR / PCA HELPERS
# ============================================================
def make_filter_key(config: dict) -> str:
    def clean(obj):
        if isinstance(obj, (pd.Timestamp, datetime, time)):
            return obj.isoformat()

        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)

        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)

        if isinstance(obj, (list, tuple)):
            return [clean(x) for x in obj]

        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}

        return obj

    safe_conf = clean(config)
    s = json.dumps(safe_conf, sort_keys=True)

    # Correct MD5 digest
    return hashlib.md5(s.encode("utf-8")).hexdigest()



@st.cache_data
def compute_corr_matrix(pivot_df: pd.DataFrame, filter_key: str) -> pd.DataFrame:
    if pivot_df.empty:
        return pd.DataFrame()
    return pivot_df.corr()


@st.cache_data
def compute_pca_2d(
    pivot_df: pd.DataFrame,
    selected_parameters: tuple,
    filter_key: str,
):
    if pivot_df.empty or len(selected_parameters) < 2:
        return np.empty((0, 2)), []

    valid_params = [p for p in selected_parameters if p in pivot_df.columns]
    if len(valid_params) < 2:
        return np.empty((0, 2)), []

    df_pca = pivot_df[valid_params].dropna()
    if df_pca.empty:
        return np.empty((0, 2)), []

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_pca)
    pca = PCA(n_components=2)
    comp = pca.fit_transform(scaled)
    return comp, valid_params


# ============================================================
# 5) LOAD DATA
# ============================================================
data = load_data(uploaded_files, data_mode)

if data.empty:
    st.error("Keine gültigen Daten nach dem Laden.")
    st.stop()

st.sidebar.header("Filteroptionen")

# ============================================================
# APPLY BUTTON
# ============================================================
st.sidebar.markdown("## Anwenden / Aktualisieren")
apply_clicked = st.sidebar.button("✅ Filter anwenden / Daten aktualisieren", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("## Filteroptionen")

# ============================================================
# PARAMETER SELECTION
# ============================================================
all_parameters = sorted(data["Characteristic designation"].unique())

if "user_selected_parameters" not in st.session_state:
    st.session_state.user_selected_parameters = []

st.sidebar.markdown("### Parameter")

if st.sidebar.button("Alle auswählen"):
    st.session_state.user_selected_parameters = all_parameters.copy()

selected = st.sidebar.multiselect(
    "Parameter auswählen",
    options=all_parameters,
    key="user_selected_parameters",
)

user_selected_parameters = selected

st.sidebar.markdown("---")

# ============================================================
# WEEKLY WINDOW
# ============================================================
with st.sidebar.expander("📅  Wöchentliches Zeitfenster", expanded=False):

    if st.session_state.active_filter_config is not None:
        default_weekly_enabled = st.session_state.active_filter_config.get("enable_weekly_window", False)
    else:
        default_weekly_enabled = False

    enable_weekly_window_widget = st.checkbox(
        "Wöchentliches Zeitfenster aktivieren",
        value=default_weekly_enabled,
        key="weekly_window_enable",
    )

    weekday_options = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]

    if enable_weekly_window_widget:
        st.subheader("Wöchentlicher Zeitraum")

        start_weekday_widget = st.selectbox("Start-Tag", weekday_options, index=0)
        start_time_of_day_widget = st.time_input(
            "Startzeit",
            datetime.strptime("09:00", "%H:%M").time(),
        )

        end_weekday_widget = st.selectbox("End-Tag", weekday_options, index=5)
        end_time_of_day_widget = st.time_input(
            "Endzeit",
            datetime.strptime("05:00", "%H:%M").time(),
        )
    else:
        start_weekday_widget = weekday_options[0]
        end_weekday_widget = weekday_options[5]
        start_time_of_day_widget = datetime.strptime("09:00", "%H:%M").time()
        end_time_of_day_widget = datetime.strptime("05:00", "%H:%M").time()

# ============================================================
# VALUE FILTERS
# ============================================================
st.sidebar.markdown("---")
with st.sidebar.expander("🔍 Parameterwert-Filter", expanded=False):

    if "value_filters" not in st.session_state:
        st.session_state.value_filters = []

    if st.button("➕ Filter hinzufügen", key="add_value_filter"):
        st.session_state.value_filters.append({
            "id": len(st.session_state.value_filters) + 1,
            "param": None,
            "min": "",
            "max": "",
        })

    filters_to_delete = []

    for f in st.session_state.value_filters:

        st.markdown(f"### Filter {f['id']}")

        f["param"] = st.selectbox(
            "Parameter",
            options=all_parameters,
            index=all_parameters.index(f["param"]) if f["param"] in all_parameters else 0,
            key=f"valuefilter_param_{f['id']}",
        )

        f["min"] = st.text_input(
            "Zwischen (Min)",
            value=f["min"],
            key=f"valuefilter_min_{f['id']}",
        )
        f["max"] = st.text_input(
            "und (Max)",
            value=f["max"],
            key=f"valuefilter_max_{f['id']}",
        )

        if st.button(f"❌ Filter entfernen", key=f"valuefilter_del_{f['id']}"):
            filters_to_delete.append(f)

    for f in filters_to_delete:
        st.session_state.value_filters.remove(f)

# ============================================================
# ADVANCED MODE
# ============================================================
st.sidebar.markdown("---")
advanced_mode = st.sidebar.checkbox("Erweiterten Analysemodus aktivieren", value=False)

# ============================================================
# APPLY BUTTON HANDLING
# ============================================================
weekday_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
}

weekly_conf_obj = {
    "start_idx": weekday_map[start_weekday_widget],
    "end_idx": weekday_map[end_weekday_widget],
    "start_time": start_time_of_day_widget,
    "end_time": end_time_of_day_widget,
}

current_filter_config = {
    "selected_parameters": tuple(user_selected_parameters),
    "enable_weekly_window": bool(enable_weekly_window_widget),
    "weekly_conf": weekly_conf_obj,
}

if apply_clicked or st.session_state.active_filter_config is None:
    st.session_state.active_filter_config = current_filter_config
    st.session_state.applied_value_filters = copy.deepcopy(st.session_state.value_filters)

active_config = st.session_state.active_filter_config
applied_parameters = list(active_config["selected_parameters"])
enable_weekly_window = active_config["enable_weekly_window"]
weekly_conf_active = active_config["weekly_conf"]

filter_key = make_filter_key(active_config)

# ============================================================
# 9) BUILD ALIGNED PIVOT + APPLY FILTERS
# ============================================================
aligned_pivot = build_aligned_pivot(
    data,
    tuple(applied_parameters),
    TS_CONFIG,
)

filtered_pivot = apply_filters_to_pivot(
    aligned_pivot,
    enable_weekly_window=enable_weekly_window,
    weekly_conf=weekly_conf_active,
    value_filters=st.session_state.get("applied_value_filters", []),
)

# ============================================================
# COMPUTE PLOT-POINT STATS (points actually used for plotting)
# ============================================================
plot_stats = {}

for param in applied_parameters:
    if param not in filtered_pivot.columns:
        continue

    series = filtered_pivot[param]

    # Convert to DataFrame (same as plotting code)
    plot_df = series.to_frame(name="Measured value").reset_index()
    plot_df = plot_df.rename(columns={"index": "Measured value from"})
    plot_df.index.name = None

    # Apply gap detection (same logic as in plots)
    plot_df = plot_df.sort_values("Measured value from")
    plot_df["time_diff"] = plot_df["Measured value from"].diff()
    plot_df.loc[plot_df["time_diff"] > gap_threshold, "Measured value"] = np.nan
    plot_df = plot_df.drop(columns=["time_diff"])

    # Count visible points (non-NaN)
    visible_points = plot_df["Measured value"].count()

    # Full timeline length (including NaNs)
    total_index_points = len(plot_df)

    plot_stats[param] = {
        "visible": visible_points,
        "index_len": total_index_points,
    }

pivot_df_clean = filtered_pivot  # used by analysis tabs

if not filtered_pivot.empty:
    start_time_final = filtered_pivot.index.min()
    end_time_final = filtered_pivot.index.max()
else:
    if not aligned_pivot.empty:
        start_time_final = aligned_pivot.index.min()
        end_time_final = aligned_pivot.index.max()
    else:
        start_time_final = data["Measured value from"].min()
        end_time_final = data["Measured value from"].max()


# ============================================================
# 11) TABS
# ============================================================
if advanced_mode:
    tab_names = [
        "Zeitreihendiagramm", "Mehrparameter-Diagramm (Timeline)",
        "Scatter-Plot", "Histogramme / Verteilungen",
        "Korrelationsmatrix", "Pair Plot",
        "Zeitverschobene Korrelation", "PCA",
        "Ausreißererkennung",
    ]
else:
    tab_names = [
        "Zeitreihendiagramm", "Mehrparameter-Diagramm (Timeline)",
        "Scatter-Plot", "Histogramme / Verteilungen",
    ]

tabs = st.tabs(tab_names)



# ============================================================
# TAB 0 — Zeitreihendiagramm (from filtered_pivot)
# ============================================================
with tabs[0]:
    st.subheader("Zeitreihendiagramm")

    if applied_parameters and not filtered_pivot.empty:
        for param in applied_parameters:
            if param not in filtered_pivot.columns:
                continue

            series = filtered_pivot[param]
            if series.dropna().empty:
                continue

            # Convert series → full DataFrame, index becomes a normal column
            df_param = series.to_frame(name="Measured value").reset_index()
            df_param = df_param.rename(columns={"index": "Measured value from"})
            df_param.index.name = None  # ensure index has no name

            df_param = df_param.sort_values("Measured value from")
            df_param["time_diff"] = df_param["Measured value from"].diff()
            df_param.loc[df_param["time_diff"] > gap_threshold, "Measured value"] = np.nan
            df_param = df_param.drop(columns=["time_diff"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_param["Measured value from"],
                y=df_param["Measured value"],
                mode="lines+markers",
                name=param,
                connectgaps=False,
                hovertemplate=f"<b>{param}</b><br>%{{x}}<br>Messwert: %{{y}}<extra></extra>",
            ))

            fig.update_layout(
                title=param,
                xaxis_title="Zeit",
                yaxis_title="Wert",
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Bitte mindestens einen Parameter auswählen und 'Filter anwenden' klicken.")

# ============================================================
# TAB 1 — Mehrparameter-Diagramm (Timeline)
# ============================================================
with tabs[1]:
    st.subheader("Mehrparameter-Diagramm (Timeline)")

    options = list(pivot_df_clean.columns)

    if not filtered_pivot.empty:
        global_min = filtered_pivot.index.min()
        global_max = filtered_pivot.index.max()
    else:
        global_min = start_time_final
        global_max = end_time_final

    for graph in st.session_state.timeline_graphs:
        valid_default = [p for p in graph["params"] if p in options]
        graph["params"] = st.multiselect(
            f"Timeline Graph {graph['id']}: Parameter auswählen",
            options=options,
            default=valid_default,
            key=f"timeline_graph_{graph['id']}",
        )

    if st.button("Neues Timeline-Diagramm hinzufügen"):
        st.session_state.timeline_graphs.append(
            {
                "id": st.session_state.next_timeline_graph_id,
                "params": [p for p in applied_parameters if p in options],
            }
        )
        st.session_state.next_timeline_graph_id += 1

    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]

    for graph in st.session_state.timeline_graphs:
        if not graph["params"]:
            continue

        if st.button(f"Diagramm entfernen {graph['id']}", key=f"remove_graph_{graph['id']}"):
            st.session_state.timeline_graphs = [
                g for g in st.session_state.timeline_graphs if g["id"] != graph["id"]
            ]
            st.experimental_rerun()

        fig = go.Figure()

        for idx, param in enumerate(graph["params"]):
            if param not in filtered_pivot.columns:
                continue

            series = filtered_pivot[param]
            if series.dropna().empty:
                continue

            dfp = series.to_frame(name="Measured value").reset_index()
            dfp = dfp.rename(columns={"index": "Measured value from"})
            dfp.index.name = None   # avoid duplicate index name

            dfp = dfp.sort_values("Measured value from")
            dfp["time_diff"] = dfp["Measured value from"].diff()
            dfp.loc[dfp["time_diff"] > gap_threshold, "Measured value"] = np.nan
            dfp = dfp.drop(columns=["time_diff"])

            fig.add_trace(go.Scatter(
                x=dfp["Measured value from"],
                y=dfp["Measured value"],
                mode="lines+markers",
                name=param,
                line=dict(color=colors[idx % len(colors)]),
                connectgaps=False,
            ))

        fig.update_layout(
            title=f"Timeline-Diagramm {graph['id']}",
            xaxis_title="Zeit",
            hovermode="x unified",
            xaxis=dict(range=[global_min, global_max]),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
            ),
            margin=dict(b=80),
        )

        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2 — Scatter
# ============================================================
with tabs[2]:
    st.subheader("Scatter-Plot")
    if len(applied_parameters) >= 2 and not pivot_df_clean.empty:
        x_param = st.selectbox("X-Parameter", applied_parameters, key="scatter_x")
        y_param = st.selectbox("Y-Parameter", applied_parameters, key="scatter_y")

        if x_param != y_param and x_param in pivot_df_clean.columns and y_param in pivot_df_clean.columns:
            df_scatter = pivot_df_clean[[x_param, y_param]].dropna()
            if not df_scatter.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(df_scatter[x_param], df_scatter[y_param])
                ax.set_xlabel(x_param)
                ax.set_ylabel(y_param)
                st.pyplot(fig)
            else:
                st.info("Keine gemeinsamen Messpunkte für diese Parameter.")
        else:
            st.warning("Bitte zwei unterschiedliche Parameter auswählen.")
    else:
        st.info("Bitte mindestens zwei Parameter auswählen und auf 'Filter anwenden' klicken.")

# ============================================================
# TAB 3 — Distribution Plot
# ============================================================
with tabs[3]:
    st.subheader("Histogramme / Verteilungen")

    if not pivot_df_clean.empty and pivot_df_clean.columns.size > 0:

        dist_params = st.multiselect(
            "Parameter auswählen für Histogramme / Verteilungen",
            options=list(pivot_df_clean.columns),
            default=[],
        )

        if not dist_params:
            st.info("Wählen Sie mindestens einen Parameter aus.")
        else:
            for param in dist_params:
                if param not in pivot_df_clean.columns:
                    continue
                series = pivot_df_clean[param].dropna()

                with st.expander(f"Verteilung von {param}", expanded=False):
                    if not series.empty:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.histplot(series, kde=True, ax=ax)
                        ax.set_title(f"{param} – Verteilung")
                        st.pyplot(fig)
                    else:
                        st.info("Keine Daten vorhanden.")
    else:
        st.info("Nicht genügend Daten.")

# ============================================================
# ADVANCED MODE
# ============================================================
if advanced_mode:

    # ---------------- CORRELATION ----------------
    with tabs[4]:
        st.subheader("Korrelationsmatrix")
        if not pivot_df_clean.empty:
            if st.button("Korrelationsmatrix berechnen"):
                corr = compute_corr_matrix(pivot_df_clean, filter_key)
                if not corr.empty:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("Nicht genügend Daten für die Korrelation.")
            else:
                st.info("Klicken Sie auf die Schaltfläche, um zu berechnen.")
        else:
            st.info("Nicht genügend Daten.")

    # ---------------- Pair Plot ----------------
    with tabs[5]:
        st.subheader("Pair Plot")
        if not pivot_df_clean.empty:
            if st.button("Pairplot berechnen"):
                try:
                    pair_cols = list(pivot_df_clean.columns)
                    if len(pair_cols) > 6:
                        pair_cols = pair_cols[:6]
                        st.info("Auf 6 Parameter beschränkt.")

                    df_for_pair = pivot_df_clean[pair_cols].dropna()
                    if len(df_for_pair) > 5000:
                        df_for_pair = df_for_pair.sample(5000, random_state=42)
                        st.info("Sample von 5000 datapoints.")

                    g = sns.pairplot(df_for_pair)
                    st.pyplot(g.fig)
                except Exception as e:
                    st.error(f"Pairplot-Fehler: {e}")
        else:
            st.info("Nicht genügend Daten.")

    # ---------------- TIME SHIFT ----------------
    with tabs[6]:
        st.subheader("Zeitverschobene Korrelation")
        if len(applied_parameters) >= 2 and not pivot_df_clean.empty:
            p1 = st.selectbox("Parameter-X", applied_parameters, key="tsc_p1")
            p2 = st.selectbox("Parameter-Y", applied_parameters, key="tsc_p2")
            lag = st.slider("Zeitverschiebung (Lag)", -20, 20, 0)

            if p1 in pivot_df_clean.columns and p2 in pivot_df_clean.columns:
                df_lag = pivot_df_clean[[p1, p2]].dropna()
                if not df_lag.empty:
                    shifted = df_lag[p2].shift(lag)
                    corr_val = df_lag[p1].corr(shifted)
                    st.write(f"Korrelation (Lag {lag}): **{corr_val:.3f}**")
                else:
                    st.info("Keine gemeinsamen Messpunkte.")
        else:
            st.info("Bitte zwei Parameter auswählen.")

    # ---------------- PCA ----------------
    with tabs[7]:
        st.subheader("PCA (2D)")
        if len(applied_parameters) > 1 and not pivot_df_clean.empty:
            if st.button("PCA berechnen"):
                comp, used_params = compute_pca_2d(
                    pivot_df_clean,
                    tuple(applied_parameters),
                    filter_key,
                )
                if comp.shape[0] > 0:
                    fig, ax = plt.subplots()
                    ax.scatter(comp[:, 0], comp[:, 1])
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_title(f"PCA: {', '.join(used_params)}")
                    st.pyplot(fig)
                else:
                    st.info("Nicht genügend Daten.")
        else:
            st.info("Wählen Sie 2+ Parameter.")

    # ---------------- OUTLIERS ----------------
    with tabs[8]:
        st.subheader("Ausreißererkennung")

        if not pivot_df_clean.empty and applied_parameters:

            method = st.selectbox("Methode", ["Z-Score", "IQR"])
            threshold = (
                st.slider("Z-Score Schwelle", 1.0, 5.0, 3.0)
                if method == "Z-Score"
                else st.slider("IQR-Multiplikator", 1.0, 3.0, 1.5)
            )

            for param in applied_parameters:
                if param not in pivot_df_clean.columns:
                    continue

                series = pivot_df_clean[param].dropna()
                if series.empty:
                    continue

                if method == "Z-Score":
                    z = zscore(series)
                    outliers = series[abs(z) > threshold]
                else:
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = series[
                        (series < Q1 - threshold * IQR) |
                        (series > Q3 + threshold * IQR)
                    ]

                with st.expander(f"Ausreißer für {param}", expanded=False):
                    fig, ax = plt.subplots()
                    sns.histplot(series, ax=ax)
                    if not outliers.empty:
                        sns.histplot(outliers, color="red", ax=ax)
                    ax.set_title(f"{param} – Ausreißer")
                    st.pyplot(fig)

        else:
            st.info("Nicht genügend Daten.")
