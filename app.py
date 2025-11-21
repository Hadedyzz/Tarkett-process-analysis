import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import hashlib
import json

# ============================================================
# 0) GLOBAL CONFIG
# ============================================================
TS_CONFIG = {
    "gap_minutes": 10,
    "max_points_per_param": 7200,
    "spike_window_points": 5,
    "rolling_window": 10,
    "spike_threshold_factor": 2.0,
}

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
# TITLE WITH ANALYTICS ICON
# ============================================================
st.title(f"📊 Prozessparameteranalyse ({data_mode}-Daten)")

# Only REJECT when user presses Apply — NOT immediately after upload
if uploaded_files is None or len(uploaded_files) == 0:
    st.info("Bitte laden Sie mindestens eine Datei hoch, um zu beginnen.")
    st.stop()
    
# ============================================================
# 3) DATA LOADING & CORE HELPERS
# ============================================================
@st.cache_data
def load_data(uploaded_files, mode):
    """
    Load data for either Hydra (Excel, skip first row)
    or SPS (CSV), normalize column names, parse timestamps,
    convert numeric values, and drop duplicates.
    """
    frames = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name.lower()

        # ---------------------------------------------------------
        # HYDRA → Excel file with first row as metadata
        # ---------------------------------------------------------
        if mode == "Hydra":
            # Expect Excel exports from Hydra
            expected_cols = [
                "Characteristic designation",
                "Measured value",
                "Measured value from",
                "Inspection station name",
            ]

            df = pd.read_excel(
                uploaded_file,
                skiprows=1,
                usecols=lambda col: col in expected_cols,  # only read required cols
            )

        # ---------------------------------------------------------
        # SPS → CSV
        # ---------------------------------------------------------
        else:  # mode == "SPS"
            sps_cols = ["VarName", "VarValue", "TimeString"]
        
            # Load CSV with robust fallback encodings
            try:
                df = pd.read_csv(
                    uploaded_file,
                    sep=";",
                    decimal=",",
                    encoding="utf-8-sig",
                    usecols=sps_cols,
                )
            except UnicodeDecodeError:
                df = pd.read_csv(
                    uploaded_file,
                    sep=";",
                    decimal=",",
                    encoding="cp1252",
                    usecols=sps_cols,
                )
        
            # SPS renaming
            df = df.rename(columns={
                "VarName": "Characteristic designation",
                "VarValue": "Measured value",
                "TimeString": "Measured value from",
            })

        # ---------------------------------------------------------
        # SHARED CLEANING FOR BOTH MODES
        # ---------------------------------------------------------

        # Convert numeric values
        if "Measured value" in df.columns:
            df["Measured value"] = pd.to_numeric(
                df["Measured value"],
                errors="coerce",
                downcast="float",
            )

        # Parse datetime values (dd.mm.yyyy supported)
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

    # Remove duplicates (same param & timestamp)
    data = data.drop_duplicates(
        subset=["Characteristic designation", "Measured value from"],
        keep="last",
    )

    # Optimize types
    data["Characteristic designation"] = data["Characteristic designation"].astype("category")

    # Sort chronologically
    data = data.sort_values("Measured value from").reset_index(drop=True)

    return data


@st.cache_data
def get_pivot_table(filtered_data: pd.DataFrame) -> pd.DataFrame:
    """Pivot for scatter / correlation / PCA etc."""
    if filtered_data.empty:
        return pd.DataFrame()
    df_unique = filtered_data.drop_duplicates(
        subset=["Measured value from", "Characteristic designation"],
        keep="last",
    )
    return df_unique.pivot(
        index="Measured value from",
        columns="Characteristic designation",
        values="Measured value",
    )


def build_single_param_timeseries(
    df_param_raw: pd.DataFrame,
    gap_minutes: int = 10,
    max_points_per_param: int = 7200,
    spike_window_points: int = 5,
    rolling_window: int = 10,
    spike_threshold_factor: float = 2.0,
) -> pd.DataFrame:
    """
    Build a single parameter time series for plotting with:
      - dynamic time-based downsampling (1/5/10/30 min)
      - dynamic spike detection (based on rolling std)
      - local high-resolution around spikes (± spike_window_points)
      - hard cap of max_points_per_param per parameter
      - gaps inserted for time jumps > gap_minutes
    """

    if df_param_raw.empty:
        return pd.DataFrame(columns=["Measured value from", "Measured value"])

    df = df_param_raw.copy()
    df["Measured value"] = pd.to_numeric(df["Measured value"], errors="coerce")
    df = df.dropna(subset=["Measured value"])
    if df.empty:
        return pd.DataFrame(columns=["Measured value from", "Measured value"])

    # Data is assumed globally sorted by time in load_data

    # ------------------------------------------------
    # 1) Choose base resample interval (global span)
    # ------------------------------------------------
    t_start = df["Measured value from"].iloc[0]
    t_end = df["Measured value from"].iloc[-1]
    total_days = (t_end - t_start).total_seconds() / 86400.0

    if total_days < 7:
        interval = "1min"
    elif total_days < 28:
        interval = "5min"
    elif total_days < 84:
        interval = "10min"
    else:
        interval = "30min"

    # ------------------------------------------------
    # 2) Base resample (uniform time grid) on numeric column only
    # ------------------------------------------------
    df_idx = df.set_index("Measured value from")[["Measured value"]]
    base = df_idx.resample(interval).mean(numeric_only=True)
    base = base.reset_index()

    # ------------------------------------------------
    # 3) Dynamic spike detection on original series
    # ------------------------------------------------
    s_val = df["Measured value"]
    diff = s_val.diff().abs()

    rolling_std = s_val.rolling(window=rolling_window, min_periods=5).std()
    threshold_series = spike_threshold_factor * rolling_std

    spike_mask = diff > threshold_series
    spike_indices = np.where(spike_mask.fillna(False).to_numpy())[0]

    # Collect high-resolution segments around spikes
    spike_segments = []
    n_raw = len(df)
    for idx in spike_indices:
        start_i = max(0, idx - spike_window_points)
        end_i = min(n_raw - 1, idx + spike_window_points)
        seg = df.iloc[start_i:end_i + 1]
        spike_segments.append(seg)

    if spike_segments:
        high_res = pd.concat(spike_segments).drop_duplicates(
            subset=["Measured value from"],
            keep="last",
        )
    else:
        high_res = pd.DataFrame(columns=df.columns)

    # ------------------------------------------------
    # 4) Merge base resample + spike windows
    # ------------------------------------------------
    merged = base.copy()
    if not high_res.empty:
        merged = pd.concat(
            [
                merged,
                high_res[["Measured value from", "Measured value"]],
            ],
            ignore_index=True,
        )

    merged = merged.drop_duplicates(
        subset=["Measured value from"],
        keep="last",
    ).sort_values("Measured value from")

    # ------------------------------------------------
    # 5) Insert gaps for large time jumps
    # ------------------------------------------------
    gap_threshold = pd.Timedelta(minutes=gap_minutes)
    merged["time_diff"] = merged["Measured value from"].diff()
    merged.loc[merged["time_diff"] > gap_threshold, "Measured value"] = np.nan
    merged = merged.drop(columns=["time_diff"])

    # ------------------------------------------------
    # 6) Enforce max_points_per_param
    # ------------------------------------------------
    if len(merged) > max_points_per_param:
        if not high_res.empty:
            spike_times = set(high_res["Measured value from"].unique())
            is_prio = merged["Measured value from"].isin(spike_times)
        else:
            is_prio = pd.Series(False, index=merged.index)

        prio_df = merged[is_prio]
        non_prio_df = merged[~is_prio]

        remaining_slots = max_points_per_param - len(prio_df)

        if remaining_slots <= 0:
            step = int(len(prio_df) / max_points_per_param) + 1
            merged_final = prio_df.iloc[::step].head(max_points_per_param)
        else:
            if len(non_prio_df) > 0:
                step = int(len(non_prio_df) / max(1, remaining_slots)) + 1
                sampled_non = non_prio_df.iloc[::step]
                merged_final = pd.concat([prio_df, sampled_non]).sort_values("Measured value from")
            else:
                merged_final = prio_df.sort_values("Measured value from")

            if len(merged_final) > max_points_per_param:
                step2 = int(len(merged_final) / max_points_per_param) + 1
                merged_final = merged_final.iloc[::step2].head(max_points_per_param)
    else:
        merged_final = merged

    return merged_final.reset_index(drop=True)


@st.cache_data
def build_single_param_timeseries_cached(
    df_param_raw: pd.DataFrame,
    param_name: str,
    gap_minutes: int,
    max_points_per_param: int,
    spike_window_points: int,
    rolling_window: int,
    spike_threshold_factor: float,
) -> pd.DataFrame:
    """
    Cached wrapper around build_single_param_timeseries.
    param_name is included explicitly to make the cache key robust.
    """
    return build_single_param_timeseries(
        df_param_raw=df_param_raw,
        gap_minutes=gap_minutes,
        max_points_per_param=max_points_per_param,
        spike_window_points=spike_window_points,
        rolling_window=rolling_window,
        spike_threshold_factor=spike_threshold_factor,
    )


def build_time_mask_for_series(
    timestamps: pd.Series,
    enable_weekly_window: bool,
    weekly_conf: dict | None,
) -> pd.Series:
    """
    Build a boolean mask for a given timestamp series based only on
    an optional Wiederkehrendes Zeitfenster (wöchentlich).
    """
    if timestamps.empty:
        return pd.Series([], dtype=bool, index=timestamps.index)

    mask = pd.Series(True, index=timestamps.index)

    # Weekly window
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


@st.cache_data
def compute_filtered_data(
    data: pd.DataFrame,
    selected_parameters: tuple,
    enable_weekly_window: bool,
    weekly_conf: dict | None,
) -> pd.DataFrame:
    """
    Apply parameter filter + weekly time filter to full data.
    """
    if data.empty:
        return data

    # Parameter filter
    if selected_parameters:
        param_mask = data["Characteristic designation"].isin(selected_parameters)
    else:
        param_mask = pd.Series(True, index=data.index)

    # Time mask (weekly window only)
    time_mask = build_time_mask_for_series(
        data["Measured value from"],
        enable_weekly_window=enable_weekly_window,
        weekly_conf=weekly_conf,
    )

    full_mask = param_mask & time_mask

    return data[full_mask].copy()


def make_filter_key(config: dict) -> str:
    """Create a stable, fully JSON-serializable key without circular references."""

    def clean(obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()

        if hasattr(obj, "isoformat") and not isinstance(obj, (str, bytes)):
            try:
                return obj.isoformat()
            except Exception:
                pass

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
) -> tuple[np.ndarray, list[str]]:
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
# 5) LOAD DATA & BASIC FILTER OPTIONS
# ============================================================
data = load_data(uploaded_files, data_mode)

if data.empty:
    st.error("Keine gültigen Daten nach dem Laden. Bitte überprüfen Sie die Dateien.")
    st.stop()

st.sidebar.header("Filteroptionen")

# ============================================================
# SIDEBAR — APPLY BUTTON (TOP)
# ============================================================
st.sidebar.markdown("## Anwenden / Aktualisieren")
apply_clicked = st.sidebar.button("✅ Filter anwenden / Daten aktualisieren", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("## Filteroptionen")

# ============================================================
# PARAMETER SELECTION WITH “Alle auswählen” BUTTON
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
# WEEKLY WINDOW ONLY (TIME FILTER BLOCK COMMENTED OUT)
# ============================================================

with st.sidebar.expander("📅  Wöchentliches Zeitfenster", expanded=False):
    # Use active_filter_config to drive the default value of the checkbox
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
        st.subheader("Wiederkehrendes Zeitfenster (wöchentlich)")

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
        # Defaults for config if weekly is disabled
        start_weekday_widget = weekday_options[0]
        end_weekday_widget = weekday_options[5]
        start_time_of_day_widget = datetime.strptime("09:00", "%H:%M").time()
        end_time_of_day_widget = datetime.strptime("05:00", "%H:%M").time()

# ============================================================
# ADVANCED MODE
# ============================================================
st.sidebar.markdown("---")
advanced_mode = st.sidebar.checkbox("Erweiterten Analysemodus aktivieren", value=False)

# ============================================================
# 8) APPLY BUTTON & ACTIVE FILTER CONFIG
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

active_config = st.session_state.active_filter_config
filter_key = make_filter_key(active_config)

applied_parameters = list(active_config["selected_parameters"])
enable_weekly_window = active_config["enable_weekly_window"]
weekly_conf_active = active_config["weekly_conf"]

# ============================================================
# 9) FILTERED DATA + PIVOT TABLE
# ============================================================
filtered_data = compute_filtered_data(
    data,
    selected_parameters=tuple(applied_parameters),
    enable_weekly_window=enable_weekly_window,
    weekly_conf=weekly_conf_active,
)

pivot_df_clean = get_pivot_table(filtered_data)

if not filtered_data.empty:
    start_time_final = filtered_data["Measured value from"].min()
    end_time_final = filtered_data["Measured value from"].max()
else:
    start_time_final = data["Measured value from"].min()
    end_time_final = data["Measured value from"].max()

# ============================================================
# ACTIVE FILTER SUMMARY (UNDER TITLE)
# ============================================================
st.markdown(
    f"""
**Aktive Filter**  
Parameter: `{', '.join(applied_parameters) if applied_parameters else 'Alle'}`  
Datenzeitraum: `{start_time_final}` → `{end_time_final}`  
Wöchentliches Zeitfenster: `{"An" if enable_weekly_window else "Aus"}`
"""
)

# ============================================================
# 10) PER-PARAM FULL TIMESERIES (from FULL DATA) + APPLY WEEKLY MASK
# ============================================================
full_param_series = {}
filtered_param_series = {}

for param in applied_parameters:
    df_param_full = data[data["Characteristic designation"] == param]

    ts_full = build_single_param_timeseries_cached(
        df_param_full,
        param,
        gap_minutes=TS_CONFIG["gap_minutes"],
        max_points_per_param=TS_CONFIG["max_points_per_param"],
        spike_window_points=TS_CONFIG["spike_window_points"],
        rolling_window=TS_CONFIG["rolling_window"],
        spike_threshold_factor=TS_CONFIG["spike_threshold_factor"],
    )
    full_param_series[param] = ts_full

    if not ts_full.empty:
        time_mask_series = build_time_mask_for_series(
            ts_full["Measured value from"],
            enable_weekly_window=enable_weekly_window,
            weekly_conf=weekly_conf_active,
        )
        filtered_param_series[param] = ts_full[time_mask_series].reset_index(drop=True)
    else:
        filtered_param_series[param] = ts_full

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
# TAB 0 — Zeitreihendiagramm
# ============================================================
with tabs[0]:
    st.subheader("Zeitreihendiagramm")

    if applied_parameters:
        for param in applied_parameters:
            df_param = filtered_param_series.get(param)
            if df_param is None or df_param.empty:
                continue

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
                xaxis=dict(
                    range=[
                        df_param["Measured value from"].min(),
                        df_param["Measured value from"].max(),
                    ]
                ),
            )

            fig.update_xaxes(autorange=True)

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Bitte mindestens einen Parameter auswählen und 'Filter anwenden' klicken.")

# ============================================================
# TAB 1 — Mehrparameter-Diagramm (Timeline)
# ============================================================
with tabs[1]:
    st.subheader("Mehrparameter-Diagramm (Timeline)")

    options = list(pivot_df_clean.columns)

    all_starts = [ts["Measured value from"].min() for ts in filtered_param_series.values() if not ts.empty]
    all_ends = [ts["Measured value from"].max() for ts in filtered_param_series.values() if not ts.empty]

    if all_starts and all_ends:
        global_min = min(all_starts)
        global_max = max(all_ends)
    else:
        global_min = start_time_final
        global_max = end_time_final

    # Update timeline graph configs safely
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

        # Remove button for each graph
        if st.button(f"Diagramm entfernen {graph['id']}", key=f"remove_graph_{graph['id']}"):
            st.session_state.timeline_graphs = [
                g for g in st.session_state.timeline_graphs if g["id"] != graph["id"]
            ]
            st.experimental_rerun()

        fig = go.Figure()
        for idx, param in enumerate(graph["params"]):
            df_param = filtered_param_series.get(param)
            if df_param is None or df_param.empty:
                continue

            fig.add_trace(go.Scatter(
                x=df_param["Measured value from"],
                y=df_param["Measured value"],
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
            st.info("Wählen Sie mindestens einen Parameter aus, um Verteilungen anzuzeigen.")
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
                st.info("Klicken Sie auf die Schaltfläche, um die Korrelationsmatrix zu berechnen.")
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
                        st.info("Pairplot auf die ersten 6 Parameter beschränkt aus Leistungsgründen.")

                    df_for_pair = pivot_df_clean[pair_cols].dropna()
                    if len(df_for_pair) > 5000:
                        df_for_pair = df_for_pair.sample(5000, random_state=42)
                        st.info("Sampling von 5000 Zeilen für Pairplot aus Leistungsgründen.")

                    g = sns.pairplot(df_for_pair)
                    st.pyplot(g.fig)
                except Exception as e:
                    st.error(f"Pairplot-Fehler: {e}")
            else:
                st.info("Klicken Sie auf die Schaltfläche, um den Pairplot zu berechnen.")
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
                    st.info("Keine gemeinsamen Messpunkte für diese Parameter.")
            else:
                st.warning("Ausgewählte Parameter nicht in den Pivot-Daten verfügbar.")
        else:
            st.info("Bitte mindestens zwei Parameter auswählen und 'Filter anwenden' klicken.")
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
                    ax.set_title(f"PCA auf Parameter: {', '.join(used_params)}")
                    st.pyplot(fig)
                else:
                    st.info("Nicht genügend saubere Daten für PCA nach Filterung / NaN-Entfernung.")
            else:
                st.info("Klicken Sie auf die Schaltfläche, um die PCA zu berechnen.")
        else:
            st.info("Wählen Sie 2+ Parameter und klicken Sie auf 'Filter anwenden'.")

    # ---------------- OUTLIERS ----------------
    with tabs[8]:
        st.subheader("Ausreißererkennung")

        if not pivot_df_clean.empty and applied_parameters:

            method = st.selectbox("Methode zur Ausreißererkennung", ["Z-Score", "IQR"])
            threshold = (
                st.slider("Z-Score Schwellenwert", 1.0, 5.0, 3.0)
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
                    ax.set_title(f"{param} – Ausreißererkennung")
                    st.pyplot(fig)

        else:
            st.info("Nicht genügend Daten.")

