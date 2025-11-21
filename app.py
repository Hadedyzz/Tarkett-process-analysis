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

# ============================================================
# 1) APP CONFIGURATION
# ============================================================
st.set_page_config(layout="wide")

# Session state initialization for include/exclude windows
if "include_windows" not in st.session_state:
    st.session_state.include_windows = []

if "exclude_windows" not in st.session_state:
    st.session_state.exclude_windows = []


# ============================================================
# 2) DATA SOURCE TOGGLE (HYDRA / SPS)
# ============================================================
data_mode = st.radio(
    "Select Data Source Format",
    ["Hydra", "SPS"],
    horizontal=True
)

st.title(f"Process Parameter Analysis ({data_mode} Data)")


# ============================================================
# 3) DATA LOADING & CORE HELPERS
# ============================================================
@st.cache_data
def load_data(uploaded_files, mode):
    """
    Load one or more CSV files, normalize column names,
    parse datetime (dayfirst), convert to numeric, and
    silently remove duplicated measurements.
    Duplicate = same (Characteristic designation, timestamp).
    """
    frames = []

    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file, sep=";", decimal=",")

        # SPS requires renaming
        if mode == "SPS":
            df = df.rename(columns={
                "VarName": "Characteristic designation",
                "VarValue": "Measured value",
                "TimeString": "Measured value from"
            })

        # Force numeric for measured values
        if "Measured value" in df.columns:
            df["Measured value"] = pd.to_numeric(df["Measured value"], errors="coerce")

        # Parse datetime (dd.mm.yyyy supported via dayfirst=True)
        if "Measured value from" in df.columns:
            df["Measured value from"] = pd.to_datetime(
                df["Measured value from"],
                dayfirst=True,
                errors="coerce",
                cache=True
            )

        required_cols = ["Characteristic designation", "Measured value", "Measured value from"]
        df = df.dropna(subset=required_cols)

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    # Combine all files
    data = pd.concat(frames, ignore_index=True)

    # Remove duplicated measurements silently
    data = data.drop_duplicates(
        subset=["Characteristic designation", "Measured value from"],
        keep="last"
    )

    return data


def filter_data(data, parameters, start, end):
    """Filter by selected parameters and time range."""
    mask_time = (data["Measured value from"] >= start) & (data["Measured value from"] <= end)
    if not parameters:
        return data[mask_time]
    mask_param = data["Characteristic designation"].isin(parameters)
    return data[mask_param & mask_time]


@st.cache_data
def get_pivot_table(filtered_data: pd.DataFrame) -> pd.DataFrame:
    """Pivot for scatter / correlation / PCA etc."""
    if filtered_data.empty:
        return pd.DataFrame()
    df_unique = filtered_data.drop_duplicates(
        subset=["Measured value from", "Characteristic designation"],
        keep="last"
    )
    return df_unique.pivot(
        index="Measured value from",
        columns="Characteristic designation",
        values="Measured value"
    )


@st.cache_data
def build_param_timeseries(
    filtered_data: pd.DataFrame,
    params: list[str],
    gap_minutes: int = 10,
    max_points_per_param: int = 7200,
    spike_window_points: int = 5,
    rolling_window: int = 10,
    spike_threshold_factor: float = 2.0,
) -> dict[str, pd.DataFrame]:
    """
    Build per-parameter time series for plotting with:
      - dynamic time-based downsampling (1/5/10/30 min)
      - dynamic spike detection (based on rolling std)
      - local high-resolution around spikes (± spike_window_points)
      - hard cap of max_points_per_param per parameter
      - gaps inserted for time jumps > gap_minutes
    """

    result: dict[str, pd.DataFrame] = {}
    if filtered_data.empty or not params:
        return result

    gap_threshold = pd.Timedelta(minutes=gap_minutes)

    for p in params:
        df_p_raw = filtered_data[filtered_data["Characteristic designation"] == p].copy()
        if df_p_raw.empty:
            continue

        # Ensure numeric type (safety if anything slipped through)
        df_p_raw["Measured value"] = pd.to_numeric(df_p_raw["Measured value"], errors="coerce")
        df_p_raw = df_p_raw.dropna(subset=["Measured value"])

        if df_p_raw.empty:
            continue

        df_p_raw = df_p_raw.sort_values("Measured value from")

        # -----------------------------
        # 1) Choose base resample interval (global span)
        # -----------------------------
        t_start = df_p_raw["Measured value from"].iloc[0]
        t_end = df_p_raw["Measured value from"].iloc[-1]
        total_days = (t_end - t_start).total_seconds() / 86400.0

        if total_days < 7:
            interval = "1min"
        elif total_days < 28:
            interval = "5min"
        elif total_days < 84:
            interval = "10min"
        else:
            interval = "30min"

        # -----------------------------
        # 2) Base resample (uniform time grid) on numeric column only
        # -----------------------------
        df_idx = df_p_raw.set_index("Measured value from")[["Measured value"]]
        base = df_idx.resample(interval).mean()
        base = base.reset_index()  # columns: Measured value from, Measured value

        # -----------------------------
        # 3) Dynamic spike detection on original series
        # -----------------------------
        s_val = df_p_raw["Measured value"]
        diff = s_val.diff().abs()

        rolling_std = s_val.rolling(window=rolling_window, min_periods=5).std()
        threshold_series = spike_threshold_factor * rolling_std

        spike_mask = diff > threshold_series
        spike_indices = np.where(spike_mask.fillna(False).to_numpy())[0]

        # Collect high-resolution segments around spikes
        spike_segments = []
        n_raw = len(df_p_raw)
        for idx in spike_indices:
            start_i = max(0, idx - spike_window_points)
            end_i = min(n_raw - 1, idx + spike_window_points)
            seg = df_p_raw.iloc[start_i:end_i + 1]
            spike_segments.append(seg)

        if spike_segments:
            high_res = pd.concat(spike_segments).drop_duplicates(
                subset=["Measured value from"],
                keep="last"
            )
        else:
            high_res = pd.DataFrame(columns=df_p_raw.columns)

        # -----------------------------
        # 4) Merge base resample + spike windows
        # -----------------------------
        merged = base.copy()
        if not high_res.empty:
            merged = pd.concat(
                [
                    merged,
                    high_res[["Measured value from", "Measured value"]]
                ],
                ignore_index=True
            )

        merged = merged.drop_duplicates(
            subset=["Measured value from"],
            keep="last"
        ).sort_values("Measured value from")

        # -----------------------------
        # 5) Insert gaps for large time jumps
        # -----------------------------
        merged["time_diff"] = merged["Measured value from"].diff()
        merged.loc[merged["time_diff"] > gap_threshold, "Measured value"] = None
        merged = merged.drop(columns=["time_diff"])

        # -----------------------------
        # 6) Enforce max_points_per_param (keep spike points preferentially)
        # -----------------------------
        if len(merged) > max_points_per_param:
            # Priority timestamps = spike-window times
            if not high_res.empty:
                spike_times = set(high_res["Measured value from"].unique())
                is_prio = merged["Measured value from"].isin(spike_times)
            else:
                spike_times = set()
                is_prio = pd.Series(False, index=merged.index)

            prio_df = merged[is_prio]
            non_prio_df = merged[~is_prio]

            remaining_slots = max_points_per_param - len(prio_df)

            if remaining_slots <= 0:
                # Too many priority points, downsample them uniformly
                step = int(len(prio_df) / max_points_per_param) + 1
                merged_final = prio_df.iloc[::step].head(max_points_per_param)
            else:
                # Take all spike points, downsample the rest
                if len(non_prio_df) > 0:
                    step = int(len(non_prio_df) / remaining_slots) + 1
                    sampled_non = non_prio_df.iloc[::step]
                    merged_final = pd.concat([prio_df, sampled_non]).sort_values("Measured value from")
                else:
                    merged_final = prio_df.sort_values("Measured value from")

                if len(merged_final) > max_points_per_param:
                    step2 = int(len(merged_final) / max_points_per_param) + 1
                    merged_final = merged_final.iloc[::step2].head(max_points_per_param)
        else:
            merged_final = merged

        result[p] = merged_final.reset_index(drop=True)

    return result


# ============================================================
# 4) SIDEBAR – FILE UPLOAD
# ============================================================
uploaded_files = st.file_uploader(
    "Upload one or more CSV files",
    type=["csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload at least one CSV file to begin.")
    st.stop()


# ============================================================
# 5) LOAD DATA & BASIC FILTER OPTIONS
# ============================================================
data = load_data(uploaded_files, data_mode)

st.sidebar.header("Filter Options")

all_parameters = sorted(data["Characteristic designation"].unique())

selected_parameters = st.sidebar.multiselect(
    "Select Parameters",
    options=all_parameters,
    default=[]
)


# ============================================================
# 6) SIMPLE OPTIONAL TIME FILTER SYSTEM
# ============================================================
st.sidebar.markdown("---")
enable_time_filter = st.sidebar.checkbox("Enable Time Filtering", value=False)

min_dt = data["Measured value from"].min().to_pydatetime()
max_dt = data["Measured value from"].max().to_pydatetime()

if enable_time_filter:

    st.sidebar.subheader("Add Time Window")

    window_type = st.sidebar.selectbox("Window Type", ["Include", "Exclude"])

    w_start_date = st.sidebar.date_input("Start Date", min_dt.date())
    w_start_time = st.sidebar.time_input("Start Time", min_dt.time())
    w_end_date = st.sidebar.date_input("End Date", max_dt.date())
    w_end_time = st.sidebar.time_input("End Time", max_dt.time())

    w_start = datetime.combine(w_start_date, w_start_time)
    w_end = datetime.combine(w_end_date, w_end_time)

    if st.sidebar.button("➕ Add Window"):
        if window_type == "Include":
            st.session_state.include_windows.append({"start": w_start, "end": w_end})
        else:
            st.session_state.exclude_windows.append({"start": w_start, "end": w_end})

    # Show include windows
    if st.session_state.include_windows:
        st.sidebar.markdown("### Include Windows")
        for i, win in enumerate(st.session_state.include_windows):
            st.sidebar.write(f"{i+1}) {win['start']} → {win['end']}")
            if st.sidebar.button(f"Remove Include {i+1}", key=f"inc_remove_{i}"):
                st.session_state.include_windows.pop(i)
                st.experimental_rerun()

    # Show exclude windows
    if st.session_state.exclude_windows:
        st.sidebar.markdown("### Exclude Windows")
        for i, win in enumerate(st.session_state.exclude_windows):
            st.sidebar.write(f"{i+1}) {win['start']} → {win['end']}")
            if st.sidebar.button(f"Remove Exclude {i+1}", key=f"exc_remove_{i}"):
                st.session_state.exclude_windows.pop(i)
                st.experimental_rerun()


# ============================================================
# 7) WEEKLY WINDOW FILTER
# ============================================================
st.sidebar.markdown("---")
enable_weekly_window = st.sidebar.checkbox("Enable Weekly Window Filter", value=False)

if enable_weekly_window:

    st.sidebar.subheader("Define Recurring Weekly Window")

    weekday_options = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]

    start_weekday = st.sidebar.selectbox("Start Weekday", weekday_options, index=0)
    start_time_of_day = st.sidebar.time_input(
        "Start Time of Day",
        datetime.strptime("09:00", "%H:%M").time()
    )

    end_weekday = st.sidebar.selectbox("End Weekday", weekday_options, index=5)
    end_time_of_day = st.sidebar.time_input(
        "End Time of Day",
        datetime.strptime("05:00", "%H:%M").time()
    )


# Advanced mode toggle
st.sidebar.markdown("---")
advanced_mode = st.sidebar.checkbox("Enable Advanced Analysis Mode", value=False)


# ============================================================
# 8) APPLY TIME-WINDOW LOGIC
# ============================================================
# Base full-range limits
start_time_final = data["Measured value from"].min()
end_time_final = data["Measured value from"].max()

# Step 1 — Base filter (parameters + full range)
filtered_data = filter_data(data, selected_parameters, start_time_final, end_time_final)

# Step 2 — Include windows (add rows)
if enable_time_filter and st.session_state.include_windows:
    include_frames = []
    for win in st.session_state.include_windows:
        extra = data[
            (data["Measured value from"] >= win["start"]) &
            (data["Measured value from"] <= win["end"])
        ]
        include_frames.append(extra)
    if include_frames:
        filtered_data = pd.concat([filtered_data] + include_frames).drop_duplicates()

# Step 3 — Exclude windows (remove rows)
if enable_time_filter and st.session_state.exclude_windows:
    for win in st.session_state.exclude_windows:
        filtered_data = filtered_data[
            ~(
                (filtered_data["Measured value from"] >= win["start"]) &
                (filtered_data["Measured value from"] <= win["end"])
            )
        ]

# Step 4 — Weekly window filter
if enable_weekly_window:

    weekday_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    start_idx = weekday_map[start_weekday]
    end_idx = weekday_map[end_weekday]

    def to_fraction(day, t):
        return day + (t.hour + t.minute / 60 + t.second / 3600) / 24.0

    start_f = to_fraction(start_idx, start_time_of_day)
    end_f = to_fraction(end_idx, end_time_of_day)

    ts_f = (
        filtered_data["Measured value from"].dt.weekday +
        filtered_data["Measured value from"].dt.hour / 24 +
        filtered_data["Measured value from"].dt.minute / (24 * 60) +
        filtered_data["Measured value from"].dt.second / (24 * 3600)
    )

    if start_f <= end_f:
        filtered_data = filtered_data[(ts_f >= start_f) &
                                      (ts_f <= end_f)]
    else:
        filtered_data = filtered_data[(ts_f >= start_f) |
                                      (ts_f <= end_f)]


# ============================================================
# 9) BUILD PIVOT TABLE + TIMESERIES
# ============================================================
pivot_df_clean = get_pivot_table(filtered_data)

if not filtered_data.empty:
    start_time_final = filtered_data["Measured value from"].min()
    end_time_final = filtered_data["Measured value from"].max()

# Precompute per-parameter timeseries with smart downsampling
param_timeseries = build_param_timeseries(
    filtered_data,
    list(pivot_df_clean.columns),
    gap_minutes=10,
    max_points_per_param=7200,
    spike_window_points=5,
    rolling_window=10,
    spike_threshold_factor=2.0
)


# ============================================================
# TABS
# ============================================================
if advanced_mode:
    tab_names = [
        "Interactive Time Series", "Interactive Timeline",
        "Scatter Plot", "Pair Plot",
        "Correlation Analysis", "Distribution Plots",
        "Time-Shifted Correlation", "PCA",
        "Outlier Analysis"
    ]
else:
    tab_names = [
        "Interactive Time Series", "Interactive Timeline",
        "Scatter Plot", "Pair Plot"
    ]

tabs = st.tabs(tab_names)


# ============================================================
# TAB 0 — Interactive Time Series
# ============================================================
with tabs[0]:
    st.subheader("Interactive Time Series")

    if not filtered_data.empty and selected_parameters:
        for param in selected_parameters:
            df_param = param_timeseries.get(param)
            if df_param is None or df_param.empty:
                continue

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_param["Measured value from"],
                y=df_param["Measured value"],
                mode="lines+markers",
                name=param,
                connectgaps=False,
                hovertemplate=f"<b>{param}</b><br>%{{x}}<br>Value: %{{y}}<extra></extra>"
            ))

            fig.update_layout(
                title=param,
                xaxis_title="Time",
                yaxis_title="Value",
                hovermode="x unified",
                xaxis=dict(range=[start_time_final, end_time_final])
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one parameter.")


# ============================================================
# TAB 1 — Interactive Timeline
# ============================================================
with tabs[1]:
    st.subheader("Interactive Timeline")

    if "timeline_graphs" not in st.session_state:
        st.session_state.timeline_graphs = [{"id": 1, "params": selected_parameters.copy()}]
        st.session_state.next_timeline_graph_id = 2

    for graph in st.session_state.timeline_graphs:
        graph["params"] = st.multiselect(
            f"Timeline Graph {graph['id']}: Select parameters",
            options=list(pivot_df_clean.columns),
            default=graph["params"],
            key=f"timeline_graph_{graph['id']}"
        )

    if st.button("Add Timeline Graph"):
        st.session_state.timeline_graphs.append(
            {"id": st.session_state.next_timeline_graph_id, "params": selected_parameters.copy()}
        )
        st.session_state.next_timeline_graph_id += 1

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    for graph in st.session_state.timeline_graphs:
        if not graph["params"]:
            continue

        fig = go.Figure()
        for idx, param in enumerate(graph["params"]):
            df_param = param_timeseries.get(param)
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
            title=f"Timeline Graph {graph['id']}",
            xaxis_title="Time",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5
            ),
            margin=dict(b=80)
        )

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 2 — Scatter
# ============================================================
with tabs[2]:
    st.subheader("Scatter Plot")
    if len(selected_parameters) >= 2 and not pivot_df_clean.empty:
        x_param = st.selectbox("X Parameter", selected_parameters)
        y_param = st.selectbox("Y Parameter", selected_parameters)

        if x_param != y_param and x_param in pivot_df_clean.columns and y_param in pivot_df_clean.columns:
            df_scatter = pivot_df_clean[[x_param, y_param]].dropna()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(df_scatter[x_param], df_scatter[y_param])
            ax.set_xlabel(x_param)
            ax.set_ylabel(y_param)
            st.pyplot(fig)
        else:
            st.warning("Choose two different parameters.")
    else:
        st.info("Select at least two parameters.")


# ============================================================
# TAB 3 — Pair Plot
# ============================================================
with tabs[3]:
    st.subheader("Pair Plot")
    if not pivot_df_clean.empty:
        try:
            sns_plot = sns.pairplot(pivot_df_clean.dropna())
            st.pyplot(sns_plot)
        except Exception as e:
            st.error(f"Pair plot error: {e}")
    else:
        st.info("Not enough data.")


# ============================================================
# ADVANCED MODE
# ============================================================
if advanced_mode:

    # ---------------- CORRELATION ----------------
    with tabs[4]:
        st.subheader("Correlation Matrix")
        if not pivot_df_clean.empty:
            corr = pivot_df_clean.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough data.")

    # ---------------- DISTRIBUTIONS ----------------
    with tabs[5]:
        st.subheader("Distribution Plots")
        if not pivot_df_clean.empty:
            for param in selected_parameters:
                if param in pivot_df_clean:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(pivot_df_clean[param].dropna(), kde=True, ax=ax)
                    st.pyplot(fig)
        else:
            st.info("Not enough data.")

    # ---------------- TIME SHIFT ----------------
    with tabs[6]:
        st.subheader("Time-Shifted Correlation")
        if len(selected_parameters) >= 2 and not pivot_df_clean.empty:
            p1 = st.selectbox("Parameter X", selected_parameters, key="tsc_p1")
            p2 = st.selectbox("Parameter Y", selected_parameters, key="tsc_p2")
            lag = st.slider("Lag", -20, 20, 0)

            if p1 in pivot_df_clean.columns and p2 in pivot_df_clean.columns:
                df_lag = pivot_df_clean[[p1, p2]].dropna()
                shifted = df_lag[p2].shift(lag)
                corr_val = df_lag[p1].corr(shifted)
                st.write(f"Correlation (lag {lag}): **{corr_val:.3f}**")
            else:
                st.warning("Selected parameters not available in pivot data.")
        else:
            st.info("Not enough data.")

    # ---------------- PCA ----------------
    with tabs[7]:
        st.subheader("PCA")
        if len(selected_parameters) > 1 and not pivot_df_clean.empty:
            valid_params = [p for p in selected_parameters if p in pivot_df_clean.columns]
            df_pca = pivot_df_clean[valid_params].dropna()
            if not df_pca.empty:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(df_pca)
                pca = PCA(n_components=2)
                comp = pca.fit_transform(scaled)

                fig, ax = plt.subplots()
                ax.scatter(comp[:, 0], comp[:, 1])
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                st.pyplot(fig)
            else:
                st.info("Not enough data after dropping NaNs.")
        else:
            st.info("Choose 2+ parameters.")

    # ---------------- OUTLIERS ----------------
    with tabs[8]:
        st.subheader("Outlier Analysis")

        if not pivot_df_clean.empty and selected_parameters:

            method = st.selectbox("Outlier detection method", ["Z-Score", "IQR"])
            threshold = (
                st.slider("Z-Score Threshold", 1.0, 5.0, 3.0)
                if method == "Z-Score"
                else st.slider("IQR Multiplier", 1.0, 3.0, 1.5)
            )

            for param in selected_parameters:
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

                fig, ax = plt.subplots()
                sns.histplot(series, ax=ax)
                if not outliers.empty:
                    sns.histplot(outliers, color="red", ax=ax)
                ax.set_title(f"{param} – Outlier Analysis")
                st.pyplot(fig)

        else:
            st.info("Not enough data.")
