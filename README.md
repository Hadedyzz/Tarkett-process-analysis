# Process Parameter Analysis Tool (Hydra / SPS)

A Streamlit app to interactively explore process parameters from **Hydra** or **SPS/PLC** data:

- Supports **one or multiple CSV files**
- Handles both **Hydra-style** and **SPS-style** column names (via format toggle)
- Automatically **normalizes columns**, **parses timestamps** (`dayfirst=True`), and
  silently **removes duplicate measurements** (same parameter + timestamp)
- Powerful **time filtering**:
  - Include / exclude arbitrary time windows
  - Recurring **weekly window** (e.g. Mon 06:00 → Fri 22:00)
- Smart **downsampling & visualization**:
  - Dynamic resampling interval based on total time span
  - Maximum **7200 points per parameter**
  - **Spike-aware** downsampling (local high resolution around big jumps)
  - Gaps in plots when time jumps are too large
- Advanced analysis (optional):
  - Correlation matrix
  - Distribution plots
  - Time-shifted correlation
  - PCA
  - Outlier analysis

---

## 1. Data Expectations

The app expects **CSV files** with semicolon separators (`;`) and comma decimals (`,`).

### Hydra format

If you select **Hydra** as data source, your CSV should already contain these columns:

- `Characteristic designation`
- `Measured value`
- `Measured value from`  (datetime, e.g. `31.10.2025 14:32:00`)

### SPS format

If you select **SPS**, the app expects PLC-style columns and will rename them:

- `VarName`   → `Characteristic designation`
- `VarValue`  → `Measured value`
- `TimeString` → `Measured value from`

The app then:

- Converts `Measured value` to numeric (`to_numeric(errors="coerce")`)
- Parses `Measured value from` with `dayfirst=True` and `errors="coerce"`
- Drops rows missing any of these three core columns
- Removes duplicate measurements based on (`Characteristic designation`, `Measured value from`)
  and keeps the **last** occurrence.

---

## 2. Main Features

### 2.1 Data Source Toggle

At the top of the app:

- **Hydra** mode → assume columns already match final naming
- **SPS** mode → auto-rename PLC columns to the final naming

The title shows which mode is active:
> `Process Parameter Analysis (Hydra Data)` or `Process Parameter Analysis (SPS Data)`

---

### 2.2 File Upload

In the main area:

- Upload **one or more** CSV files.
- All files are concatenated into one dataset after normalization and deduplication.

---

### 2.3 Filter Options (Sidebar)

#### Parameter selection

- **Multiselect** of `Characteristic designation` values.
- If no parameters are selected → base filter uses only the time range (effectively all parameters).

---

### 2.4 Time Window Filters

Toggle: **“Enable Time Filtering”**

You can add **Include** and **Exclude** windows:

- **Include window**  
  Adds rows from the selected time window on top of the base filtered data.
- **Exclude window**  
  Removes rows inside the selected time window from the current filtered data.

You can add multiple windows of both types and delete them individually.

---

### 2.5 Weekly Window Filter

Toggle: **“Enable Weekly Window Filter”**

Define a recurring weekly interval:

- Start weekday + time (e.g. `Monday, 09:00`)
- End weekday + time (e.g. `Saturday, 05:00`)

The app converts timestamps into a **fraction of week** and keeps only points inside this weekly window. Wrap-around (e.g. Fri evening → Mon morning) is supported.

---

## 3. Downsampling & Time Series Logic

For each parameter, the app:

1. Sorts the data by time.
2. Determines the **total timespan** of the parameter:
   - `< 1 week`  → resample interval = **1 minute**
   - `1–4 weeks` → **5 minutes**
   - `4–12 weeks` → **10 minutes**
   - `> 12 weeks` → **30 minutes**
3. Resamples to a uniform grid using `resample(interval).mean()` on the numeric `Measured value`.
4. Detects **spikes** based on a rolling standard deviation:
   - compute `diff = |Δvalue|`
   - compute rolling `std` (window size configurable, default 10 points)
   - a point is a spike if `diff > spike_threshold_factor * rolling_std`
5. Around each spike, add a **high-resolution window**:
   - keep ± `spike_window_points` original points (default: 5) before & after each spike.
6. Merge the **base resampled series** + **all spike windows** and sort by time.
7. Insert **gaps**:
   - if the time difference between neighboring points is greater than `gap_minutes` (default 10),
     set `Measured value` to `None` so Plotly draws a break in the line.
8. Enforce **max 7200 points per parameter**:
   - Priority: keep spike-related points first.
   - Remaining capacity is filled with uniformly downsampled non-spike points.

This downsampled & gap-aware series is used in:

- **Interactive Time Series**
- **Interactive Timeline**

---

## 4. Tabs Overview

### 4.1 Interactive Time Series

- One chart per selected parameter.
- Uses the precomputed, downsampled, spike-preserving time series.
- Breaks lines on large time gaps (no misleading jumps).

### 4.2 Interactive Timeline

- You can create multiple **Timeline Graphs**.
- Each graph has its own multiselect of parameters.
- Same downsampled time series as in the previous tab.
- Legend is shown **below** each plot for better readability.

### 4.3 Scatter Plot

- Choose X and Y parameters from your selection.
- Plots scatter using the pivoted data (no downsampling here, just pivot across timestamps).

### 4.4 Pair Plot

- Seaborn `pairplot` on the pivoted data (for all pivot columns).
- Visualizes pairwise relationships between parameters.

---

### 4.5 Advanced Mode

If **“Enable Advanced Analysis Mode”** is checked, extra tabs appear:

- **Correlation Analysis**  
  Heatmap of correlation matrix across parameters.
- **Distribution Plots**  
  Histogram + KDE for each selected parameter.
- **Time-Shifted Correlation**  
  Pick two parameters and a lag in steps (index shift) and see correlation vs lag.
- **PCA**  
  Principal component analysis on selected parameters (2 components, scatter plot).
- **Outlier Analysis**  
  Per-parameter histogram with outliers highlighted using Z-score or IQR rules.

---

## 5. Installation

### 5.1 Prerequisites

- **Python 3.10+** (tested with 3.11)
- A virtual environment is recommended.

### 5.2 Setup

```bash
# create and activate a virtual environment (example for Windows PowerShell)
python -m venv .venv
.venv\Scripts\activate

# or on Linux / macOS
# python -m venv .venv
# source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
