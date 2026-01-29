import io
import math
from datetime import datetime, timedelta, date, time
from collections import defaultdict
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------
# Streamlit page config and title
# -----------------------------------
st.set_page_config(page_title="SI & Price Dashboard", layout="wide")
st.title("System Imbalance & Imbalance Price — Dashboard")
st.caption("Cumulative & instantaneous SI, ΔSI per minute, and quarter-hour stats (live from Elia Open Data).")

# -----------------------------------
# API config (Elia Opendatasoft Explore v2.1)
# -----------------------------------
DATASET_ID = "ods133"
BASE_URL = f"https://opendata.elia.be/api/explore/v2.1/catalog/datasets/{DATASET_ID}/records"
HEADERS = {"Accept": "application/json"}
API_TIMEZONE = "Europe/Brussels"  # interpret query times & records in Belgium time (DST-safe)

LOCAL_TZ = ZoneInfo("Europe/Brussels")  # for client-side conversion

# -----------------------------------
# Sidebar controls for time window
# Default = yesterday in BE time, full day (00:00–23:00)
# -----------------------------------
st.sidebar.header("Controls")

now_local = datetime.now(LOCAL_TZ)
yesterday_date = (now_local - timedelta(days=1)).date()

default_start_date = yesterday_date
default_end_date = yesterday_date
default_start_time = time(0, 0)
default_end_time = time(23, 0)

start_date = st.sidebar.date_input("Start date", value=default_start_date)
start_hour = st.sidebar.number_input("Start hour (0–23)", min_value=0, max_value=23, value=default_start_time.hour)
end_date = st.sidebar.date_input("End date", value=default_end_date)
end_hour = st.sidebar.number_input("End hour (0–23)", min_value=0, max_value=23, value=default_end_time.hour)

# Build full local datetimes (inclusive)
start_dt_local = datetime.combine(start_date, time(start_hour, 0)).replace(tzinfo=LOCAL_TZ)
end_dt_local = datetime.combine(end_date, time(end_hour, 0)).replace(tzinfo=LOCAL_TZ)

st.sidebar.markdown("---")
st.sidebar.write("**Notes**")
st.sidebar.write("- Data is fetched live from Elia Open Data `ods133` (minute-level).")
st.sidebar.write("- All times are shown in Belgium local time (Europe/Brussels), DST handled automatically.")
st.sidebar.write("- Fields parsed: `systemimbalance`, `imbalanceprice`, and `datetime` from the API JSON.")

# -----------------------------------
# Fetch data from Elia Open Data (ODS133)
# -----------------------------------
@st.cache_data(ttl=300, show_spinner=True)
def fetch_ods133(start_local_dt: datetime, end_local_dt: datetime) -> pd.DataFrame:
    """
    Fetch minute data from Elia Open Data (dataset ods133) in the given BE-local interval.
    Pagination is applied safely and records are ordered ascending by datetime.
    Returned DataFrame columns: Datetime (local, naive), System imbalance, Imbalance Price.
    """
    # We instruct the API to interpret our WHERE datetimes in Europe/Brussels (API_TIMEZONE).
    # According to the Explore API v2.1 docs, the 'timezone' parameter affects both query and records. [2](https://help.opendatasoft.com/apis/ods-explore-v2/)
    params_base = {
        "where": f"datetime >= '{start_local_dt.isoformat()}' AND datetime <= '{end_local_dt.isoformat()}'",
        "order_by": "datetime asc",
        "timezone": API_TIMEZONE,
        "limit": 100,  # safe default for records endpoint
    }

    results = []
    offset = 0
    total_count = None

    # Basic pagination loop
    while True:
        params = params_base.copy()
        params["offset"] = offset

        resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        batch = payload.get("results", [])
        if total_count is None:
            total_count = payload.get("total_count", 0)

        results.extend(batch)

        # Stop when we reach or exceed total_count, or no more records
        offset += params["limit"]
        if not batch or offset >= total_count:
            break

        # Extra guard for very large windows (the records endpoint is limited vs exports) [2](https://help.opendatasoft.com/apis/ods-explore-v2/)
        if offset > 20000:
            # We should never hit this for a single day of minute data (~1440 rows).
            break

    if not results:
        return pd.DataFrame(columns=["Datetime", "System imbalance", "Imbalance Price"])

    df = pd.DataFrame(results)

    # Parse numeric fields consistently
    df["System imbalance"] = pd.to_numeric(df.get("systemimbalance"), errors="coerce")
    df["Imbalance Price"] = pd.to_numeric(df.get("imbalanceprice"), errors="coerce")

    # Parse and convert datetime
    # The API returns ISO datetimes with timezone; we convert them to BE local (DST) and drop tzinfo for plotting.
    # If the API already applied API_TIMEZONE to records, this still stays correct (pandas respects tz offsets).
    dt_utc = pd.to_datetime(df.get("datetime"), utc=True, errors="coerce")
    df["Datetime"] = dt_utc.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)

    # Drop invalid rows and order
    df = df.dropna(subset=["Datetime", "System imbalance", "Imbalance Price"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df


# Load API data (cached)
try:
    data = fetch_ods133(start_dt_local, end_dt_local)
except Exception as e:
    st.error(f"Failed to fetch from Elia Open Data: {e}")
    st.stop()

# Local filter (redundant but keeps your original flow)
mask = (data["Datetime"] >= start_dt_local.replace(tzinfo=None)) & (data["Datetime"] <= end_dt_local.replace(tzinfo=None))
df_filtered = data.loc[mask].sort_values("Datetime").reset_index(drop=True)

if df_filtered.empty:
    st.warning("No data found for the selected time range.")
    st.stop()

# -----------------------------------
# Compute series for plots (same logic as your original)
# -----------------------------------
timestamps = df_filtered["Datetime"].tolist()
cumulative_si = df_filtered["System imbalance"].tolist()
imbalance_price = df_filtered["Imbalance Price"].tolist()

# Instantaneous SI reconstruction per minute within 15-min blocks
instantaneous_si = []
for i, cumul in enumerate(cumulative_si):
    minute = timestamps[i].minute
    block_start = (minute // 15) * 15
    block_index = minute - block_start + 1
    if block_index == 1:
        inst = cumul
    else:
        prev_cumul = cumulative_si[i - 1]
        inst = block_index * cumul - (block_index - 1) * prev_cumul
    instantaneous_si.append(inst)

# Minute-by-minute delta of instantaneous SI
delta_si = []
for i in range(1, len(instantaneous_si)):
    delta_si.append(instantaneous_si[i] - instantaneous_si[i - 1])

# Group instantaneous SI by quarter-hour start, preserving chronological order
qh_to_series = defaultdict(list)  # {qh_start: [(dt, val), ...]}
for i, dt in enumerate(timestamps):
    qh_start = dt.replace(minute=(dt.minute // 15) * 15, second=0, microsecond=0)
    qh_to_series[qh_start].append((dt, instantaneous_si[i]))

# Sort quarter-hours chronologically and extract ordered values
qh_sorted_items = sorted(qh_to_series.items())  # [(qh_start, [(dt, val), ...]), ...]
qh_ordered_vals = []
qh_starts = []
for qh, pairs in qh_sorted_items:
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    qh_ordered_vals.append([v for (_, v) in pairs_sorted])
    qh_starts.append(qh)

# Helper: longest run amplitudes (strictly increasing/decreasing)
def longest_run_amplitudes(vals):
    """
    Amplitudes (not lengths) of the longest strictly increasing and strictly decreasing
    consecutive runs within a list of values. Returns (max_inc_amp, max_dec_amp).
    """
    n = len(vals)
    if n < 2:
        return 0.0, 0.0

    # Longest strictly increasing run amplitude
    inc_start = vals[0]
    inc_prev = vals[0]
    max_inc_amp = 0.0
    for x in vals[1:]:
        if x > inc_prev:
            pass
        else:
            amp = inc_prev - inc_start
            if amp > max_inc_amp:
                max_inc_amp = amp
            inc_start = x
        inc_prev = x
    last_amp = inc_prev - inc_start
    if last_amp > max_inc_amp:
        max_inc_amp = last_amp

    # Longest strictly decreasing run amplitude
    dec_start = vals[0]
    dec_prev = vals[0]
    max_dec_amp = 0.0
    for x in vals[1:]:
        if x < dec_prev:
            pass
        else:
            amp = dec_start - dec_prev
            if amp > max_dec_amp:
                max_dec_amp = amp
            dec_start = x
        dec_prev = x
    last_amp = dec_start - dec_prev
    if last_amp > max_dec_amp:
        max_dec_amp = last_amp

    if max_inc_amp < 0:
        max_inc_amp = 0.0
    if max_dec_amp < 0:
        max_dec_amp = 0.0
    return float(max_inc_amp), float(max_dec_amp)

qh_longest_inc_amp = []
qh_longest_dec_amp = []
for vals in qh_ordered_vals:
    inc_amp, dec_amp = longest_run_amplitudes(vals)
    qh_longest_inc_amp.append(inc_amp)
    qh_longest_dec_amp.append(dec_amp)

# Max/Min per quarter-hour
qh_max = [max(vals) if len(vals) else None for vals in qh_ordered_vals]
qh_min = [min(vals) if len(vals) else None for vals in qh_ordered_vals]

# -----------------------------------
# Figure 1: 4-panel time series
# -----------------------------------
fig, axs = plt.subplots(4, 1, figsize=(18, 18), sharex=True)

# 1) Cumulative + Instantaneous SI
axs[0].step(timestamps, cumulative_si, where="post", label="Cumulative SI (API values)")
axs[0].step(timestamps, instantaneous_si, where="post", label="Instantaneous SI", alpha=0.7)
axs[0].set_title("Cumulative SI and Instantaneous SI")
axs[0].set_ylabel("System Imbalance")
axs[0].legend()
axs[0].grid(True)

# 2) Lines per quarter-hour (Δ longest increase/decrease amplitudes)
ax_lines = axs[1]
if qh_starts:
    first_label_done = False
    for i, qh in enumerate(qh_starts):
        start_time = qh
        end_time = qh + timedelta(minutes=15)
        inc_amp = qh_longest_inc_amp[i]
        dec_amp = qh_longest_dec_amp[i]

        ax_lines.plot(
            [start_time, end_time], [inc_amp, inc_amp],
            color="green", linewidth=3,
            label="Δ longest consecutive increase" if not first_label_done else None
        )
        ax_lines.plot(
            [start_time, end_time], [dec_amp, dec_amp],
            color="red", linewidth=3,
            label="Δ longest consecutive decrease" if not first_label_done else None
        )
        first_label_done = True

    ax_lines.set_title("Per quarter-hour: amplitudes of longest consecutive ↑ and ↓ (ΔSI) — lines")
    ax_lines.set_ylabel("ΔSI amplitude")
    ax_lines.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax_lines.legend()
    ax_lines.xaxis_date()
else:
    ax_lines.text(0.5, 0.5, "No quarter-hour data available in the selected window",
                  ha="center", va="center", transform=ax_lines.transAxes)
    ax_lines.set_axis_off()

# 3) Minute-by-minute change of instantaneous SI
axs[2].step(timestamps[1:], delta_si, where="post",
            label="Minute-by-Minute Change of Instantaneous SI", color="purple")
axs[2].set_title("Minute-by-Minute Change of Instantaneous SI")
axs[2].set_ylabel("Delta SI")
axs[2].legend()
axs[2].grid(True)

# 4) Imbalance Price
axs[3].step(timestamps, imbalance_price, where="post",
            label="Imbalance Price", color="orange")
axs[3].set_title("Imbalance Price")
axs[3].set_xlabel("Time")
axs[3].set_ylabel("Imbalance Price")
axs[3].legend()
axs[3].grid(True)

plt.xticks(rotation=45)

# Vertical quarter-hour lines (only if <= 24h span)
if timestamps:
    span_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
    if span_seconds <= 24 * 3600:
        quarter_hour_lines = [ts for ts in timestamps if ts.minute in [0, 15, 30, 45]]
        for ax in axs:
            for qhl in quarter_hour_lines:
                ax.axvline(qhl, color="gray", linestyle="--", alpha=0.5)

try:
    fig.tight_layout()
except Exception:
    pass

st.subheader("Figure 1 — Time series: SI & Price (+ ΔSI lines)")
st.pyplot(fig, use_container_width=True)

# -----------------------------------
# Figure 2: Scatter (x=max, y=min) per quarter-hour
# -----------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 8))

if qh_max and qh_min and any(v is not None for v in qh_max) and any(v is not None for v in qh_min):
    points2 = [(mx, mn) for mx, mn in zip(qh_max, qh_min) if mx is not None and mn is not None]
    xs2 = [p[0] for p in points2]
    ys2 = [p[1] for p in points2]

    ax2.scatter(xs2, ys2, s=70, c="teal", edgecolors="white", alpha=0.85)
    ax2.set_title("Quarter-hour Instantaneous System Imbalance: Max vs Min")
    ax2.set_xlabel("Highest instantaneous system imbalance")
    ax2.set_ylabel("Lowest instantaneous system imbalance")
    ax2.grid(True, linestyle="--", alpha=0.5)

    all_vals = xs2 + ys2
    lim_min = min(all_vals)
    lim_max = max(all_vals)
    ax2.plot([lim_min, lim_max], [lim_min, lim_max], color="gray",
             linestyle=":", linewidth=1, label="y = x")
    ax2.legend()
else:
    ax2.text(0.5, 0.5, "No quarter-hour data available in the selected window",
             ha="center", va="center", transform=ax2.transAxes)
    ax2.set_axis_off()

try:
    fig2.tight_layout()
except Exception:
    pass

st.subheader("Figure 2 — Scatter: Max vs Min per Quarter-Hour")
st.pyplot(fig2, use_container_width=True)

# -----------------------------------
# Figure 3: Scatter (x=longest increase amplitude, y=longest decrease amplitude)
# -----------------------------------
fig3, ax3 = plt.subplots(figsize=(10, 8))

if qh_longest_inc_amp and qh_longest_dec_amp:
    xs3 = qh_longest_inc_amp
    ys3 = qh_longest_dec_amp
    ax3.scatter(xs3, ys3, s=70, c="#d45500", edgecolors="white", alpha=0.9)
    ax3.set_title("Quarter-hour: Longest Consecutive Increase vs Decrease (ΔSI amplitude)")
    ax3.set_xlabel("Sum of longest consecutive increase (ΔSI)")
    ax3.set_ylabel("Sum of longest consecutive decrease (ΔSI)")
    ax3.grid(True, linestyle="--", alpha=0.5)

    if any(xs3) or any(ys3):
        max_val = max(xs3 + ys3 + [0.0])
        pad = max(1e-9, 0.02 * max_val)
        ax3.set_xlim(-pad, max_val + pad)
        ax3.set_ylim(-pad, max_val + pad)
else:
    ax3.text(0.5, 0.5, "No quarter-hour sequences available in the selected window",
             ha="center", va="center", transform=ax3.transAxes)
    ax3.set_axis_off()

try:
    fig3.tight_layout()
except Exception:
    pass

st.subheader("Figure 3 — Scatter: Longest Δ↑ vs Δ↓ per Quarter-Hour")
st.pyplot(fig3, use_container_width=True)

# -----------------------------------
# Download button (filtered data)
# -----------------------------------
st.markdown("### Download filtered data")
csv_buf = io.StringIO()
df_dl = df_filtered[["Datetime", "System imbalance", "Imbalance Price"]].copy()
df_dl.to_csv(csv_buf, index=False)
fname = f"ods133_{start_dt_local.strftime('%Y%m%d%H')}_{end_dt_local.strftime('%Y%m%d%H')}_EuropeBrussels.csv"
st.download_button(
    label="Download CSV",
    data=csv_buf.getvalue(),
    file_name=fname,
    mime="text/csv",
)
