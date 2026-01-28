import io
import csv
from datetime import datetime, timedelta, date, time
from collections import defaultdict

import streamlit as st
import matplotlib.pyplot as plt

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title="SI & Price Dashboard", layout="wide")

st.title("System Imbalance & Imbalance Price — Dashboard")
st.caption("Visualizing cumulative & instantaneous SI, ΔSI per minute, and quarter-hour stats.")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (semicolon ';' delimited)", type=["csv"], help="Expected columns: Datetime;System imbalance;Imbalance Price"
)

csv_path = st.sidebar.text_input(
    "Or provide local CSV path (optional)", value="", help="If you don't upload, the app will try to read from this path."
)

# Default window from your script
default_start_date = date(2025, 10, 23)
default_end_date = date(2025, 10, 23)
default_start_time = time(18, 0)  # 18:00
default_end_time = time(22, 0)    # 22:00

start_date = st.sidebar.date_input("Start date", value=default_start_date)
start_hour = st.sidebar.number_input("Start hour (0–23)", min_value=0, max_value=23, value=default_start_time.hour)

end_date = st.sidebar.date_input("End date", value=default_end_date)
end_hour = st.sidebar.number_input("End hour (0–23)", min_value=0, max_value=23, value=default_end_time.hour)

st.sidebar.markdown("---")
st.sidebar.write("**Notes**")
st.sidebar.write("- Datetime parsing expects ISO format (e.g., `YYYY-MM-DDTHH:MM:SS+...`).")
st.sidebar.write("- Numeric fields may use comma decimal separators; we'll normalize those.")

# -------------------------
# Helper functions
# -------------------------

def parse_csv_dicts(file_like_text):
    """Read semicolon-delimited CSV into dict rows."""
    reader = csv.DictReader(file_like_text, delimiter=';')
    return list(reader)

def load_rows():
    """Load CSV rows from the uploaded file or local path."""
    # Prefer uploaded file
    if uploaded_file is not None:
        text = uploaded_file.getvalue().decode("utf-8-sig")
        return parse_csv_dicts(io.StringIO(text))

    # Fallback: local path
    if csv_path.strip():
        try:
            with open(csv_path.strip(), 'r', encoding='utf-8-sig') as f:
                return parse_csv_dicts(f)
        except Exception as e:
            st.error(f"Failed to read local file: {e}")
            return []

    st.info("Upload a CSV file or provide a local path to proceed.")
    return []

def in_window(dt: datetime, s_date: date, s_hour: int, e_date: date, e_hour: int) -> bool:
    """
    Replicates your window logic:
    Include rows where:
      (dt.date() > start_date OR (dt.date() == start_date AND dt.hour >= start_hour))
       AND
      (dt.date() < end_date OR (dt.date() == end_date AND dt.hour <= end_hour))
    """
    return (
        (dt.date() > s_date or (dt.date() == s_date and dt.hour >= s_hour)) and
        (dt.date() < e_date or (dt.date() == e_date and dt.hour <= e_hour))
    )

def longest_run_amplitudes(vals):
    """
    Compute amplitudes (not lengths) of the longest strictly increasing and strictly
    decreasing consecutive runs within a list of values.
    Returns (max_increase_amplitude, max_decrease_amplitude), both >= 0.
    Example: [100, 160, 240, 220] -> inc_amp = 240-100 = 140; dec_amp = 240-220 = 20
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
            # continue increasing run
            pass
        else:
            # run ended at inc_prev
            amp = inc_prev - inc_start
            if amp > max_inc_amp:
                max_inc_amp = amp
            # start new run at current value
            inc_start = x
        inc_prev = x
    # finalize last run
    last_amp = inc_prev - inc_start
    if last_amp > max_inc_amp:
        max_inc_amp = last_amp

    # Longest strictly decreasing run amplitude
    dec_start = vals[0]
    dec_prev = vals[0]
    max_dec_amp = 0.0
    for x in vals[1:]:
        if x < dec_prev:
            # continue decreasing run
            pass
        else:
            # run ended at dec_prev
            amp = dec_start - dec_prev
            if amp > max_dec_amp:
                max_dec_amp = amp
            # start new run at current value
            dec_start = x
        dec_prev = x
    # finalize last run
    last_amp = dec_start - dec_prev
    if last_amp > max_dec_amp:
        max_dec_amp = last_amp

    # Ensure non-negative
    if max_inc_amp < 0:
        max_inc_amp = 0.0
    if max_dec_amp < 0:
        max_dec_amp = 0.0

    return float(max_inc_amp), float(max_dec_amp)

# -------------------------
# Data loading & processing
# -------------------------

rows = load_rows()

if rows:
    timestamps = []
    cumulative_si = []
    imbalance_price = []

    for row in rows:
        try:
            dt_str = row['Datetime']
        except KeyError:
            st.error("CSV must contain a 'Datetime' column.")
            st.stop()

        try:
            # Strip timezone if present, use naive local time
            dt = datetime.fromisoformat(dt_str.split('+')[0])
        except Exception:
            # Try alternative: replace comma or space (robustness)
            try:
                cleaned = dt_str.replace(' ', 'T')
                dt = datetime.fromisoformat(cleaned.split('+')[0])
            except Exception:
                # Skip rows with unparseable datetime
                continue

        if in_window(dt, start_date, start_hour, end_date, end_hour):
            try:
                # Normalize comma decimal separators
                si_value = float(row['System imbalance'].replace(',', '.'))
                price_value = float(row['Imbalance Price'].replace(',', '.'))
            except (ValueError, KeyError):
                # Skip bad/missing values
                continue

            timestamps.append(dt)
            cumulative_si.append(si_value)
            imbalance_price.append(price_value)

    if len(timestamps) == 0:
        st.warning("No data in the selected window. Adjust your date/hour filters.")
        st.stop()

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

    # Compute amplitudes of longest strictly increasing/decreasing runs per quarter-hour
    qh_longest_inc_amp = []
    qh_longest_dec_amp = []
    for vals in qh_ordered_vals:
        inc_amp, dec_amp = longest_run_amplitudes(vals)
        qh_longest_inc_amp.append(inc_amp)
        qh_longest_dec_amp.append(dec_amp)

    # Max/Min per quarter-hour
    qh_max = [max(vals) if len(vals) else None for vals in qh_ordered_vals]
    qh_min = [min(vals) if len(vals) else None for vals in qh_ordered_vals]

    # -------------------------
    # Figure 1: 4-panel time series
    # -------------------------
    fig, axs = plt.subplots(4, 1, figsize=(18, 18), sharex=True)

    # 1) Cumulative + Instantaneous SI
    axs[0].step(timestamps, cumulative_si, where='post', label='Cumulative SI (csv values)')
    axs[0].step(timestamps, instantaneous_si, where='post', label='Instantaneous SI', alpha=0.7)
    axs[0].set_title('Cumulative SI and Instantaneous SI')
    axs[0].set_ylabel('System Imbalance')
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

            # Green line for longest consecutive increase amplitude across the QH
            ax_lines.plot(
                [start_time, end_time], [inc_amp, inc_amp],
                color='green', linewidth=3,
                label='Δ longest consecutive increase' if not first_label_done else None
            )

            # Red line for longest consecutive decrease amplitude across the QH
            ax_lines.plot(
                [start_time, end_time], [dec_amp, dec_amp],
                color='red', linewidth=3,
                label='Δ longest consecutive decrease' if not first_label_done else None
            )

            first_label_done = True

        ax_lines.set_title('Per quarter-hour: amplitudes of longest consecutive ↑ and ↓ (ΔSI) — lines')
        ax_lines.set_ylabel('ΔSI amplitude')
        ax_lines.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax_lines.legend()
        ax_lines.xaxis_date()  # ensure date formatting on x-axis
    else:
        ax_lines.text(
            0.5, 0.5, 'No quarter-hour data available in the selected window',
            ha='center', va='center', transform=ax_lines.transAxes
        )
        ax_lines.set_axis_off()

    # 3) Minute-by-minute change of instantaneous SI
    axs[2].step(timestamps[1:], delta_si, where='post',
                label='Minute-by-Minute Change of Instantaneous SI', color='purple')
    axs[2].set_title('Minute-by-Minute Change of Instantaneous SI')
    axs[2].set_ylabel('Delta SI')
    axs[2].legend()
    axs[2].grid(True)

    # 4) Imbalance Price
    axs[3].step(timestamps, imbalance_price, where='post',
                label='Imbalance Price', color='orange')
    axs[3].set_title('Imbalance Price')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('Imbalance Price')
    axs[3].legend()
    axs[3].grid(True)

    # Rotate x-ticks
    plt.xticks(rotation=45)

    # Only add vertical quarter-hour lines if plotted period is 24 hours or less
    if timestamps:
        span_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
        if span_seconds <= 24 * 3600:
            quarter_hour_lines = [ts for ts in timestamps if ts.minute in [0, 15, 30, 45]]
            for ax in axs:
                for qhl in quarter_hour_lines:
                    ax.axvline(qhl, color='gray', linestyle='--', alpha=0.5)

    try:
        fig.tight_layout()
    except Exception:
        pass

    st.subheader("Figure 1 — Time series: SI & Price (+ ΔSI lines)")
    st.pyplot(fig, use_container_width=True)

    # -------------------------
    # Figure 2: Scatter (x=max, y=min) per quarter-hour
    # -------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 8))

    if qh_max and qh_min and any(v is not None for v in qh_max) and any(v is not None for v in qh_min):
        points2 = [(mx, mn) for mx, mn in zip(qh_max, qh_min) if mx is not None and mn is not None]
        xs2 = [p[0] for p in points2]
        ys2 = [p[1] for p in points2]

        sc = ax2.scatter(xs2, ys2, s=70, c='teal', edgecolors='white', alpha=0.85)
        ax2.set_title('Quarter-hour Instantaneous System Imbalance: Max vs Min')
        ax2.set_xlabel('Highest instantaneous system imbalance')
        ax2.set_ylabel('Lowest instantaneous system imbalance')
        ax2.grid(True, linestyle='--', alpha=0.5)

        all_vals = xs2 + ys2
        lim_min = min(all_vals)
        lim_max = max(all_vals)
        ax2.plot([lim_min, lim_max], [lim_min, lim_max], color='gray',
                 linestyle=':', linewidth=1, label='y = x')
        ax2.legend()
    else:
        ax2.text(
            0.5, 0.5, 'No quarter-hour data available in the selected window',
            ha='center', va='center', transform=ax2.transAxes
        )
        ax2.set_axis_off()

    try:
        fig2.tight_layout()
    except Exception:
        pass

    st.subheader("Figure 2 — Scatter: Max vs Min per Quarter-Hour")
    st.pyplot(fig2, use_container_width=True)

    # -------------------------
    # Figure 3: Scatter (x=longest increase amplitude, y=longest decrease amplitude)
    # -------------------------
    fig3, ax3 = plt.subplots(figsize=(10, 8))

    if qh_longest_inc_amp and qh_longest_dec_amp:
        xs3 = qh_longest_inc_amp
        ys3 = qh_longest_dec_amp
        sc3 = ax3.scatter(xs3, ys3, s=70, c='#d45500', edgecolors='white', alpha=0.9)
        ax3.set_title('Quarter-hour: Longest Consecutive Increase vs Decrease (ΔSI amplitude)')
        ax3.set_xlabel('Sum of longest consecutive increase (ΔSI)')
        ax3.set_ylabel('Sum of longest consecutive decrease (ΔSI)')
        ax3.grid(True, linestyle='--', alpha=0.5)

        if any(xs3) or any(ys3):
            max_val = max(xs3 + ys3 + [0.0])
            pad = max(1e-9, 0.02 * max_val)
            ax3.set_xlim(-pad, max_val + pad)
            ax3.set_ylim(-pad, max_val + pad)
    else:
        ax3.text(
            0.5, 0.5, 'No quarter-hour sequences available in the selected window',
            ha='center', va='center', transform=ax3.transAxes
        )
        ax3.set_axis_off()

    try:
        fig3.tight_layout()
    except Exception:
        pass

    st.subheader("Figure 3 — Scatter: Longest Δ↑ vs Δ↓ per Quarter-Hour")
    st.pyplot(fig3, use_container_width=True)

else:
    # No rows loaded yet; instructions already shown.
    pass
