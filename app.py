import streamlit as st
import pandas as pd
from datetime import datetime, time
import matplotlib.pyplot as plt

st.set_page_config(page_title="SI Analysis Dashboard", layout="wide")

st.title("System Imbalance Analysis")

st.markdown(
    """
    This dashboard allows you to analyze the Belgian System Imbalance (SI) per minute, the rate of change of SI
    and the Imbalance Prices per minute over a selected time window.

    Simply change the dates and hours to get the time window requested.
    Data coverage: **1 January 2025 to 14 January 2026**.
    """
)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Select Time Window")

# Date Inputs
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime(2025, 7, 14))
    end_date = st.date_input("End Date", datetime(2025, 7, 14))

# Hour Inputs
with col2:
    start_hour = st.number_input("Start Hour", min_value=0, max_value=23, value=16)
    end_hour = st.number_input("End Hour", min_value=0, max_value=23, value=20)

# Combine Date and Hour for filtering
start_dt = datetime.combine(start_date, time(start_hour, 0))
end_dt = datetime.combine(end_date, time(end_hour, 59))

# --- FIXED CSV FILES ---
CSV_FILES = [
    "IP1-1.csv",  # 1 Jan 2025 → 30 May 2025
    "IP1-2.csv",  # 1 Jun 2025 → 30 Sep 2025
    "IP1-3.csv",  # 1 Oct 2025 → 14 Jan 2026
]

try:
    @st.cache_data
    def load_data(file_paths):
        all_data = []

        for file_path in file_paths:
            df_raw = pd.read_csv(file_path, header=None, encoding='utf-8-sig')

            # Split header
            header = df_raw.iloc[0, 0].split(';')

            # Split data rows
            data = df_raw.iloc[1:, 0].str.split(';', expand=True)
            data.columns = header
            data = data.reset_index(drop=True)

            # Clean numeric columns
            data['System imbalance'] = pd.to_numeric(
                data['System imbalance'].str.replace(',', '.', regex=False),
                errors='coerce'
            )

            data['Imbalance Price'] = pd.to_numeric(
                data['Imbalance Price'].str.replace(',', '.', regex=False),
                errors='coerce'
            )

            # Parse datetime (remove timezone)
            data['Datetime'] = pd.to_datetime(
                data['Datetime'].str.split('+').str[0],
                errors='coerce'
            )

            # Drop invalid rows
            data = data.dropna(subset=['Datetime', 'System imbalance', 'Imbalance Price'])

            all_data.append(data)

        return pd.concat(all_data, ignore_index=True)

    # Load and merge all CSVs
    data = load_data(CSV_FILES)

    # Filter data based on sidebar inputs
    mask = (data['Datetime'] >= start_dt) & (data['Datetime'] <= end_dt)
    df_filtered = data.loc[mask].sort_values('Datetime').reset_index(drop=True)

    if df_filtered.empty:
        st.warning("No data found for the selected time range.")
    else:
        # --- CALCULATIONS ---
        timestamps = df_filtered['Datetime'].tolist()
        cumulative_si = df_filtered['System imbalance'].tolist()
        imbalance_price = df_filtered['Imbalance Price'].tolist()

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

        delta_si = [
            instantaneous_si[i] - instantaneous_si[i - 1]
            for i in range(1, len(instantaneous_si))
        ]

        # --- PLOTTING ---
        fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

        axs[0].step(timestamps, cumulative_si, where='post', label='Cumulative SI')
        axs[0].step(timestamps, instantaneous_si, where='post', label='Instantaneous SI', alpha=0.7)
        axs[0].set_title('Cumulative SI and Instantaneous SI')
        axs[0].set_ylabel('System Imbalance')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].step(
            timestamps[1:], delta_si, where='post',
            label='Minute-by-Minute Change', color='purple'
        )
        axs[1].set_title('Minute-by-Minute Change of Instantaneous SI')
        axs[1].set_ylabel('Delta SI')
        axs[1].legend()
        axs[1].grid(True)

        axs[2].step(
            timestamps, imbalance_price, where='post',
            label='Imbalance Price', color='orange'
        )
        axs[2].set_title('Imbalance Price')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Price')
        axs[2].legend()
        axs[2].grid(True)

        # Vertical lines logic (24h check)
        span_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
        if span_seconds <= 24 * 3600:
            quarter_hour_lines = [
                ts for ts in timestamps if ts.minute in [0, 15, 30, 45]
            ]
            for ax in axs:
                for qhl in quarter_hour_lines:
                    ax.axvline(qhl, color='gray', linestyle='--', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        if st.checkbox("Show Raw Data Table"):
            st.write(df_filtered)

except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
except Exception as e:

    st.error(f"An error occurred: {e}")

if 'logged_in' in st.session_state.keys():
    if st.session_state['logged_in']:
        st.markdown('## Ask Me Anything')
        question = st.text_input('Ask your question')
        if question != '':
            st.write('I drink and I know things.')
