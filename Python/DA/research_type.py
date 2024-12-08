import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Base directory setup relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Current directory of the script
PROJECT_DIR = os.path.join(BASE_DIR, "..", "..")  # Adjust path to two levels above
KAFKA_DIR = os.path.join(PROJECT_DIR, "Kafka", "output_csv")

# Load data for multiple years (2018-2024)
dfs = []
for year in range(2018, 2025):
    file_path = os.path.join(KAFKA_DIR, f"{year}.csv")
    try:
        df = pd.read_csv(file_path)
        df['Year'] = year  # Add a Year column to track the data
        dfs.append(df)
    except FileNotFoundError:
        st.warning(f"File for year {year} not found.")
    except Exception as e:
        st.error(f"Error loading data for year {year}: {e}")

# Combine all dataframes into a single dataframe
if not dfs:
    st.error("No data found for the selected years.")
    st.stop()

data = pd.concat(dfs, ignore_index=True)

# Clean column names
data.columns = data.columns.str.strip()

# Streamlit App Title
st.title("Aggregation Type Analysis (2018-2024)")

# Year Selection
available_years = sorted(data['Year'].unique())
selected_years = st.multiselect(
    "Select Year(s) for Analysis", options=available_years, default=available_years
)

# Check if any years are selected
if not selected_years:
    st.warning("Please select at least one year to display results.")
    st.stop()

# Filter data based on selected years
filtered_data = data[data['Year'].isin(selected_years)]

# Group by Aggregation_Type and count occurrences
aggregation_summary = filtered_data.groupby('Aggregation_Type').size().reset_index(name='Count')

# Sorting Option
st.subheader("Sorting Options")
sort_order = st.radio(
    "Sort Data By Count",
    options=["Descending (High to Low)", "Ascending (Low to High)"],
    index=0
)

# Apply sorting based on user's selection
ascending = True if sort_order == "Ascending (Low to High)" else False
aggregation_summary = aggregation_summary.sort_values(by="Count", ascending=ascending)


# Visualization
st.subheader("Visualization")
fig = px.bar(
    aggregation_summary,
    x="Aggregation_Type",
    y="Count",
    color="Aggregation_Type",
    title=f"Count of Aggregation Type for Selected Years: {', '.join(map(str, selected_years))}",
    labels={"Count": "Total Frequency", "Aggregation_Type": "Type"}
)

st.plotly_chart(fig)