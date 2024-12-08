import streamlit as st
import pandas as pd
import plotly.express as px

# Define the file path pattern
file_path_pattern =  "/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/Kafka/output_csv/{year}.csv"

# Load and combine data for each year (2018-2024)
dfs = []
for year in range(2018, 2025):  # Adjusted range to include 2024
    file_path = file_path_pattern.format(year=year)
    try:
        df = pd.read_csv(file_path)
        df['Year'] = year  # Add year column to track the data
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

# Display the raw data
st.title("Aggregation Type Analysis (2018-2024)")
st.subheader("Raw Data")
st.write(data)

# Year Selection
years = sorted(data['Year'].unique())  # Available years (now includes 2024)
selected_years = st.multiselect("Select Year(s)", options=years, default=years)

# Check if any years are selected
if not selected_years:
    st.warning("Please select at least one year to display results.")
    st.stop()

# Filter data based on selected years
filtered_data = data[data['Year'].isin(selected_years)]

# Group by Aggregation_Type and sum the counts across years
aggregation_summary = filtered_data.groupby('Aggregation_Type').size().reset_index(name='Count')

# Display aggregated data (sum of counts across selected years)
st.subheader("Aggregated Data (Summed across years)")
st.write(aggregation_summary)

# Visualization
st.subheader("Visualization")
fig = px.bar(
    aggregation_summary,
    x="Aggregation_Type",
    y="Count",
    color="Aggregation_Type",
    title="Count of Aggregation Type Summed Across Selected Years (2018-2024)",
    labels={"Count": "Total Frequency", "Aggregation_Type": "Type"}
)

st.plotly_chart(fig)
