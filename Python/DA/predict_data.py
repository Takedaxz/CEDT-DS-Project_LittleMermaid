import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
file_path = "/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/Python/DA/VisualizeData/predict_data.csv"

# Apply Streamlit settings for wide mode and styling
st.set_page_config(page_title="Quarterly Trends Dashboard", layout="wide", page_icon="📊")

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        st.error("The file is empty or cannot be loaded.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Load the dataset
df = load_data(file_path)

if df is not None:
    # Ensure proper column names
    if {"Year", "Quarter", "Percentage"}.issubset(df.columns):
        st.title("📊 Quarterly Percentage Trends Dashboard")
        st.write("### Analyze and visualize trends in percentage data over quarters and years.")

        # Sidebar Filters
        with st.sidebar:
            st.header("Filter Options")
            year_filter = st.multiselect("Select Year(s):", options=sorted(df["Year"].unique()), default=sorted(df["Year"].unique()))
            quarter_filter = st.multiselect("Select Quarter(s):", options=df["Quarter"].unique(), default=df["Quarter"].unique())
            st.markdown("---")
            st.write("Powered by **Streamlit** 💻")

        # Filter data
        filtered_df = df[(df["Year"].isin(year_filter)) & (df["Quarter"].isin(quarter_filter))]

        # Layout with two columns for data and charts
        col1, col2 = st.columns([1, 2])

        # Display filtered data
        with col1:
            st.subheader("Filtered Data")
            st.dataframe(filtered_df, use_container_width=True)
            st.write("### Summary Statistics")
            st.dataframe(filtered_df.describe(), use_container_width=True)

        # Display trends chart
        with col2:
            st.subheader("Quarterly Percentage Trends")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=filtered_df, x="Quarter", y="Percentage", hue="Year", marker="o", ax=ax)
            ax.set_title("Quarterly Percentages by Year", fontsize=16)
            ax.set_xlabel("Quarter", fontsize=14)
            ax.set_ylabel("Percentage", fontsize=14)
            ax.legend(title="Year", fontsize=10)
            st.pyplot(fig)

        # Add an additional insights section
        st.write("### Insights and Observations")
        st.markdown(
            """
            - Use the filters in the sidebar to explore trends for specific years or quarters.
            - The chart provides a year-over-year comparison of quarterly performance.
            - Hover over points in the data table for specific values.
            """
        )
    else:
        st.error("The dataset must include `Year`, `Quarter`, and `Percentage` columns.")
else:
    st.stop()
