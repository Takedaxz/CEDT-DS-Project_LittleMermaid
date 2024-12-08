import streamlit as st
import pandas as pd

# Title
st.title("Count Aggregation Types in a Dataset")

# File path (replace with your file path)
file_path = '/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/ExtractedData/2018.csv'

try:
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Check if the column 'Aggregation_Type' exists
    if 'Aggregation_Type' in df.columns:
        # Count the occurrences of each aggregation type
        agg_counts = df['Aggregation_Type'].value_counts()

        # Display results
        st.write("### Aggregation Type Counts")
        st.write(agg_counts)
        
        # Display a pie chart
        st.write("### Aggregation Type Distribution")
        st.pyplot(agg_counts.plot.pie(autopct='%1.1f%%', figsize=(6, 6)).get_figure())
    else:
        st.error("The column 'Aggregation_Type' does not exist in the dataset.")

except Exception as e:
    st.error(f"An error occurred: {e}")
