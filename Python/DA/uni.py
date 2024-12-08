import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Load CSV file
file_path = '/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/ExtractedData/2018.csv'


# Check if the file exists
if not os.path.exists(file_path):
    st.error("File not found! Please check the file path.")
else:
    # Read the CSV data into a DataFrame
    df = pd.read_csv(file_path)

    # Specify the institution column (adjust the name if necessary)
    institution_column = 'Institutions'  # Make sure this matches the exact column name

    # Check if the 'Institutions' column exists in the dataset
    if institution_column in df.columns:
        # Remove null values in the 'Institutions' column
        df_cleaned = df[institution_column].dropna()

        # Split the institutions in each row if there are multiple institutions (separated by ';')
        institutions_split = df_cleaned.str.split(';')

        # Flatten the list of institutions and remove duplicates within each row
        all_institutions = [institution.strip() for sublist in institutions_split for institution in set(sublist)]

        # Count how many times each institution appears
        publication_counts = pd.Series(all_institutions).value_counts()

        # User can select how many institutions to display
        max_institutions = st.slider(
            "Select number of institutions to display:",
            min_value=1,
            max_value=500,
            value=10  # Default value
        )

        # Filter the top institutions
        top_publications = publication_counts.head(max_institutions)

        # Display the results in a table
        st.write(f"Top {max_institutions} Publication Counts per Institution:")
        st.write(top_publications)

        # Create a bar chart using Plotly
        fig = go.Figure(data=[go.Bar(
            x=top_publications.index,
            y=top_publications.values,
            text=top_publications.values,
            textposition='auto',
            marker_color='skyblue'
        )])

        # Add title and labels to the chart
        fig.update_layout(
            title=f"Top {max_institutions} Publication Counts per Institution",
            xaxis_title="Institution",
            yaxis_title="Publication Count",
            template="plotly_white"
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig)
    else:
        st.error(f"Column '{institution_column}' not found in the dataset.")