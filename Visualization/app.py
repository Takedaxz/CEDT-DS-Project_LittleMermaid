import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Set Streamlit page configuration
st.set_page_config(
    page_title="Research Publications & Keywords Analysis",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📚 Research Publications & Keywords Analysis")
st.markdown("""
Explore **Research Publications** and **Top Keywords** in one unified dashboard. 
Choose intervals, years, and chart types to visualize trends effectively.
""")

# List of local CSV files for publications and keywords
pub_file_names = [
    "/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/ExtractedData/2018.csv",
    "/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/ExtractedData/2019.csv",
    "/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/ExtractedData/2020.csv",
    "/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/ExtractedData/2021.csv",
    "/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/ExtractedData/2022.csv",
    "/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/ExtractedData/2023.csv"
]

keyword_file_names = [
    "/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/ExtractedData/2018_keywords_counts.csv",
    "/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/ExtractedData/2019_keywords_counts.csv",
    "/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/ExtractedData/2020_keywords_counts.csv",
    "/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/ExtractedData/2021_keywords_counts.csv",
    "/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/ExtractedData/2022_keywords_counts.csv",
    "/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/ExtractedData/2023_keywords_counts.csv"
]


# Load publications data
pub_dataframes = []
for file_name in pub_file_names:
    if not os.path.exists(file_name):
        st.error(f"❌ File not found: {file_name}")
        st.stop()
    df = pd.read_csv(file_name)
    df['Publication_Date'] = pd.to_datetime(df['Publication_Date'], errors='coerce')
    pub_dataframes.append(df)
combined_pub_data = pd.concat(pub_dataframes)

# Load keywords data
keyword_dataframes = []
for file_name in keyword_file_names:
    if not os.path.exists(file_name):
        st.error(f"❌ File not found: {file_name}")
        st.stop()
    year = os.path.basename(file_name).split('_')[0]
    df = pd.read_csv(file_name)
    df = df.rename(columns={f"{year}_Count": "Count"})
    df['Year'] = year
    keyword_dataframes.append(df[['Keywords', 'Count', 'Year']])
combined_keyword_data = pd.concat(keyword_dataframes)

# Create tabs for "Publications" and "Keywords"
tab1, tab2 = st.tabs(["📅 Publications Analysis", "🏷️ Keywords Analysis"])

if tab1:  # Publications Analysis
    with st.sidebar:
        st.subheader("📅 Publications Filters")
        interval = st.radio("Select Interval", ["Yearly", "Quarterly", "Monthly"])
        chart_type = st.radio("Select Chart Type", ["Bar Chart", "Line Chart", "Pie Chart"])

    st.subheader("Publications Analysis")
    if interval == "Yearly":
        combined_pub_data['Year'] = combined_pub_data['Publication_Date'].dt.year
        counts = combined_pub_data['Year'].value_counts().sort_index()
    elif interval == "Quarterly":
        combined_pub_data['Quarter'] = combined_pub_data['Publication_Date'].dt.to_period('Q')
        counts = combined_pub_data['Quarter'].value_counts().sort_index()
    elif interval == "Monthly":
        combined_pub_data['Month_Year'] = combined_pub_data['Publication_Date'].dt.to_period('M')
        counts = combined_pub_data['Month_Year'].value_counts().sort_index()

    if chart_type == "Bar Chart":
        fig = px.bar(counts, x=counts.index.astype(str), y=counts, labels={'x': interval, 'y': 'Number of Publications'}, title=f"Publications by {interval}")
    elif chart_type == "Line Chart":
        fig = px.line(counts, x=counts.index.astype(str), y=counts, labels={'x': interval, 'y': 'Number of Publications'}, title=f"Publications by {interval}")
    elif chart_type == "Pie Chart":
        fig = px.pie(counts, names=counts.index.astype(str), values=counts, title=f"Publications by {interval}")
    st.plotly_chart(fig, use_container_width=True)

if tab2:  # Keywords Analysis
    with st.sidebar:
        st.subheader("🏷️ Keywords Filters")
        selected_years = st.multiselect("Select Years", options=combined_keyword_data['Year'].unique(), default=combined_keyword_data['Year'].unique())
        chart_type = st.radio("Select Chart Type", ["Bar Chart", "Pie Chart"], key="keyword_chart")

    st.subheader("Keywords Analysis")
    if selected_years:
        filtered_keywords = combined_keyword_data[combined_keyword_data['Year'].isin(selected_years)]
        top_keywords = filtered_keywords.groupby(['Year', 'Keywords']).sum().reset_index()
        top_keywords = top_keywords.sort_values(['Year', 'Count'], ascending=[True, False]).groupby('Year').head(5)

        if chart_type == "Bar Chart":
            fig = px.bar(top_keywords, x='Keywords', y='Count', color='Year', barmode='group', title="Top Keywords by Year")
        elif chart_type == "Pie Chart" and len(selected_years) == 1:
            year = selected_years[0]
            year_data = filtered_keywords[filtered_keywords['Year'] == year].nlargest(5, 'Count')
            fig = px.pie(year_data, names='Keywords', values='Count', title=f"Top Keywords in {year}")
        else:
            st.warning("Please select only one year for Pie Chart.")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least one year.")
