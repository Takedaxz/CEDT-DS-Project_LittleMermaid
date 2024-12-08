import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import networkx as nx


# Base directory setup relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Current directory of the script
PROJECT_DIR = os.path.join(BASE_DIR, "..", "..")  # Adjust path to two levels above
KAFKA_DIR = os.path.join(PROJECT_DIR, "Kafka", "output_csv")
KEYWORD_DIR = os.path.join(BASE_DIR, "VisualizeData")

# File paths for publications and keywords
pub_file_names = [
    os.path.join(KAFKA_DIR, f"{year}.csv") for year in range(2018, 2024)
]
keyword_file_names = [
    os.path.join(KEYWORD_DIR, f"{year}_keywords_counts.csv") for year in range(2018, 2024)
]

# Validate directories and file existence
def validate_files(file_paths):
    missing_files = [file for file in file_paths if not os.path.exists(file)]
    if missing_files:
        st.error(f"Missing files: {', '.join(missing_files)}")
        st.stop()

# Validate files for publications and keywords
validate_files(pub_file_names)
validate_files(keyword_file_names)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Research Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📚 Research Analysis Dashboard")
st.markdown("""
Explore **Co-authorship Networks**, **Research Publications**, and **Top Keywords** in one unified dashboard. 
Choose intervals, years, and chart types to visualize trends effectively.
""")

# Function to process co-authorship data
def process_data(file_path, top_nodes):
    df = pd.read_csv(file_path)
    df = df.head(30)

    coauthorships = defaultdict(int)
    for authors in df['Author']:
        author_list = [author.strip() for author in authors.split(';')]
        if len(author_list) >= 2:
            for i in range(len(author_list)):
                for j in range(i + 1, len(author_list)):
                    pair = tuple(sorted([author_list[i], author_list[j]]))
                    coauthorships[pair] += 1

    G = nx.Graph()
    for pair, count in coauthorships.items():
        G.add_edge(pair[0], pair[1], weight=count)

    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_nodes]
    filtered_nodes = [node[0] for node in sorted_nodes]
    return G.subgraph(filtered_nodes)

# Load publications data
pub_dataframes = []
for file_name in pub_file_names:
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        df['Publication_Date'] = pd.to_datetime(df['Publication_Date'], errors='coerce')
        pub_dataframes.append(df)
combined_pub_data = pd.concat(pub_dataframes)

# Load keywords data
keyword_dataframes = []
for file_name in keyword_file_names:
    if os.path.exists(file_name):
        year = os.path.basename(file_name).split('_')[0]
        df = pd.read_csv(file_name)
        df = df.rename(columns={f"{year}_Count": "Count"})
        df['Year'] = year
        keyword_dataframes.append(df[['Keywords', 'Count', 'Year']])
combined_keyword_data = pd.concat(keyword_dataframes)

# Multiselect for analysis
selected_analyses = st.multiselect(
    "Select Analyses to Perform",
    ["📅 Publications Analysis", "🏷️ Keywords Analysis", "🔗 Co-authorship Network", "🏫 Institution Analysis"],
    default=["📅 Publications Analysis"]
)

if "📅 Publications Analysis" in selected_analyses:
    with st.sidebar:
        st.subheader("📅 Publications Filters")
        interval = st.radio("Select Interval", ["Yearly", "Quarterly", "Monthly"], key="pub_interval")
        chart_type = st.radio("Select Chart Type", ["Bar Chart", "Line Chart", "Pie Chart"], key="pub_chart")

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

    # Summary for Publications
    st.subheader("📊 Summary")
    st.write(f"- **Total Publications Analyzed**: {counts.sum()}")
    st.write(f"- **Highest Interval**: {counts.idxmax()} ({counts.max()} publications)")

if "🏷️ Keywords Analysis" in selected_analyses:
    with st.sidebar:
        st.subheader("🏷️ Keywords Filters")
        selected_years = st.multiselect(
            "Select Years", 
            options=combined_keyword_data['Year'].unique(), 
            default=combined_keyword_data['Year'].unique()
        )
        chart_type = st.radio("Select Chart Type", ["Bar Chart", "Pie Chart"], key="keyword_chart")

    st.subheader("Keywords Analysis")
    if selected_years:
        filtered_keywords = combined_keyword_data[combined_keyword_data['Year'].isin(selected_years)]
        aggregated_keywords = filtered_keywords.groupby('Keywords')['Count'].sum().reset_index()
        aggregated_keywords = aggregated_keywords.sort_values('Count', ascending=False).head(10)

        if chart_type == "Bar Chart":
            fig = px.bar(
                aggregated_keywords, 
                x='Keywords', 
                y='Count', 
                title="Top Keywords",
                labels={'Keywords': 'Keywords', 'Count': 'Total Count'}
            )
        elif chart_type == "Pie Chart":
            fig = px.pie(
                aggregated_keywords, 
                names='Keywords', 
                values='Count', 
                title="Top Keywords"
            )

        st.plotly_chart(fig, use_container_width=True)

        # Summary for Keywords
        st.subheader("📊 Summary")
        st.write(f"- **Total Keywords Analyzed**: {len(filtered_keywords['Keywords'].unique())}")
        st.write(f"- **Top Keyword**: {aggregated_keywords.iloc[0]['Keywords']} ({aggregated_keywords.iloc[0]['Count']} mentions)")
    else:
        st.warning("Please select at least one year.")

if "🔗 Co-authorship Network" in selected_analyses:
    with st.sidebar:
        st.subheader("🔗 Co-authorship Filters")
        year = st.selectbox("Select Year", [str(y) for y in range(2018, 2024)])
        top_nodes = st.slider("Number of Top Authors to Display (Nodes):", min_value=5, max_value=30, value=10)

    st.subheader("Co-authorship Network")
    file_path = os.path.join(KAFKA_DIR, f"{year}.csv")
    if st.button("Generate Network"):
        try:
            H = process_data(file_path, top_nodes)
            pos = nx.spring_layout(H, k=1, iterations=100)

            edge_x, edge_y = [], []
            for edge in H.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            node_x, node_y, node_labels = [], [], []
            for node in H.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_labels.append(node)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(color='gray', width=0.5),
                hoverinfo='none'
            ))

            fig.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                marker=dict(size=10, color='lightblue', line_width=1),
                text=node_labels,
                textposition="top center",
                hoverinfo='text'
            ))

            fig.update_layout(
                title=f"Co-authorship Network for {year} (Top {top_nodes} Authors)",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
                height=800,
                width=800,
            )

            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error: {e}")
if "🏫 Institution Analysis" in selected_analyses:
    with st.sidebar:
        st.subheader("🏫 Institution Filters")
        selected_years = st.multiselect(
            "Select Years",
            options=[str(year) for year in range(2018, 2024)],
            default=[str(year) for year in range(2018, 2024)],
            key="institution_years"
        )
        max_institutions = st.slider(
            "Select number of institutions to display:",
            min_value=1,
            max_value=500,
            value=10
        )
        sort_order = st.radio(
            "Sort Order",
            options=["Descending (High to Low)", "Ascending (Low to High)"],
            index=0,
            key="institution_sort_order"
        )

    st.subheader("Institution Analysis")
    all_data = []

    for year in selected_years:
        file_path = os.path.join(KAFKA_DIR, f"{year}.csv")
        if not os.path.exists(file_path):
            st.warning(f"File not found for {year}! Skipping...")
            continue
        df = pd.read_csv(file_path)
        all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        institution_column = 'Institutions'

        if institution_column in combined_df.columns:
            df_cleaned = combined_df[institution_column].dropna()
            institutions_split = df_cleaned.str.split(';')
            all_institutions = [institution.strip() for sublist in institutions_split for institution in set(sublist)]
            publication_counts = pd.Series(all_institutions).value_counts()

            # Adjust sorting order
            if sort_order == "Descending (High to Low)":
                publication_counts = publication_counts.sort_values(ascending=False)
            else:
                publication_counts = publication_counts.sort_values(ascending=True)

            top_publications = publication_counts.head(max_institutions)

            fig = go.Figure(data=[go.Bar(
                x=top_publications.index,
                y=top_publications.values,
                text=top_publications.values,
                textposition='auto',
                marker_color='skyblue'
            )])

            fig.update_layout(
                title=f"Top {max_institutions} Publication Counts per Institution ({', '.join(selected_years)})",
                xaxis_title="Institution",
                yaxis_title="Publication Count",
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Summary for Institutions
            st.subheader("📊 Summary")
            top_institution = top_publications.idxmax() if sort_order == "Descending (High to Low)" else top_publications.idxmin()
            top_count = top_publications.max() if sort_order == "Descending (High to Low)" else top_publications.min()
            st.write(f"- **Top Institution**: {top_institution} ({top_count} publications)")
            st.write(f"- **Total Institutions Analyzed**: {len(publication_counts)}")
        else:
            st.error(f"Column '{institution_column}' not found in the combined dataset.")
    else:
        st.error("No valid data files found for the selected years.")
