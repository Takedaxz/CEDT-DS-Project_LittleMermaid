import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os

# Ensure necessary NLTK downloads
nltk.download('stopwords')
nltk.download('punkt_tab')

# Preprocessing setup
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocesses the given text:
    - Tokenizes
    - Converts to lowercase
    - Removes stopwords
    - Keeps only alphanumeric tokens
    """
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load the data
@st.cache_data
def load_data(year):
    data_path = f'/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/Kafka/output_csv/{year}.csv'
    df = pd.read_csv(data_path)
    df['Abstract'] = df['Abstract'].fillna('')
    df['Processed_Abstract'] = df['Abstract'].apply(preprocess_text)
    return df

# Initialize TF-IDF Vectorizer
@st.cache_resource
def vectorize_data(df):
    df['Abstract'] = df['Abstract'].fillna('')
    df['Title'] = df['Title'].fillna('')
    vectorizer = TfidfVectorizer()
    combined_vectors = vectorizer.fit_transform(df['Abstract'] + " " + df['Title'])
    return vectorizer, combined_vectors


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
    ["📅 Publications Analysis", "🏷️ Keywords Analysis", "🔗 Co-authorship Network", "🏫 Institution Analysis", "📑 Research Type Analysis", "📚 Top 10 Recommended Titles Finder"],
    default=["📅 Publications Analysis"]
)

if "📅 Publications Analysis" in selected_analyses:
    with st.sidebar:
        st.subheader("📅 Publications Filters")
        interval = st.radio("Select Interval", ["Yearly", "Quarterly", "Monthly"], key="pub_interval")
        chart_type = st.radio("Select Chart Type", ["Bar Chart", "Line Chart", "Pie Chart"], key="pub_chart")

    st.subheader("Publications Analysis")
    
    if 'Publication_Date' not in combined_pub_data.columns:
        st.error("'Publication_Date' column is missing in the data!")
    else:
        combined_pub_data['Publication_Date'] = pd.to_datetime(combined_pub_data['Publication_Date'], errors='coerce')
        
        # Check for invalid dates and drop them
        invalid_dates = combined_pub_data['Publication_Date'].isnull().sum()
        if invalid_dates > 0:
            st.warning(f"Found {invalid_dates} invalid dates, they will be excluded.")
        
        if interval == "Yearly":
            combined_pub_data['Year'] = combined_pub_data['Publication_Date'].dt.year
            counts = combined_pub_data['Year'].value_counts().sort_index()
        elif interval == "Quarterly":
            combined_pub_data['Quarter'] = combined_pub_data['Publication_Date'].dt.to_period('Q')
            counts = combined_pub_data['Quarter'].value_counts().sort_index()
        elif interval == "Monthly":
            combined_pub_data['Month_Year'] = combined_pub_data['Publication_Date'].dt.to_period('M')
            counts = combined_pub_data['Month_Year'].value_counts().sort_index()

        if counts.empty:
            st.warning("No data available for this selection.")
        else:
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
            pos = nx.spring_layout(H, seed=42)
            edge_trace = go.Scatter(
                x=[],
                y=[],
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            for edge in H.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += (x0, x1, None)
                edge_trace['y'] += (y0, y1, None)

            node_trace = go.Scatter(
                x=[],
                y=[],
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    size=10,
                    colorbar=dict(
                        thickness=15,
                        title='Node Connections',
                        xanchor='left',
                        titleside='right'
                    )
                )
            )
            for node in H.nodes():
                x, y = pos[node]
                node_trace['x'] += (x,)
                node_trace['y'] += (y,)

            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                showlegend=False,
                                hovermode='closest',
                                xaxis=dict(showgrid=False, zeroline=False),
                                yaxis=dict(showgrid=False, zeroline=False),
                                title=f"Co-authorship Network for {year}",
                                title_x=0.5
                            ))

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred while generating the network: {e}")
# Ensure Publication_Date is parsed as datetime
if 'Publication_Date' not in combined_pub_data.columns:
    st.error("'Publication_Date' column is missing in the data!")
else:
    combined_pub_data['Publication_Date'] = pd.to_datetime(combined_pub_data['Publication_Date'], errors='coerce')

    # Check for invalid dates and drop them
    invalid_dates = combined_pub_data['Publication_Date'].isnull().sum()
    if invalid_dates > 0:
        st.warning(f"Found {invalid_dates} invalid dates, they will be excluded.")

    # Ensure 'Year' column is generated
    combined_pub_data['Year'] = combined_pub_data['Publication_Date'].dt.year

# Now you can use the 'Year' column in the multiselect
if "📑 Research Type Analysis" in selected_analyses:
    with st.sidebar:
        st.subheader("📑 Research Type Filters")
        selected_years = st.multiselect(
            "Select Years", 
            options=combined_pub_data['Year'].unique(), 
            default=combined_pub_data['Year'].unique()
        )
        chart_type = st.radio("Select Chart Type", ["Bar Chart", "Pie Chart"], key="research_type_chart")

    st.subheader("Research Type Analysis")
    if selected_years:
        filtered_data = combined_pub_data[combined_pub_data['Year'].isin(selected_years)]
        if 'Aggregation_Type' in filtered_data.columns:
            aggregated_data = filtered_data.groupby('Aggregation_Type').size().reset_index(name='Count')
            aggregated_data = aggregated_data.sort_values('Count', ascending=False)

            if chart_type == "Bar Chart":
                fig = px.bar(
                    aggregated_data, 
                    x='Aggregation_Type', 
                    y='Count', 
                    title="Research Type Analysis",
                    labels={'Aggregation_Type': 'Research Type', 'Count': 'Total Count'}
                )
            elif chart_type == "Pie Chart":
                fig = px.pie(
                    aggregated_data, 
                    names='Aggregation_Type', 
                    values='Count', 
                    title="Research Type Analysis"
                )

            st.plotly_chart(fig, use_container_width=True)

            # Summary for Research Types
            st.subheader("📊 Summary")
            st.write(f"- **Total Research Types Analyzed**: {len(aggregated_data)}")
            st.write(f"- **Most Common Research Type**: {aggregated_data.iloc[0]['Aggregation_Type']} ({aggregated_data.iloc[0]['Count']} occurrences)")
        else:
            st.error("The 'Aggregation_Type' column is missing in the data.")
    else:
        st.warning("Please select at least one year.")
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
        
# Top 10 Recommended Titles Finder
if "📚 Top 10 Recommended Titles Finder" in selected_analyses:
    st.title("Top 10 Recommended Titles Finder")

    # Load year selection for publications data
    year = st.sidebar.selectbox("Select Year", ['2018', '2019', '2020','2021','2022','2023'], index=0)
    df = load_data(year)
    st.write(f"Data Loaded for the Year: {year}")

    vectorizer, combined_vectors = vectorize_data(df)

    # User input for finding recommendations
    input_data = st.text_area("Enter Input Text:", placeholder="Type or paste your input here...")
    if st.button("Find Recommendations"):
        if input_data.strip():
            # Preprocess and vectorize user input
            processed_input = preprocess_text(input_data)
            input_vector = vectorizer.transform([processed_input])

            # Compute similarities
            similarities = cosine_similarity(input_vector, combined_vectors).flatten()
            top_indices = similarities.argsort()[-10:][::-1]

            # Display top 10 recommendations
            st.write("### Top 10 Recommended Titles:")
            recommended_titles = df.iloc[top_indices]['Title']
            for idx, title in enumerate(recommended_titles, start=1):
                st.write(f"{idx}. {title}")
        else:
            st.warning("Please enter some text to find recommendations.")