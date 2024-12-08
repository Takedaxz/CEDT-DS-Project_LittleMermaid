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
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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

# Validate files for publications and keywords
def validate_files(file_paths):
    missing_files = [file for file in file_paths if not os.path.exists(file)]
    if missing_files:
        st.error(f"Missing files: {', '.join(missing_files)}")
        st.stop()

# Validate files
validate_files(pub_file_names)
validate_files(keyword_file_names)

# Set Streamlit page configuration
st.set_page_config(page_title="Research Analysis Dashboard", page_icon="📊", layout="wide")

# Title and description
st.title("📚 Research Analysis Dashboard")
st.markdown("""
Explore **Co-authorship Networks**, **Research Publications**, and **Top Keywords** in one unified dashboard. 
Choose intervals, years, and chart types to visualize trends effectively.
""")

# Function for plotting publication data
def plot_publication_data(counts, interval, chart_type):
    if chart_type == "Bar Chart":
        fig = px.bar(counts, x=counts.index.astype(str), y=counts, labels={'x': interval, 'y': 'Number of Publications'}, title=f"Publications by {interval}")
    elif chart_type == "Line Chart":
        fig = px.line(counts, x=counts.index.astype(str), y=counts, labels={'x': interval, 'y': 'Number of Publications'}, title=f"Publications by {interval}")
    elif chart_type == "Pie Chart":
        fig = px.pie(counts, names=counts.index.astype(str), values=counts, title=f"Publications by {interval}")
    return fig

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
            fig = plot_publication_data(counts, interval, chart_type)
            st.plotly_chart(fig, use_container_width=True)

            # Summary for Publications
            st.subheader("📊 Summary")
            st.write(f"- **Total Publications Analyzed**: {counts.sum()}")
            st.write(f"- **Highest Interval**: {counts.idxmax()} ({counts.max()} publications)")

# Keywords Analysis
if "🏷️ Keywords Analysis" in selected_analyses:
    with st.sidebar:
        st.subheader("🏷️ Keywords Filters")
        show_wordcloud = st.checkbox("Display WordCloud", value=True)
        selected_years = st.multiselect(
            "Select Years", 
            options=combined_keyword_data['Year'].unique(), 
            default=combined_keyword_data['Year'].unique()
        )
        chart_type = st.radio("Select Chart Type", ["Bar Chart", "Scatter Chart", "Pie Chart (Select Single Year)"], key="keyword_chart")
        

    st.subheader("Keywords Analysis")
    
    if selected_years:
        filtered_keywords = combined_keyword_data[combined_keyword_data['Year'].isin(selected_years)]
        
        # Create a DataFrame for Top 5 Keywords for each year
        top5_keywords = filtered_keywords.groupby(['Year', 'Keywords']).sum().reset_index()
        top5_keywords = top5_keywords.sort_values(['Year', 'Count'], ascending=[True, False]).groupby('Year').head(5)

        # Summary for Keywords
        st.subheader("📊 Summary")
        
        # หาคำที่ใช้มากที่สุดในทุกปี
        overall_top_keywords = filtered_keywords.groupby('Keywords')['Count'].sum().reset_index()
        overall_top_keywords = overall_top_keywords.sort_values(by='Count', ascending=False).head(1)

        total_keywords = len(filtered_keywords['Keywords'].unique())
        top_keyword = overall_top_keywords['Keywords'].values[0] if not overall_top_keywords.empty else "N/A"
        top_keyword_count = overall_top_keywords['Count'].values[0] if not overall_top_keywords.empty else 0
        
        st.write(f"- **Total Keywords Analyzed**: {total_keywords}")
        st.write(f"- **Top Keyword**: {top_keyword} ({top_keyword_count} mentions)")

        if show_wordcloud:
            st.subheader("WordCloud of Keywords")
            keyword_freq = dict(zip(filtered_keywords['Keywords'], filtered_keywords['Count']))
            wordcloud = WordCloud(
                width=800, height=400, background_color="white", colormap="viridis"
            ).generate_from_frequencies(keyword_freq)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

        if chart_type == "Bar Chart":
            fig = px.bar(
                top5_keywords, 
                x='Keywords', 
                y='Count', 
                color='Year',
                title="Top 5 Keywords by Year",
                labels={'Keywords': 'Keywords', 'Count': 'Total Count'},
                text='Count',
                barmode='group'
            )
        elif chart_type == "Scatter Chart":
            # Create a scatter chart showing counts of top keywords across selected years
            fig = px.scatter(
                top5_keywords,
                x='Keywords',
                y='Count',
                color='Year',
                title="Top 5 Keywords Count by Year",
                labels={'Keywords': 'Keywords', 'Count': 'Total Count'},
                size='Count',  # Set size of markers based on Count
                size_max=20,
                hover_name='Keywords'
            )
        elif chart_type == "Pie Chart (Select Single Year)":
            if len(selected_years) == 1:
                selected_year = selected_years[0]
                pie_data = filtered_keywords[filtered_keywords['Year'] == selected_year]

                # Find Top 5 Keywords
                top5_keywords_pie = pie_data.nlargest(5, 'Count')

                fig = px.pie(
                    top5_keywords_pie, 
                    names='Keywords', 
                    values='Count', 
                    title=f"Top 5 Keywords in {selected_year}"
                )
            else:
                st.warning("Please select a single year to view the Pie Chart.")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least one year.")
# Co-authorship Network
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

# Top 10 Recommended Titles Finder
if "📚 Top 10 Recommended Titles Finder" in selected_analyses:
    st.title("Top 10 Recommended Titles Finder")

    # Load year selection for publications data
    year = st.sidebar.selectbox("Select Year", ['2018', '2019', '2020', '2021', '2022', '2023'])

    # Get data and vectorize
    df = load_data(year)
    vectorizer, vectors = vectorize_data(df)

    st.write("Input the title of a research paper to find similar titles:")
    query_title = st.text_input("Enter Research Paper Title")

    if query_title:
        query_vec = vectorizer.transform([query_title])
        cosine_similarities = cosine_similarity(query_vec, vectors).flatten()
        similar_indices = cosine_similarities.argsort()[-10:][::-1]
        st.write("Top 10 Similar Titles:")
        for i in similar_indices:
            st.write(f"- {df.iloc[i]['Title']}")

