# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import nltk

# # Ensure necessary NLTK downloads
# nltk.download('stopwords')
# nltk.download('punkt_tab')

# # Preprocessing setup
# stop_words = set(stopwords.words('english'))

# def preprocess_text(text):
#     """
#     Preprocesses the given text:
#     - Tokenizes
#     - Converts to lowercase
#     - Removes stopwords
#     - Keeps only alphanumeric tokens
#     """
#     tokens = word_tokenize(text.lower())
#     tokens = [word for word in tokens if word.isalnum()]
#     tokens = [word for word in tokens if word not in stop_words]
#     return ' '.join(tokens)

# # Load the data
# @st.cache_data
# def load_data(year):
#     data_path = f'/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/Kafka/output_csv/{year}.csv'
#     df = pd.read_csv(data_path)
#     df['Abstract'] = df['Abstract'].fillna('')
#     df['Processed_Abstract'] = df['Abstract'].apply(preprocess_text)
#     return df

# # Initialize TF-IDF Vectorizer
# @st.cache_resource
# def vectorize_data(df):
#     df['Abstract'] = df['Abstract'].fillna('')
#     df['Title'] = df['Title'].fillna('')
#     vectorizer = TfidfVectorizer()
#     combined_vectors = vectorizer.fit_transform(df['Abstract'] + " " + df['Title'])
#     return vectorizer, combined_vectors

# # Streamlit app
# def main():
#     st.title("Top 10 Recommended Titles Finder")

#     # Load year selection
#     year = st.sidebar.selectbox("Select Year", ['2018', '2019', '2020','2021','2022','2023'], index=0)
#     df = load_data(year)
#     st.write(f"Data Loaded for the Year: {year}")

#     vectorizer, combined_vectors = vectorize_data(df)

#     # User input
#     input_data = st.text_area("Enter Input Text:", placeholder="Type or paste your input here...")
#     if st.button("Find Recommendations"):
#         if input_data.strip():
#             # Preprocess and vectorize user input
#             processed_input = preprocess_text(input_data)
#             input_vector = vectorizer.transform([processed_input])

#             # Compute similarities
#             similarities = cosine_similarity(input_vector, combined_vectors).flatten()
#             top_indices = similarities.argsort()[-10:][::-1]

#             # Display top 10 recommendations
#             st.write("### Top 10 Recommended Titles:")
#             recommended_titles = df.iloc[top_indices]['Title']
#             for idx, title in enumerate(recommended_titles, start=1):
#                 st.write(f"{idx}. {title}")
#         else:
#             st.warning("Please enter some text to find recommendations.")

# if __name__ == "__main__":
#     main()





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
nltk.download('punkt')

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

# Base directory setup relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Current directory of the script
PROJECT_DIR = os.path.join(BASE_DIR, "..", "..")  # Adjust path to two levels above
KAFKA_DIR = os.path.join(PROJECT_DIR, "Kafka", "output_csv")  # Path for publications
KEYWORD_DIR = os.path.join(BASE_DIR, "VisualizeData")  # Path for keywords data

# File paths for publications and keywords
pub_file_names = [
    os.path.join(KAFKA_DIR, f"{year}.csv") for year in range(2015,2025)
]
keyword_file_names = [
    os.path.join(KEYWORD_DIR, f"{year}_keywords_counts.csv") for year in range(2015,2025)
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

# Load the data
@st.cache_data
def load_data(year):
    """
    Load data for the specified year from the dynamically constructed path.
    """
    data_path = os.path.join(KAFKA_DIR, f"{year}.csv")  # Use the correct dynamic path
    if not os.path.exists(data_path):
        st.error(f"File for year {year} not found: {data_path}")
        st.stop() 
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

# Set Streamlit page configuration
st.set_page_config(
    page_title="Research Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📚 Research Analysis Dashboard")
st.markdown("""
Explore **Co-authorship Networks**, **Research Publications**, and **Top Keywords** and others in one unified dashboard. 
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
    ["📅 Publications Analysis", "Predict Trend"],
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
            