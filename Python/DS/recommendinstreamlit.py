import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

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

# Load the data
@st.cache_data
def load_data(year):
    data_path = f'/Users/naphat-c/Documents/naphat/Data/project/CEDT-DS-Project_LittleMermaid/ExtractedData/{year}.csv'
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

# Streamlit app
def main():
    st.title("Top 10 Recommended Titles Finder")

    # Load year selection
    year = st.sidebar.selectbox("Select Year", ['2018', '2019', '2020','2021','2022','2023'], index=0)
    df = load_data(year)
    st.write(f"Data Loaded for the Year: {year}")

    vectorizer, combined_vectors = vectorize_data(df)

    # User input
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

if __name__ == "__main__":
    main()
