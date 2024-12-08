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
    ["📅 Publications Analysis",
     "🏷️ Keywords Analysis",
     "🔗 Co-authorship Network",
     "🏫 Institution Analysis",
     "📑 Research Type Analysis",
     "📚 Top 10 Recommended Titles Finder",
     "📈 Predict Trend"],
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
        
if "🔗 Co-authorship Network" in selected_analyses:
    with st.sidebar:
        st.subheader("🔗 Co-authorship Filters")
        year = st.selectbox("Select Year", [str(y) for y in range(2015,2025)])
        top_nodes = st.slider("Number of Top Authors to Display (Nodes):", min_value=5, max_value=30, value=10)

    st.subheader("Co-authorship Network")
    file_path = os.path.join(KAFKA_DIR, f"{year}.csv")
    if st.button("Generate Network"):
        try:
            # Process the data to create the co-authorship network graph
            H = process_data(file_path, top_nodes)

            # Position nodes using a layout
            pos = nx.spring_layout(H, seed=42)

            # Create edge traces
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

            # Create node traces
            node_trace = go.Scatter(
                x=[],
                y=[],
                mode='markers+text',  # Enable markers and text
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
                ),
                text=[],  # Initialize text attribute for node names
                textposition="top center"  # Position text above the nodes
            )
            for node in H.nodes():
                x, y = pos[node]
                node_trace['x'] += (x,)
                node_trace['y'] += (y,)
                # Add node name (author) as text for display
                node_trace['text'] += (str(node),)

            # Create the figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f"Co-authorship Network for {year}",
                    title_x=0.5,
                    showlegend=False,
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False)
                )
            )

            # Render the plot in Streamlit
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
            options=[str(year) for year in range(2015,2025)],
            default=[str(year) for year in range(2015,2025)],
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
    with st.sidebar:
            st.subheader("📚 Top 10 Recommended Titles Finder")
            selected_year = st.selectbox(
                "Select Year for Prediction",
                options=[str(year) for year in range(2015, 2025)],
                index=0,  # Default selection (2024)
                key="institution_year"
        )
        
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
 
# # Correct file path for predict_data.csv based on the project structure
file_path = os.path.join(KEYWORD_DIR, "predict_data.csv")
combined_pub_data = pd.read_csv(file_path)

# Define a simple trend prediction function by quarter
def predict_trend(data, year, quarter):
    """
    Predicts the trend for a given year and quarter.
    Assuming simple linear growth for each quarter with a fixed percentage increase.
    """
    # Get the actual percentage for the given year and quarter
    recent_data = data[(data['Year'] == year) & (data['Quarter'] == quarter)]
    if not recent_data.empty:
        recent_percentage = recent_data['Percentage'].values[0]
        predicted_percentage = recent_percentage * 1.05  # Example: 5% increase for the prediction
        return predicted_percentage
    else:
        return 0  # Return 0 if no data is found for that quarter


# Assuming 'selected_analyses' and Streamlit integration is already set up.
if "📈 Predict Trend" in selected_analyses:
    
    with st.sidebar:
        st.subheader("📈 Predict Trend")
        selected_year = st.selectbox(
            "Select Year for Prediction",
            options=[str(year) for year in range(2025, 2027)],
            index=0,  # Default selection (2024)
            key="institution_year"
        )

    # Convert selected_year to an integer
    selected_year_int = int(selected_year)

    # Show predicted trend for selected year
    if selected_year_int:
        st.write(f"Predicted Trends for Year {selected_year_int}:")

        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:  # Iterate over all quarters
            predicted_percentage = predict_trend(combined_pub_data, selected_year_int - 1, quarter)  # Predict based on previous year's data
            st.write(f"{quarter}: {predicted_percentage:.2f}%")  # Display predicted percentage for each quarter

    
    # Create two columns for side-by-side layout
    col1, col2 = st.columns([1, 2])  # Adjust the ratio as needed (e.g., [1, 2])
    
    # In column 1 (col1), show the table
    with col1:
        st.write("### Data Table")
        st.write(combined_pub_data)
    
    # In column 2 (col2), show the graph
    with col2:
        # Create the plotly figure
        fig = go.Figure()
        
        # Ensure the data is sorted by Year and Quarter for proper plotting
        combined_pub_data_sorted = combined_pub_data.sort_values(by=['Year', 'Quarter'])
        
        # Plot the actual data (only for years up to 2024)
        actual_data_up_to_2024 = combined_pub_data_sorted[combined_pub_data_sorted['Year'] <= 2024]
        
        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:  # Iterate over all quarters
            quarter_data = actual_data_up_to_2024[actual_data_up_to_2024['Quarter'] == quarter]
            fig.add_trace(go.Scatter(x=quarter_data['Year'], 
                                     y=quarter_data['Percentage'], 
                                     mode='lines+markers', 
                                     name=f'Actual Data {quarter}'))
        
        # Plot the predicted trend for each quarter (from 2024 to 2026)
        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:  # Iterate over all quarters
            predicted_percentage_2024 = predict_trend(combined_pub_data, 2024, quarter)  # Predict based on 2024 data
            predicted_percentage_2025 = predict_trend(combined_pub_data, 2025, quarter)  # Predict based on 2025 data
            predicted_percentage_2026 = predict_trend(combined_pub_data, 2026, quarter)  # Predict based on 2026 data

            # Add dashed line prediction from 2024 to 2026
            fig.add_trace(go.Scatter(x=[2024, 2025, 2026], 
                                     y=[predicted_percentage_2024, predicted_percentage_2025, predicted_percentage_2026], 
                                     mode='lines+text',  # Adding text labels
                                     name=f'Predicted Trend {quarter}', 
                                     line=dict(dash='dash'),
                                     text=[f'{predicted_percentage_2024:.2f}%', 
                                           f'{predicted_percentage_2025:.2f}%', 
                                           f'{predicted_percentage_2026:.2f}%'],  # Text labels with percentages
                                     textposition="top center"))  # Position of the text labels
        
        # Show the chart in column 2
        st.plotly_chart(fig)