import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from collections import defaultdict
import networkx as nx

# Function to load data and process co-authorship
def process_data(file_path, top_nodes):
    df = pd.read_csv(file_path)
    df = df.head(30)

    # Create a dictionary to count co-authorships
    coauthorships = defaultdict(int)

    # Loop through each paper's authors
    for authors in df['Author']:
        # Split authors by semicolon
        author_list = [author.strip() for author in authors.split(';')]

        # Only proceed if the author list has 2 or more authors
        if len(author_list) >= 2:
            # For each pair of authors, create a co-authorship edge
            for i in range(len(author_list)):
                for j in range(i + 1, len(author_list)):
                    pair = tuple(sorted([author_list[i], author_list[j]]))
                    coauthorships[pair] += 1

    # Create a graph using NetworkX
    G = nx.Graph()
    for pair, count in coauthorships.items():
        G.add_edge(pair[0], pair[1], weight=count)

    # Sort nodes by degree (number of connections) and filter top N nodes
    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_nodes]
    filtered_nodes = [node[0] for node in sorted_nodes]
    H = G.subgraph(filtered_nodes)

    return H

# Streamlit UI
st.title("Co-authorship Network Visualization")

# Select year
year = st.selectbox("Select Year to Generate Network:", ["2018", "2019", "2020", "2021", "2022", "2023"])
file_path = f"/Users/ppaamm/Desktop/little mermaid_data/CEDT-DS-Project_LittleMermaid/ExtractedData/{year}.csv"

# Select number of top nodes
top_nodes = st.slider("Number of Top Authors to Display (Nodes):", min_value=5, max_value=30, value=10)

if st.button("Generate Network"):
    try:
        # Process data and get subgraph with top nodes
        H = process_data(file_path, top_nodes)

        # Extract node positions using spring layout
        pos = nx.spring_layout(H, k=1, iterations=100)

        # Create nodes and edges for Plotly
        edge_x = []
        edge_y = []
        for edge in H.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x = []
        node_y = []
        node_labels = []
        for node in H.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_labels.append(node)

        # Plotly figure
        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(color='gray', width=0.5),
            hoverinfo='none'
        ))

        # Add nodes
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