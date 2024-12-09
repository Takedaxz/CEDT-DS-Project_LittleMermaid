# CEDT-DS-Project_LittleMermaid

## Project Objectives
This project aims to analyze engineering research data to identify key trends, collaboration patterns, and emerging topics while presenting actionable insights through a comprehensive and visually engaging pipeline.

## Data Sources
1. **Scopus API**
   - Time range: 2015-2024
   - Data collected: Title, Abstract, Authors, Aggregation type, Publisher, Publication Date, Institutions, and Keywords

2. **Web Scraping (ArXiv)**
   - Data collected: Title, Abstract, Authors, and Publication Date

3. **[Raw Data (JSON)](https://drive.google.com/file/d/107WikNVtve-QY7I7-pMsdFFHpAnNFxmO/view?usp=sharing)**
   - Additional data from provided sources

## Technical Architecture

### Data Engineering Pipeline
1. **Data Collection**
   - Scopus API integration
   - ArXiv web scraping
   - Raw JSON data processing

2. **Data Processing**
   - Kafka streaming implementation on Docker
   - Year-wise topic segregation
   - Producer-Consumer architecture for data organization

3. **Data Cleansing**
   - Text preprocessing
   - Data standardization
   - Quality checks

### Analysis Components

#### 1. Topic Modeling
- Uncovers latent thematic structures in unstructured text data
- Groups words into coherent topics
- Assigns topics to documents

#### 2. Recommendation System
- NLTK-based text processing
- Similarity matching
- Research article recommendations based on input queries

#### 3. Publication Trend Analysis
- Quarterly publication trend analysis
- Year-over-year comparison
- Future trend predictions

#### 4. Network Analysis
- **Coauthor Network Visualization**
  - Author collaboration mapping
  - Research cluster identification
  - Influential author analysis

#### 5. Content Analysis
- **Topic Analysis**
  - Year-wise trending research topics
  - Theme extraction using ML
  - Research evolution tracking

- **Keyword Analysis**
  - Temporal keyword comparison
  - Trend identification
  - Research landscape mapping

3. **Emerging Topics (2024)**
   - AI and ML applications in industry
   - Sustainability
   - Environmental concerns
   - Agriculture
   - Aggregation systems
