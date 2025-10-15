import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
# Use the public GitHub URL for your CSV file for reliable deployment
DATA_URL = 'https://raw.githubusercontent.com/Harshit-M0307/TravelEXP-MiniProject/main/travel_recommendation/travel_data_up.csv'

# --- Data Loading and Preprocessing ---

# Use st.cache_data to load the data only once across all users and runs
@st.cache_data
def load_data(url):
    """Loads and preprocesses the travel data."""
    try:
        df = pd.read_csv(url)
        df['features'] = df['Type'] + " " + df['Description']
        return df
    except Exception as e:
        st.error(f"Error loading data from the URL: {e}")
        return pd.DataFrame() # Return empty DF on failure

# Load the data
df = load_data(DATA_URL)

# Use st.cache_resource for heavy computations that only need to run once
@st.cache_resource
def compute_similarity(data_frame):
    """Computes the TF-IDF matrix and cosine similarity."""
    if data_frame.empty:
        return None, None
        
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data_frame['features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf_matrix, cosine_sim

# Compute the matrix and similarity
tfidf_matrix, cosine_sim = compute_similarity(df)

# Recommendation function
def recommend(destination_name, top_n=3, sim_matrix=cosine_sim, data_frame=df):
    """Generates destination recommendations based on cosine similarity."""
    if sim_matrix is None or data_frame.empty or destination_name not in data_frame['Destination'].values:
        return []

    idx = data_frame[data_frame['Destination'] == destination_name].index[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:top_n+1]

    recommended = [data_frame.iloc[i[0]] for i in scores]
    return recommended

# --- Streamlit UI ---
st.set_page_config(page_title="Travel Recommender", layout="centered")

st.title("🌍 Travel Recommendation System")
st.markdown("Discover new destinations based on your favorite spot!")

if not df.empty and cosine_sim is not None:
    selected = st.selectbox("Choose a destination you like:", df['Destination'].values)

    if st.button("Recommend Similar Places"):
        with st.spinner('Finding similar destinations...'):
            recs = recommend(selected)
            
            if recs:
                st.success(f"Top {len(recs)} Recommendations for {selected}:")
                for rec in recs:
                    st.subheader(rec['Destination'] + f" ({rec['Country']})")
                    st.markdown(f"*Type:* **{rec['Type']}**")
                    st.markdown(f"*About:* {rec['Description']}")
                    
                    # This line is now set up to use the public URL from your corrected CSV
                    st.image(rec['Images'], caption=rec['Destination']) 
                    st.markdown("---")
            else:
                st.error("No similar recommendations found.")
else:
    st.error("The application failed to load necessary data or resources. Please ensure the CSV file is accessible.")