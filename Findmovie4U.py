import pandas as pd
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

st.set_page_config(
    page_title="MovieMatch: Your Personal Film Guide",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 Styling
st.markdown("""
    <style>
        .main { background-color: #0E1117; }
        .sidebar .sidebar-content { background-color: #1a1a2e; }
        h1, h2, h3, h4, h5, h6 { color: #FFD700 !important; }
        .stTextInput>div>div>input {
            background-color: #2d2d44; color: white;
        }
        .stButton>button {
            background-color: #FF5733; color: white;
            border-radius: 10px; padding: 10px 24px;
        }
        .stButton>button:hover { background-color: #C70039; color: white; }
        .movie-card {
            background-color: #16213E; border-radius: 10px;
            padding: 15px; margin: 10px 0;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        }
        .movie-card:hover {
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.3);
        }
        .quote-box {
            background-color: #1a1a2e;
            border-left: 5px solid #FFD700;
            padding: 10px 20px;
            border-radius: 0 10px 10px 0;
        }
        .fact-box {
            background-color: #1a1a2e;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #FF5733;
        }
    </style>
""", unsafe_allow_html=True)

# Quotes & facts
movie_quotes = [
    "May the Force be with you.", "I'm going to make him an offer he can't refuse.",
    "Why so serious?", "Life is like a box of chocolates...", "Just keep swimming."
]
movie_facts = [
    "The Godfather used a real horse head.", "Inception had 500+ VFX shots.",
    "Endgame's budget was over $350 million.", "Parasite was the first non-English Best Picture.",
    "The Joker's style was inspired by punk rock."
]

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🎬 MovieMatch</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Discover your next favorite film!</p>", unsafe_allow_html=True)
    st.markdown("### 🎭 Filter by Genre")
    selected_genre = st.selectbox("Select a genre", ["All", "Action", "Comedy", "Drama", "Sci-Fi", "Horror", "Romance", "Thriller"])
    if st.button("🎲 Get Random Movie"):
        st.session_state.random_movie = True

# Main columns
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("<h1 style='text-align: center; color: #FFD700;'>🎬 MovieMatch: Your Personal Film Guide</h1>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="quote-box">
            <h3 style='text-align: center;'>✨ Movie Quote of the Day ✨</h3>
            <h4 style='text-align: center;'>{random.choice(movie_quotes)}</h4>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>🔍 Find Similar Movies</h2>", unsafe_allow_html=True)
    search_col1, search_col2 = st.columns([4, 1])
    with search_col1:
        movie_name = st.text_input("", placeholder="Type a movie name here...", key="movie_search")
    with search_col2:
        if st.button("Search"):
            st.session_state.search_clicked = True

with col2:
    st.markdown(f"""
        <div class="fact-box">
            <h3 style='text-align: center;'>🎭 Did You Know?</h3>
            <p>{random.choice(movie_facts)}</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>🍿 Popular This Week</h3>", unsafe_allow_html=True)
    for movie in ["The Shawshank Redemption", "Pulp Fiction", "The Dark Knight"]:
        st.markdown(f"<div class='movie-card'>{movie}</div>", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("action.csv")
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={"movie_name": "title", "genre": "genres", "description": "overview"}, inplace=True)
    if "title" not in df or "genres" not in df:
        st.error("❌ Required columns ('title' or 'genres') not found!")
        st.stop()
    df["title"] = df["title"].astype(str).str.strip().str.lower()
    df["genres"] = df["genres"].str.replace("|", " ", regex=True).fillna("").str.strip()
    df["overview"] = df["overview"].fillna("") if "overview" in df else "No overview"
    df["content"] = df["title"] + " " + df["genres"] + " " + df["overview"]
    df = df[df["content"].str.strip() != ""]
    return df

movies = load_data()

# TF-IDF
@st.cache_data
def compute_tfidf(data):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    return vectorizer.fit_transform(data)

tfidf_matrix = compute_tfidf(movies["content"])

def train_nearest_neighbors(tfidf_matrix):
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=6)
    nn.fit(tfidf_matrix)
    return nn

model_nn = train_nearest_neighbors(tfidf_matrix)

def recommend_movie(movie_title):
    try:
        movie_title = movie_title.strip().lower()
        exact = [t for t in movies["title"] if t == movie_title]
        partial = [t for t in movies["title"] if movie_title in t]
        if not exact and not partial:
            st.error(f"❌ Movie '{movie_title}' not found!")
            return []
        title = exact[0] if exact else partial[0]
        idx = movies[movies["title"] == title].index[0]
        _, indices = model_nn.kneighbors(tfidf_matrix[idx])
        return [movies.iloc[i]["title"].title() for i in indices[0][1:]]
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return []

# Show recommendations
if movie_name and (st.session_state.get("search_clicked") or movie_name):
    recommendations = recommend_movie(movie_name)
    if recommendations:
        st.markdown("<h2 style='text-align: center; color: #FFD700;'>🎬 Recommended Movies</h2>", unsafe_allow_html=True)
        for i, rec in enumerate(recommendations, 1):
            movie_data = movies[movies["title"].str.lower() == rec.lower()]
            if not movie_data.empty:
                genres = movie_data["genres"].values[0]
                rating = movie_data["rating"].values[0]
                year = movie_data["year"].values[0]
                director = movie_data["director"].values[0]
            else:
                genres = rating = year = director = "N/A"
            st.markdown(f"""
                <div class="movie-card">
                    <h3>#{i}: {rec}</h3>
                    <p><strong>Genres:</strong> {genres}</p>
                    <p><strong>Rating:</strong> {rating}</p>
                    <p><strong>Year:</strong> {year}</p>
                    <p><strong>Director:</strong> {director}</p>
                </div>
            """, unsafe_allow_html=True)

# Show random movie
if st.session_state.get("random_movie", False):
    random_movie = random.choice(movies["title"].tolist()).title()
    st.markdown("<h2 style='text-align: center; color: #FF5733;'>🎲 Your Random Movie Pick</h2>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="movie-card" style="text-align: center;">
            <h2>{random_movie}</h2>
            <p>Why not give this one a try?</p>
        </div>
    """, unsafe_allow_html=True)
    st.session_state.random_movie = False
