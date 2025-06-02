import streamlit as st
import pandas as pd
import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

st.title("🧠 LDA Topic Modeling App (CSV or PDF)")

# File upload
uploaded_file = st.file_uploader("Upload a CSV or PDF file", type=["csv", "pdf"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]

    # Text extraction
    text_data = []

    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns:
            st.warning("CSV must have a 'text' column.")
        else:
            text_data = df['text'].dropna().tolist()

    elif file_type == "pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text()
        text_data = raw_text.split("\n\n")  # crude split

    if text_data:
        st.success(f"Loaded {len(text_data)} documents.")

        # LDA config
        n_topics = st.slider("Select number of topics", 2, 10, 3)
        n_words = st.slider("Select number of top words per topic", 5, 20, 10)

        # Vectorization
        vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2, max_features=1000)
        X = vectorizer.fit_transform(text_data)
        feature_names = vectorizer.get_feature_names_out()

        # LDA modeling
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)

        st.subheader("🧾 Topics:")
        for idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            st.write(f"**Topic {idx + 1}:**", ", ".join(top_terms))
