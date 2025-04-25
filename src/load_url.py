from IPython.core.display import display, Markdown
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import torch
import re

file = pd.read_csv("data/MachineLearningMastery.csv")

# If 'Citations' column exists, process it; otherwise skip the filtering step
if "Citations" in file.columns:
    # Ensure 'Citations' column is numeric
    file["Citations"] = pd.to_numeric(file["Citations"], errors="coerce")

    # Drop rows with missing citation values
    file = file.dropna(subset=["Citations"])

    # Filter top 30% by citations
    threshold = file["Citations"].quantile(0.70)
    df = file[file["Citations"] > threshold].reset_index(drop=True)

else:
    df = file.copy().reset_index(drop=True)
# Determine the best text column to use
# It will be usefull for datasets like "Youtube" which does not have abstract
text_column = "Abstract" if "Abstract" in df.columns else "Title"
if "Abstract" in df.columns and "Title" in df.columns:
    df["Abstract"] = df["Abstract"].fillna(df["Title"])
else:
    print(
        "'Abstract' column missing or only 'Title' is available. Please skipping fill."
    )


# Preprocess text
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"\d+", "", text)  # Remove numbers
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = text.strip()  # Remove extra spaces
    return text


# Apply preprocessing
df[text_column] = df[text_column].fillna("").apply(clean_text)

# Generate embeddings efficiently
print("Generating text embeddings...")
embeddings = model.encode(
    df[text_column].tolist(), batch_size=128, show_progress_bar=True
)

# Assign raw embeddings and reduced vectors
df["Embeddings"] = list(embeddings)  # Raw BERT embeddings
df["ReducedEmbeddings"] = list(data)  # Reduced vectors from autoencoder

# First-level clustering using reduced embeddings
kmeans_model = KMeans(n_clusters=23, n_init="auto", random_state=42)
df["Cluster"] = kmeans_model.fit_predict(data)

# Load BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define which text column is being used ("Title" or "Abstract")
text_column = "Title"  # or "Abstract"


def search_similar_resources(
    query, df, top_n=10, min_similarity=0.50, title_cluster_k=3
):
    if not query.strip():
        print("Query is empty. Please enter valid keywords.")
        return pd.DataFrame()

    # Encode and reduce query
    query_embedding = model.encode([query])

    autoencoder.eval()
    with torch.no_grad():
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32).to(device)
        query_embedding_reduced = autoencoder.encoder(query_tensor).cpu().numpy()

    # Find most relevant main cluster
    query_cluster = kmeans_model.predict(query_embedding_reduced)[0]
    cluster_df = df[df["Cluster"] == query_cluster].copy()

    if cluster_df.empty:
        print("No relevant resources found in the identified cluster.")
        return pd.DataFrame()

    # Compute cosine similarity
    cluster_embeddings = np.vstack(cluster_df["Embeddings"].to_numpy())
    similarity_scores = cosine_similarity(query_embedding, cluster_embeddings).flatten()
    cluster_df["Similarity"] = similarity_scores
    cluster_df = cluster_df[cluster_df["Similarity"] >= min_similarity]

    if cluster_df.empty:
        print("No resources met the minimum similarity threshold.")
        return pd.DataFrame()

    # Title-Level Clustering if dataset has both abstract and title
    if text_column.lower() == "abstract":
        text_data = cluster_df["Title"].tolist()
        title_embeddings = model.encode(text_data)

        if len(title_embeddings) < title_cluster_k:
            title_cluster_k = max(1, len(title_embeddings))

        title_kmeans = KMeans(n_clusters=title_cluster_k, random_state=42)
        cluster_df["TitleCluster"] = title_kmeans.fit_predict(title_embeddings)

        query_title_embedding = model.encode([query])
        query_title_cluster = title_kmeans.predict(query_title_embedding)[0]

        cluster_df = cluster_df[
            cluster_df["TitleCluster"] == query_title_cluster
        ].copy()
        if cluster_df.empty:
            print("No relevant titles found in the identified subcluster.")
            return pd.DataFrame()

    # Final selection
    # Handle error if the dataset has not URL column
    cluster_df = cluster_df.drop_duplicates(subset=["Title", "Similarity"])
    top_results = cluster_df.sort_values(by="Similarity", ascending=False).head(top_n)

    # Check if 'URL' exists before selecting it
    if "URL" in cluster_df.columns:
        return top_results[["Title", "URL", "Similarity"]]
    else:
        print("'URL' column not found. Returning results without URLs.")
        return top_results[["Title", "Similarity"]]


def displayResults(recommendations):
    """
    Display search results as Markdown in Jupyter Notebook.
    Handles missing URL column gracefully.
    """
    if recommendations.empty:
        display(Markdown("**No results to display.**"))
        return

    output = "### 🔍 Search Results:\n"
    url_exists = "URL" in recommendations.columns

    for index, row in recommendations.iterrows():
        title = row["Title"]
        score = row["Similarity"]

        if url_exists and pd.notna(row["URL"]):
            output += f"- **Title:** [{title}]({row['URL']})\n"
        else:
            output += f"- **Title:** {title}\n"

        output += f"  - ⭐ **Similarity Score:** {score:.2f}\n\n"

    display(Markdown(output))


# User Interface
query = input("Enter keywords to search: ")
recommendations = search_similar_resources(
    query, df, top_n=10, min_similarity=0.50, title_cluster_k=3
)
# Call function
displayResults(recommendations)
