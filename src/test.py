import re
import os
import random
import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from kneed import KneeLocator
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display, Markdown


# Preprocess text
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"\d+", "", text)  # Remove numbers
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = text.strip()  # Remove extra spaces
    return text


# Lemmatization
def lemmatize_with_spacy(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text.lower())
    lemmatized_words = [token.lemma_ for token in doc]
    return " ".join(lemmatized_words)


# Reproducibility Setup fuction
def setRandomSeed(seed=42):
    """
    Sets seeds across libraries to ensure reproducibility of results.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Autoencoder Model function
class Autoencoder(nn.Module):
    """
    Defines a simple symmetric autoencoder using two fully connected layers
    for both encoding and decoding.
    """

    def __init__(self, input_dim, encoding_dim=64):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),  # First layer reduces to 128
            nn.ReLU(),
            nn.Linear(128, encoding_dim),  # Bottleneck layer
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),  # Expand back from bottleneck
            nn.ReLU(),
            nn.Linear(128, input_dim),  # Final output layer (same size as input)
        )

    def forward(self, x):
        """
        Defines the forward pass through encoder and decoder.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Training Function
def trainAutoencoder(
    model, data_tensor, epochs=50, batch_size=32, learning_rate=0.001, val_ratio=0.2
):
    """
    Trains the autoencoder with validation loss monitoring.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Split data into training and validation
    data_np = data_tensor.cpu().numpy()
    train_data, val_data = train_test_split(
        data_np, test_size=val_ratio, random_state=42
    )
    train_tensor = torch.tensor(train_data, dtype=torch.float32).to(data_tensor.device)
    val_tensor = torch.tensor(val_data, dtype=torch.float32).to(data_tensor.device)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        epoch_train_loss = 0.0
        for batch in train_loader:
            inputs = batch[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tensor)
            val_loss = criterion(val_outputs, val_tensor).item()
        val_losses.append(val_loss)
        model.train()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}"
            )

    return model, train_losses, val_losses


# Evaluation & Visualization function
def evaluateReconstruction(original_data, reconstructed_data, threshold_percentile=95):
    """
    Computes reconstruction errors and identifies poorly reconstructed samples
    based on a percentile threshold.
    """
    original_np = original_data.detach().cpu().numpy()
    reconstructed_np = reconstructed_data.detach().cpu().numpy()

    # MSE per sample
    errors = np.mean((original_np - reconstructed_np) ** 2, axis=1)

    # Threshold at the given percentile (e.g., 95%)
    threshold = np.percentile(errors, threshold_percentile)

    # Indices where reconstruction error is greater than threshold
    outlier_indices = np.where(errors > threshold)[0]

    return errors, threshold, outlier_indices


def plotErrors(errors, fig_dir):
    """
    Visualizes the distribution of reconstruction errors.
    Highlights the 95th percentile cutoff for detecting poor reconstructions.
    """
    # plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=50, kde=True, color="skyblue", edgecolor="black")
    plt.axvline(
        np.percentile(errors, 95),
        color="red",
        linestyle="--",
        linewidth=2,
        label="95th Percentile",
    )
    plt.title("Distribution of Reconstruction Errors", fontsize=16)
    plt.xlabel("Reconstruction MSE", fontsize=14)
    plt.ylabel("Number of Samples", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}recon_error.png")


def plotLossCurve(train_losses, val_losses, fig_dir):
    """
    Plots both training and validation loss curves.
    """
    # plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training vs Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}loss.png")


# Main Function
def main(embedding_vectors_np, fig_dir):
    """
    Main pipeline to:
    - Prepare data
    - Train autoencoder
    - Generate encoded & reconstructed vectors
    - Evaluate reconstruction quality
    - Plot errors and training performance
    """

    global autoencoder, device

    # Ensure reproducibility
    setRandomSeed(42)

    input_dim = embedding_vectors_np.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert input data to torch tensor
    data_tensor = torch.tensor(embedding_vectors_np, dtype=torch.float32).to(device)

    # Build and train autoencoder
    model = Autoencoder(input_dim=input_dim, encoding_dim=64).to(device)
    model, train_loss, val_loss = trainAutoencoder(model, data_tensor)

    # Set model as global encoder for use in search engine
    autoencoder = model

    # Switch to evaluation mode for inference
    model.eval()
    with torch.no_grad():
        encoded = model.encoder(
            data_tensor
        )  # Get bottleneck (compressed) representations
        reconstructed = model(data_tensor)  # Reconstruct input

    # Analyze reconstruction quality
    errors, threshold, outliers = evaluateReconstruction(data_tensor, reconstructed)
    print(f"\nNumber of poorly reconstructed samples: {len(outliers)}")

    # Visualizations
    plotErrors(errors, fig_dir)
    plotLossCurve(train_loss, val_loss, fig_dir)

    return encoded.cpu().numpy(), reconstructed.cpu().numpy(), errors, outliers


def search_similar_resources(
    query,
    df,
    n_clusters,
    top_n=10,
    min_similarity=0.10,
    title_cluster_k=3,
    data=[],
):
    # First-level clustering using reduced embeddings
    kmeans_model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    df["Cluster"] = kmeans_model.fit_predict(data)

    # Load BERT model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Define which text column is being used ("Title" or "Abstract")
    text_column = "Title"  # or "Abstract"

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

    md = Markdown(output)
    display(md.data)


def iteration(
    input_keyword="data/Arxiv_Resources.csv",
    output_keyword="txt/output_keywords.txt",
    SBERT_pretrain="all-MiniLM-L6-v2",
    fig_dir="fig/iter1_cluslter0/",
):
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    file = pd.read_csv(input_keyword)

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

    print(df.columns)
    print(df.shape)
    print(df.head())

    # Determine the best text column to use
    # It will be usefull for datasets like "Youtube" which does not have abstract
    text_column = "Abstract" if "Abstract" in df.columns else "Title"
    print(f"Clustering through {text_column}")

    if "Abstract" in df.columns and "Title" in df.columns:
        df["Abstract"] = df["Abstract"].fillna(df["Title"])
    else:
        print(
            "'Abstract' column missing or only 'Title' is available. Please skipping fill."
        )

    # Method adapted from:
    # Rahultiwari.(2024). Unlocking the Power of Sentence Embeddings with all-MiniLM-L6-v2. Medium
    # Retrieved from https://medium.com/@rahultiwari065/unlocking-the-power-of-sentence-embeddings-with-all-minilm-l6-v2-7d6589a5f0aa

    # Load Sentence-BERT model
    model = SentenceTransformer(SBERT_pretrain)

    # Apply lemmatization
    df["lemmatized_text"] = df[text_column].apply(lemmatize_with_spacy)

    # Apply preprocessing
    df[text_column] = df["lemmatized_text"].fillna("").apply(clean_text)

    # Generate embeddings efficiently
    print("Generating text embeddings...")
    embeddings = model.encode(
        df[text_column].tolist(), batch_size=128, show_progress_bar=True
    )

    print(embeddings.shape)
    if embeddings.shape[0] < 30:
        print("The embedding size is too small")
        print(f"Stop the clustering of {fig_dir}")
        return
    # Method adapted from:
    # GeeksforGeeks.(2025). Implementing an Autoencoder in PyTorch. Retrieved from https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/

    # Run the Full Pipeline
    data, reconstructed, reconstruction_errors, bad_indices = main(embeddings, fig_dir)

    print(f"data shape: {data.shape}")

    # Initialize lists to store WCSS and Silhouette Scores
    wcss = []
    silhouette_scores = []

    # Define the range of cluster numbers from 3 to 30 to test
    K_nums = range(3, 31)

    # Iterate over different values of k to evaluate clustering performance
    for k in K_nums:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)

        wcss.append(kmeans.inertia_)

        silhouette_avg = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)

    # Set seaborn settings
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)

    # Find the best elbow point
    kneedle = KneeLocator(K_nums, wcss, curve="convex", direction="decreasing")
    elbow_score = kneedle.elbow

    # Find the best silhouette score
    bestScore = K_nums[silhouette_scores.index(max(silhouette_scores))]

    # Figure and twin axes
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot WCSS
    sns.lineplot(x=K_nums, y=wcss, marker="o", ax=ax1, label="Inertia", color="blue")
    ax1.axvline(
        elbow_score, color="blue", linestyle="--", label=f"Elbow at k={elbow_score}"
    )
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("Inertia", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Create twin axis for silhouette
    ax2 = ax1.twinx()
    sns.lineplot(
        x=K_nums,
        y=silhouette_scores,
        marker="s",
        ax=ax2,
        label="Silhouette Score",
        color="red",
    )
    ax2.axvline(
        bestScore,
        color="red",
        linestyle="--",
        label=f"Best Silhouette at k={bestScore}",
    )
    ax2.set_ylabel("Silhouette Score", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    # Add title and layout tweaks
    plt.title("Elbow Method vs Silhouette Score")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}elbow_silhouette.png")

    # Reduce data to 2D
    pca = PCA(n_components=2, random_state=42)
    pca_embeddings = pca.fit_transform(data)

    # Create a DataFrame with the 2D coordinates
    pca_df = pd.DataFrame(pca_embeddings, columns=["PCA1", "PCA2"])

    # Fit KMeans with 12 and 23 clusters
    kmeans_14 = KMeans(n_clusters=elbow_score, n_init="auto", random_state=42).fit(
        pca_embeddings
    )
    kmeans_22 = KMeans(n_clusters=bestScore, n_init="auto", random_state=42).fit(
        pca_embeddings
    )

    # Add cluster labels to the DataFrame
    pca_df["Cluster_12"] = kmeans_14.labels_.astype(str)
    pca_df["Cluster_23"] = kmeans_22.labels_.astype(str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.scatterplot(
        data=pca_df,
        x="PCA1",
        y="PCA2",
        hue="Cluster_12",
        palette="tab10",
        s=20,
        ax=axes[0],
        legend=False,
    )
    axes[0].set_title(f"KMeans with {elbow_score} Clusters")

    sns.scatterplot(
        data=pca_df,
        x="PCA1",
        y="PCA2",
        hue="Cluster_23",
        palette="tab20",
        s=20,
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title(f"KMeans with {bestScore} Clusters")

    plt.suptitle("Kmeans Clustering Shape Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}kmeans.png")

    # Method adopted from:
    # Geeksforgeeks.(2024).Mastering TF-IDF Calculation with Pandas DataFrame in Python.
    # Retrieved from https://www.geeksforgeeks.org/mastering-tf-idf-calculation-with-pandas-dataframe-in-python/

    # Fit kmeans model on the data
    KMeans_data = KMeans(n_clusters=bestScore, random_state=42)
    KMeans_labels = KMeans_data.fit_predict(data)

    # Assign cluster labels to the DataFrame
    df["Cluster"] = KMeans_labels

    # Initialize TF-IDF vectorizer (ignoring common English stopwords and limiting features to 2000)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)

    # Transform the original text data into a TF-IDF matrix
    X_tfidf = vectorizer.fit_transform(df[text_column])

    # Get the list of feature names (The words in the TF-IDF vocab)
    terms = vectorizer.get_feature_names_out()

    top_n = 10  # Number of top keywords to extract per cluster
    cluster_keywords = {}

    # Loop over each cluster to find top keywords
    for cluster_num in sorted(df["Cluster"].unique()):
        # Get the indices of documents in the current cluster
        cluster_indices = df[df["Cluster"] == cluster_num].index.to_list()

        # Compute the mean TF-IDF score for each word across all docs in the cluster
        cluster_tfidf = X_tfidf[cluster_indices].mean(axis=0)

        # Convert sparse matrix to a flat NumPy array
        cluster_array = np.squeeze(np.asarray(cluster_tfidf))

        # Get indices of the top n highest scoring words
        top_words = cluster_array.argsort()[::-1][:top_n]

        # Retrieve the actual words corresponding to those indices
        keywords = [terms[i] for i in top_words]

        # Store the keywords for this cluster
        cluster_keywords[cluster_num] = keywords

    df_output_keywords = pd.DataFrame(columns=["keyword"])
    # Print the top keywords for each cluster
    for cluster, keywords in cluster_keywords.items():
        df_output_keywords.loc[cluster] = ", ".join(keywords)
        print(f"\n🔹Cluster {cluster} — Top Keywords:")
        print(", ".join(keywords))
    df_output_keywords.to_csv(output_keyword, sep="\t", header=False, index=False)

    # Count how many non-zero TF-IDF terms each cluster has
    for cluster_num in sorted(df["Cluster"].unique()):
        cluster_indices = df[df["Cluster"] == cluster_num].index.to_list()
        cluster_tfidf = X_tfidf[cluster_indices].mean(axis=0)
        cluster_array = np.squeeze(np.asarray(cluster_tfidf))

        nonzero_count = np.count_nonzero(cluster_array)
        print(f"🔹 Cluster {cluster_num} has {nonzero_count} non-zero TF-IDF keywords.")

    # Assign raw embeddings and reduced vectors
    df["Embeddings"] = list(embeddings)  # Raw BERT embeddings
    df["ReducedEmbeddings"] = list(data)  # Reduced vectors from autoencoder
    return df, data, bestScore


def UI(
    df,
    data,
    n_clusters,
    top_n=10,
    min_similarity=0.50,
    title_cluster_k=3,
):
    # User Interface
    query = input("Enter keywords to search: ")
    recommendations = search_similar_resources(
        query=query,
        df=df,
        n_clusters=n_clusters,
        top_n=top_n,
        min_similarity=min_similarity,
        title_cluster_k=title_cluster_k,
        data=data,
    )

    # Call function
    displayResults(recommendations)


if __name__ == "__main__":
    df, data, n_clusters = iteration(
        input_keyword="data/Arxiv_Resources.csv",
        output_keyword="txt/output_keywords.txt",
        SBERT_pretrain="all-MiniLM-L6-v2",
        fig_dir="fig/iter0/",
    )

    UI(
        df=df,
        data=data,
        n_clusters=n_clusters,
        top_n=10,
        min_similarity=0.50,
        title_cluster_k=3,
    )
