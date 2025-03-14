import feedparser
import pandas as pd
from datetime import datetime
import os

def fetch_arxiv_data(query, max_results=50):
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query=all:{query.replace(' ', '+')}"
    params = f"&start=0&max_results={max_results}"
    url = base_url + search_query + params

    # Fetch and parse data from arXiv
    feed = feedparser.parse(url)

    # Process retrieved data
    papers = []
    current_year = datetime.now().year
    min_year = current_year - 10  # Only include papers from the last 10 years

    if not feed.entries:
        print(f"No results found for query: {query}")
        return papers  # Return empty list if no results found

    for entry in feed.entries:
        paper_year = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ").year
        if paper_year >= min_year:
            paper = {
                "Title": entry.title,
                "Abstract": entry.summary,
                "Authors": ", ".join([author.name for author in entry.authors]),
                "Year": paper_year,
                "URL": entry.link,
            }
            papers.append(paper)

    return papers

# List of queries to search
queries = [
    "machine learning",
    "deep learning",
    "neural networks",
    "PCA", 
    "computer vision",
    "reinforcement learning"
]

# Fetch data for multiple queries
all_papers = []
for query in queries:
    all_papers.extend(fetch_arxiv_data(query))

# Convert to DataFrame
df = pd.DataFrame(all_papers)

# Define the filename and save path
csv_filename = "Arxiv_Resources.csv"
save_path = os.path.join(os.getcwd(), csv_filename)  

# Save the DataFrame as a CSV file only if it contains data
if not df.empty:
    df.to_csv(save_path, index=False)
    print(f"CSV file saved successfully: {save_path}")
else:
    print("No data found. CSV file was not created.")

# Display first five rows (if there is data)
if not df.empty:
    print(df.head())
