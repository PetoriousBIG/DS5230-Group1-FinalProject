import requests
import pandas as pd
import feedparser  # To parse RSS feeds
from time import sleep

def load_keywords():
    """
    Load keywords from the external keywords.txt file.
    Returns a list of keywords.
    """
    try:
        with open("keywords.txt", "r") as file:
            keywords = [line.strip() for line in file if line.strip()]  # Read and clean up lines
        return keywords
    except FileNotFoundError:
        print("Error: 'keywords.txt' file not found. Make sure it is in the same folder as this script.")
        return []

def fetch_medium_articles(query):
    """
    Fetch articles from Medium using the provided query.
    Returns a list of dictionaries containing article details.
    """
    rss_url = f"https://medium.com/feed/tag/{query.replace(' ', '-')}"
    feed = feedparser.parse(rss_url)
    results = []

    for entry in feed.entries[:10]:  # Fetch top 10 articles per query
        results.append({
            "platform": "Medium",
            "query": query,
            "title": entry.title,
            "url": entry.link
        })
    return results


def main():
    all_articles = []

    # Load keywords from the external file
    keywords = load_keywords()

    if not keywords:
        print("No keywords to search for. Exiting.")
        return

    print("Fetching Medium articles for each keyword...")
    for query in keywords:
        print(f"Fetching articles for keyword: {query}")
        articles = fetch_medium_articles(query)
        all_articles.extend(articles)
        sleep(2)  # Prevent excessive requests

    # Save the results to a CSV file
    if all_articles:
        df = pd.DataFrame(all_articles)
        df.to_csv("medium_articles.csv", index=False)
        print("CSV file saved successfully: medium_articles.csv")
    else:
        print("No articles found.")


if __name__ == "__main__":
    main()
