import requests
import pandas as pd
import feedparser  # To parse RSS feeds
from time import sleep
from keywords import load_keywords


def fetch_medium_articles(query):
    """
    Fetch articles from Medium using the provided query.
    Returns a list of dictionaries containing article details.
    """
    rss_url = f"https://medium.com/feed/tag/{query.replace(' ', '-')}"
    feed = feedparser.parse(rss_url)
    results = []

    for entry in feed.entries[:10]:  # Fetch top 10 articles per query
        results.append(
            {
                "platform": "Medium",
                "query": query,
                "Title": entry.title,
                "URL": entry.link,
            }
        )
    return results


def main(keyword_filepath="txt/keywords.txt", output_csv="data/medium_resources.csv"):
    all_articles = []

    # Load keywords from the external file
    keywords = load_keywords(keyword_filepath=keyword_filepath)

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
        df.to_csv(output_csv, index=False)
        print("CSV file saved successfully: medium_articles.csv")
    else:
        print("No articles found.")


if __name__ == "__main__":
    main(keyword_filepath="txt/keywords.txt", output_csv="data/medium_resources.csv")
