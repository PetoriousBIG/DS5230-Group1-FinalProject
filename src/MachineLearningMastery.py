import os
import sys
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from keywords import load_keywords

COMMENT_THRESHOLD = 100
INDEX_OF_LAST_PAGE = -2


def search_machine_learning_mastery(query, headers):
    # Search URL format for Machine Learning Mastery
    base_url = "https://machinelearningmastery.com/"
    search_url = f"{base_url}?s={query}"

    # Send a GET request
    response = requests.get(search_url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        articles = []
        number_of_pages = get_number_of_pages(response)
        print(f"{number_of_pages} pages of articles found.")
        for i in range(1, number_of_pages + 1):
            if i % 20 == 0:
                print(f"Processing page {i} of {number_of_pages}.")
            page_url = f"{base_url}?s={query}&paged={i}"
            response = requests.get(page_url, headers=headers)

            if response.status_code == 200:
                articles += scan_page_for_articles(response, query)
            else:
                print(
                    f"Failed to retrieve {page_url}. Status code: {response.status_code}"
                )

        return articles

    else:
        print(f"Failed to retrieve content. Status code: {response.status_code}")
        return None


def get_number_of_pages(response):
    soup = BeautifulSoup(response.content, "html.parser")
    try:
        num_pages = int(
            soup.find_all("a", {"class": "page-numbers"})[
                INDEX_OF_LAST_PAGE
            ].text.replace(",", "")
        )
    except:
        num_pages = 1
    return num_pages


def scan_page_for_articles(page_of_articles, query):
    # Parse the HTML content
    soup = BeautifulSoup(page_of_articles.content, "html.parser")

    # Find relevant articles
    articles = soup.find_all("h2", class_="entry-title")
    results = []
    for article in articles:
        title = article.get_text()
        link = article.a["href"]
        results.append((query, title, link))

    return results


def main(
    input_txt="txt/keywords.txt", output_csv="data/MachineLearningMastery_Resources.csv"
):
    # Set up headers to mimic a real browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }

    # Dataframe set up
    df = pd.DataFrame(columns=["platform", "query", "Title", "URL"])
    # index = 0

    # Get keywords
    keywords = load_keywords(keyword_filepath=input_txt)
    if not keywords:
        print("No keywords with which to search. Exiting.")
        sys.exit()

    # Scrape Machine Learning Mastery
    data = []
    for word in keywords:
        print(f"Searching Machine Learning Mastery for {word}.")
        data += search_machine_learning_mastery(word, headers)

    np_data = np.array(data)
    df = pd.DataFrame(
        {
            "platform": "MachineLearningMastery",
            "query": np_data[:, 0].tolist(),
            "Title": np_data[:, 1].tolist(),
            "URL": np_data[:, 2].tolist(),
        }
    )

    # Output to CSV
    print("Saving data to csv.")
    df.dropna().to_csv(output_csv, index=False)


if __name__ == "__main__":
    main(
        input_txt="txt/keywords.txt",
        output_csv="data/MachineLearningMastery_Resources.csv",
    )
