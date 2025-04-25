import requests  # For making API requests
import pandas as pd  #
import time  # For handling API rate limits
import os  # For accessing environment variables
from datetime import datetime  # For handling date operations
from langdetect import (
    detect,
    LangDetectException,
)  # detect = language identifier, LangDetectException = error if detection fails
import configparser
from keywords import load_keywords

# Load API_key
config = configparser.ConfigParser()
config.read("API_key.ini")

# Load API key from an environment variable for security
SEMANTIC_SCHOLAR_API_KEY = config["GoogleScholar"]["API_key"]
# os.getenv("SEMANTIC_SCHOLAR_API_KEY", "Inset API Key")

# Get the current year and define the past 20 years range
CURRENT_YEAR = datetime.now().year  # Get the current year
START_YEAR = CURRENT_YEAR - 20  # Define the start year as 20 years ago


# Function to fetch AI/ML research papers from Semantic Scholar with pagination
def fetch_research_papers(
    query, max_results=100
):  # Allows fetching up to 100 results per query
    url = "https://api.semanticscholar.org/graph/v1/paper/search"  # API endpoint for research papers
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}  # API authentication header

    papers = []  # List to store fetched research papers
    total_fetched = 0  # Counter to track the total number of papers fetched
    offset = 0  # Track pagination to avoid duplicate results

    # Loop until we fetch the required number of results
    while total_fetched < max_results:
        params = {
            "query": query,  # Search query term
            "limit": min(
                100, max_results - total_fetched
            ),  # Fetch up to 100 at a time (Predefined by Google)
            "offset": offset,  # Pagination offset to fetch next batch of results
            "fields": "title,abstract,authors,year,url,citationCount,journal,venue,publicationTypes",  # Lables of dataset
            "year": f"{START_YEAR}-{CURRENT_YEAR}",  # Fetch only from the last 20 years
        }

        response = requests.get(url, headers=headers, params=params)  # Make API request

        # Handle API rate limits and errors with exponential backoff
        if response.status_code == 429:  # If API rate limit is exceeded
            wait_time = 2  # Initial wait time in seconds
            while response.status_code == 429:  # Keep retrying until allowed
                print(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
                time.sleep(wait_time)  # Wait before retrying
                wait_time *= 2  # Increase wait time (exponential backoff)
                response = requests.get(
                    url, headers=headers, params=params
                )  # Retry API request

        elif response.status_code != 200:  # Handle other errors
            print(f"Error fetching data: {response.status_code} - {response.text}")
            break  # Stop execution on API error

        data = response.json()  # Convert API response to JSON format
        papers_fetched = data.get("data", [])  # Extract paper data from response

        # If no more results, stop fetching
        if not papers_fetched:
            print(f"No more results for query: {query}")
            break

        for paper in papers_fetched:
            year = paper.get("year", 0)
            if START_YEAR <= year <= CURRENT_YEAR:
                title = paper.get("title", "")
                abstract = paper.get("abstract", "")

            try:
                if detect(title) != "en":
                    continue
                if abstract and detect(abstract) != "en":
                    continue
            except LangDetectException:
                continue  # skip if language detection fails

            papers.append(
                {
                    "Title": title or "N/A",
                    "Abstract": abstract or "N/A",
                    "Authors": ", ".join(
                        [author["name"] for author in paper.get("authors", [])]
                    ),
                    "Year": year,
                    "URL": paper.get("url", "N/A"),
                    "Citations": paper.get("citationCount", "N/A"),
                    "Journal": (paper.get("journal") or {}).get("name", "N/A"),
                    "Venue": paper.get("venue", "N/A"),
                    "Publication Types": ", ".join(
                        paper.get("publicationTypes", []) or []
                    ),
                }
            )

        total_fetched += len(papers_fetched)  # Update the total fetched count
        offset += len(papers_fetched)  # Move the offset forward for pagination

        # Respect API rate limits by adding a small delay
        time.sleep(1)

    return papers  # Return the list of fetched papers


def main(input_txt="txt/keywords.txt", output_csv="data/GoogleScholar_Resources.csv"):
    # List of queries to fetch AI & ML research papers
    # queries = pd.read_csv("txt/keywords.txt")
    queries = load_keywords(keyword_filepath=input_txt)
    all_papers = []  # List to store papers from all queries

    # Loop through each query and fetch research papers
    for query in queries:
        print(f"Fetching up to 100 papers for: {query}")
        papers = fetch_research_papers(
            query, max_results=100
        )  # Fetch papers for the given query
        all_papers.extend(papers)  # Append fetched papers to the main list

    # Convert fetched data to a Pandas DataFrame
    df = pd.DataFrame(all_papers)

    # Save data as a CSV file
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main(input_txt="txt/keywords.txt", output_csv="data/GoogleScholar_Resources.csv")
