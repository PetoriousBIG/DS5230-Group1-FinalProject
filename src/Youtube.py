import requests
import pandas as pd
from bs4 import BeautifulSoup
from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from googleapiclient.discovery import build  # YouTube API
from keywords import load_keywords
import configparser


# **Function: Fetch YouTube Playlists using API**
def fetch_youtube_playlists(query, youtube_service):
    results = []
    response = (
        youtube_service.search()
        .list(q=query, part="snippet", maxResults=10, type="playlist")
        .execute()
    )

    for item in response.get("items", []):
        results.append(
            {
                "platform": "YouTube",
                "query": query,
                "Title": item["snippet"]["title"],
                "URL": f"https://www.youtube.com/playlist?list={item['id']['playlistId']}",
            }
        )
    return results


def main(keyword_filepath="txt/keywords.txt", output_csv="data/Youtube_resources.csv"):

    # Load keywords from the external file
    queries = {
        "youtube": load_keywords(
            keyword_filepath=keyword_filepath
        )  # Reads all the keywords from the keywords.txt file
    }

    # Load API_key
    config = configparser.ConfigParser()
    config.read("API_key.ini")

    # Set driver path
    driver_path = open("txt/driver_path.txt", "r").read().rstrip()

    # **API KEYS & SETUP**
    YOUTUBE_API_KEY = config["Youtube"]["API_key"]
    youtube_service = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    # Selenium Setup (Headless Mode for Efficiency)
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # **Execute Scraping & API Calls**
    all_resources = []

    # Fetch YouTube Playlists
    print("Fetching YouTube learning playlists...")
    for query in queries["youtube"]:
        print(f"Fetching articles for keyword: {query}")
        all_resources.extend(
            fetch_youtube_playlists(query, youtube_service=youtube_service)
        )
        sleep(2)

    # Convert Data to DataFrame and Save to CSV
    df = pd.DataFrame(all_resources)
    df.to_csv(output_csv, index=False)

    print("CSV file saved successfully: Youtube_resources.csv")

    driver.quit()  # Close Selenium driver


if __name__ == "__main__":
    main(keyword_filepath="txt/keywords.txt", output_csv="data/Youtube_resources.csv")
