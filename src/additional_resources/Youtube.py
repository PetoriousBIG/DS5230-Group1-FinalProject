import requests
import pandas as pd
from bs4 import BeautifulSoup
from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from googleapiclient.discovery import build  # YouTube API

# **API KEYS & SETUP**
YOUTUBE_API_KEY = "YOU TUBE API KEY" #add your API key
youtube_service = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Selenium Setup (Headless Mode for Efficiency)
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

 
driver_path = "/usr/local/bin/chromedriver"  # Change this to your actual chromedriver path
service = Service(driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

def load_keywords():
    with open("keywords.txt", "r") as file:
        keywords = [line.strip() for line in file if line.strip()]  # Read and clean up lines
    return keywords

# Load keywords from the external file

queries = {
    "youtube": load_keywords()  # Reads all the keywords from the keywords.txt file
}

# **Function: Fetch YouTube Playlists using API**
def fetch_youtube_playlists(query):
    results = []
    response = youtube_service.search().list(
        q=query,
        part="snippet",
        maxResults=10,
        type="playlist"
    ).execute()

    for item in response.get("items", []):
        results.append({
            "platform": "YouTube",
            "query": query,
            "title": item["snippet"]["title"],
            "url": f"https://www.youtube.com/playlist?list={item['id']['playlistId']}"
        })
    return results



# **Execute Scraping & API Calls**
all_resources = []

# Fetch YouTube Playlists
print("Fetching YouTube learning playlists...")
for query in queries["youtube"]:
    all_resources.extend(fetch_youtube_playlists(query))
    sleep(2)

# Convert Data to DataFrame and Save to CSV
df = pd.DataFrame(all_resources)
df.to_csv("Youtube_resources.csv", index=False)

print("CSV file saved successfully: Youtube_resources.csv")

driver.quit()  # Close Selenium driver
