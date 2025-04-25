import requests
import pandas as pd
import feedparser  # For Medium RSS feeds
from bs4 import BeautifulSoup
from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from googleapiclient.discovery import build  # YouTube API

# **API KEYS & SETUP**
YOUTUBE_API_KEY = "your youtube API key" #add your API key
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

# Queries
queries = {
    "youtube": ["machine learning tutorial", "deep learning course"],
    "medium": ["machine learning", "deep learning"],
    "stanford": "https://cs229.stanford.edu/syllabus.html"
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

# **Function: Fetch Medium Articles via RSS Feed**
def fetch_medium_articles(query):
    rss_url = f"https://medium.com/feed/tag/{query.replace(' ', '-')}"
    feed = feedparser.parse(rss_url)
    results = []

    for entry in feed.entries[:10]:  # Fetch 10 articles
        results.append({
            "platform": "Medium",
            "query": query,
            "title": entry.title,
            "url": entry.link
        })
    return results


# **Function: Fetch Stanford AI Course**
def fetch_stanford_course():
    return [{
        "platform": "Stanford",
        "query": "Stanford CS229 Machine Learning",
        "title": "CS229: Machine Learning Course",
        "url": queries["stanford"]
    }]

# **Execute Scraping & API Calls**
all_resources = []

# Fetch YouTube Playlists
print("Fetching YouTube learning playlists...")
for query in queries["youtube"]:
    all_resources.extend(fetch_youtube_playlists(query))
    sleep(2)

# Fetch Medium Articles
print("Fetching Medium AI & ML articles...")
for query in queries["medium"]:
    all_resources.extend(fetch_medium_articles(query))
    sleep(2)

# Fetch Stanford Course
print("Adding Stanford CS229 AI/ML course...")
all_resources.extend(fetch_stanford_course())

# Convert Data to DataFrame and Save to CSV
df = pd.DataFrame(all_resources)
df.to_csv("Youtube_Medium_resources_resources.csv", index=False)

print("CSV file saved successfully: Youtube_Medium_resources.csv")

driver.quit()  # Close Selenium driver
