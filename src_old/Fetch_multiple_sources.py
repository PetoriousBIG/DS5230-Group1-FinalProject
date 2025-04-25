# Install Required Libraries in VSCode:
# requests: Run `pip install requests`
# beautifulsoup4: Run `pip install beautifulsoup4`

import requests  # For making HTTP requests
import pandas as pd  
from bs4 import BeautifulSoup  # For parsing HTML content
from time import sleep  # To introduce delays and avoid request blocking

# Define Queries for Each Source

# YouTube AI & ML Tutorials
youtube_queries = [
    "machine learning tutorial",
    "deep learning course",
    "artificial intelligence full course",
    "computer vision tutorial",
    "reinforcement learning tutorial",
    "natural language processing tutorial"
]

# Medium AI & ML Articles
medium_queries = [
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "computer vision",
    "reinforcement learning",
    "natural language processing",
    "data science",
    "neural networks"
]

# MIT OpenCourseWare & University Courses
university_queries = [
    "machine learning courses",
    "deep learning MIT course",
    "AI courses Harvard",
    "computer vision course Stanford",
    "NLP course MIT",
    "reinforcement learning Berkeley course"
]

# Stanford AI Course (Static Link)
stanford_cs229_query = ["Stanford CS229 Machine Learning"]


# Function to Scrape YouTube Playlists
def scrape_youtube_playlists(query):
    """
    Scrapes YouTube search results for AI & ML playlists dynamically.
    NOTE: YouTube search results are JavaScript-based, so BeautifulSoup alone may not work well.
    """
    youtube_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
    response = requests.get(youtube_url, headers={"User-Agent": "Mozilla/5.0"})  # Mimic browser request
    soup = BeautifulSoup(response.text, "html.parser")

    videos = soup.find_all("a", {"id": "video-title"})  # Find video links
    results = []

    for video in videos[:15]:  # Attempt to fetch 15 results per query
        video_title = video.text.strip()
        video_url = f"https://www.youtube.com{video['href']}"
        results.append({
            "platform": "YouTube",
            "query": query,
            "title": video_title,
            "url": video_url
        })

    return results


# Function to Scrape Medium Articles
def scrape_medium_articles(query):
    """
    Scrapes Medium for recent AI & ML-related articles.
    """
    medium_url = f"https://medium.com/tag/{query.replace(' ', '-')}"
    response = requests.get(medium_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")

    articles = soup.find_all("h2")  # Medium article titles are usually in <h2> tags
    results = []

    for article in articles[:15]:  # Fetch 15 articles per query
        link = article.find_parent("a")  # Get parent <a> tag for the link
        if link and "href" in link.attrs:
            article_title = article.text.strip()
            article_url = link["href"] if link["href"].startswith("http") else f"https://medium.com{link['href']}"
            results.append({
                "platform": "Medium",
                "query": query,
                "title": article_title,
                "url": article_url
            })

    return results


# Function to Scrape MIT OpenCourseWare
def scrape_opencourseware():
    """
    Scrapes MIT OpenCourseWare for AI & ML courses.
    """
    ocw_url = "https://ocw.mit.edu/courses/find-by-topic/#cat=engineering&subcat=computerlearning"
    response = requests.get(ocw_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")

    courses = soup.find_all("h4", class_="course-title")  # Find all course titles
    results = []

    for course in courses:  # Fetch all available courses
        link = course.find("a")  # Get course link
        if link and "href" in link.attrs:
            course_title = course.text.strip()
            course_url = f"https://ocw.mit.edu{link['href']}"
            results.append({
                "platform": "MIT OpenCourseWare",
                "query": "MIT AI/ML Courses",
                "title": course_title,
                "url": course_url
            })

    return results


# Function to Fetch Stanford AI Course 
def fetch_stanford_ai_course():
    """
    Adds the Stanford CS229 AI/ML course without scraping.
    """
    return [{
        "platform": "Stanford",
        "query": "Stanford CS229 Machine Learning",
        "title": "CS229: Machine Learning Course",
        "url": "https://cs229.stanford.edu/syllabus.html"
    }]


# Fetch Data from All Sources
all_resources = []

# Scrape YouTube Playlists
print("Fetching YouTube learning playlists...")
for query in youtube_queries:
    all_resources.extend(scrape_youtube_playlists(query))
    sleep(2)  # Prevent excessive requests

# Scrape Medium Articles
print("Fetching Medium AI & ML articles...")
for query in medium_queries:
    all_resources.extend(scrape_medium_articles(query))
    sleep(2)

# Scrape MIT OpenCourseWare Courses
print("Fetching MIT OpenCourseWare AI/ML courses...")
all_resources.extend(scrape_opencourseware())
sleep(2)

# Fetch Stanford CS229 AI/ML Course
print("Adding Stanford CS229 AI/ML course...")
all_resources.extend(fetch_stanford_ai_course())

# Convert Data to DataFrame and Save to CSV
df = pd.DataFrame(all_resources)

# Save results to a CSV file
csv_filename = "web_resources.csv"
df.to_csv(csv_filename, index=False)

# Print confirmation message
print(f"CSV file saved successfully: {csv_filename}")
