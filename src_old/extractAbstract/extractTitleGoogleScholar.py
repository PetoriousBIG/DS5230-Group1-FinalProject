from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import requests
import json
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

# Set up Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run in headless mode (no browser window)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def clean_title(title_tag):
    if title_tag:
        full_title = title_tag.text.strip()
        # Remove ALL starting tags like [PDF], [HTML] (can be repeated)
        while full_title.startswith('[') and ']' in full_title:
            full_title = full_title.split(']', 1)[1].strip()
        return full_title
    return "No title available"

def get_title(query, num_pages=1):
    results = []
    for i in range(num_pages):
        start_index = int(10 * num_pages)
        search_url = f"https://scholar.google.com/scholar?hl=en&q={query.replace(' ', '+')}&start={start_index}"
        driver.get(search_url)
        #time.sleep(1)  # Wait for the page to load

        # Parse page source with Beautifu   lSoup
        soup = BeautifulSoup(driver.page_source, "html.parser")
        papers = soup.find_all("div", class_="gs_ri")

        for paper in papers:
            title_tag = paper.find("h3", class_="gs_rt")
            desc_tag = paper.find("div", class_="gs_rs")  # The "description" field
            link_tag = title_tag.find("a") if title_tag else None  # Get the paper link

            title = title_tag.text if title_tag else "No title available"
            title = clean_title(title_tag)
            description = desc_tag.text if desc_tag else "No description available"
            link = link_tag["href"] if link_tag else "No link available"

            results.append(title)

    return results


query = "dark matter"
titles = get_title(query, num_pages=3)
print(titles)
print(len(titles))