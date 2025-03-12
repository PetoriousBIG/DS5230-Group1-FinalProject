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


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download("punkt_tab")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stop_word_add = [
    "abstract",
    ".",
    ":",
    "'",
    '"',
    ",",
    "%",
    "&",
]
stop_words.update(stop_word_add)
print(stop_words)


# Set up Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run in headless mode (no browser window)
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()), options=options
)


def get_google_scholar_results(query, num_pages=1):
    results = []
    for i in range(num_pages):
        start_index = int(10 * num_pages)
        search_url = f"https://scholar.google.com/scholar?hl=en&q={query.replace(' ', '+')}&start={start_index}"
        driver.get(search_url)
        time.sleep(5)  # Wait for the page to load

        # Parse page source with Beautifu   lSoup
        soup = BeautifulSoup(driver.page_source, "html.parser")
        papers = soup.find_all("div", class_="gs_ri")

        for paper in papers:
            title_tag = paper.find("h3", class_="gs_rt")
            desc_tag = paper.find("div", class_="gs_rs")  # The "description" field
            link_tag = title_tag.find("a") if title_tag else None  # Get the paper link

            title = title_tag.text if title_tag else "No title available"
            description = desc_tag.text if desc_tag else "No description available"
            link = link_tag["href"] if link_tag else "No link available"

            results.append({"title": title, "description": description, "link": link})

    return results


def get_abstract(paper_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }

    # Try requests first for static HTML
    try:
        response = requests.get(paper_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # arXiv abstracts
        if "arxiv.org" in paper_url:
            abstract_tag = soup.find("blockquote", class_="abstract")
            if abstract_tag:
                return abstract_tag.text.strip()

        # JSON metadata extraction
        script_tag = soup.find("script", type="application/ld+json")
        if script_tag:
            metadata = json.loads(script_tag.string)
            if "abstract" in metadata:
                return metadata["abstract"]

    except Exception as e:
        print("Requests method failed:", e)

    # Use Selenium for dynamic pages
    try:
        options = webdriver.ChromeOptions()
        # REMOVE HEADLESS MODE FOR DEBUGGING
        # options.add_argument("--headless")  # REMOVE this line to see browser actions
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )
        driver.get(paper_url)

        wait = WebDriverWait(driver, 15)  # Increased wait time

        # Scroll to the bottom to trigger JavaScript
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)  # Wait for JS content to load

        abstract = None

        # ScienceDirect
        if "sciencedirect.com" in paper_url:
            try:
                abstract = wait.until(
                    EC.presence_of_element_located(
                        (By.XPATH, "//div[contains(@class, 'Abstracts')]")
                    )
                ).text
            except Exception as e:
                print("ScienceDirect abstract not found:", e)

        # Wiley Online Library
        elif "wiley.com" in paper_url:
            try:
                abstract = wait.until(
                    EC.presence_of_element_located(
                        (By.XPATH, "//div[@class='article-section__content en main']/p")
                    )
                ).text
            except Exception as e:
                print("Wiley abstract not found:", e)

        # IEEE Xplore
        elif "ieee.org" in paper_url:
            try:
                abstract = wait.until(
                    EC.presence_of_element_located(
                        (By.XPATH, "//div[contains(@class, 'abstract-text')]")
                    )
                ).text
            except Exception as e:
                print("IEEE abstract not found:", e)

        driver.quit()
        return abstract

    except Exception as e:
        print("Selenium method failed:", e)
        return None


print("Start.")
# Example search query
query = "pca"
papers = get_google_scholar_results(query)

print("number of resources: ", len(papers))

abstract_total = []
for paper in papers:
    url = paper["link"]
    abstract = get_abstract(url)
    if abstract is not None:
        abstract_total.append(abstract)
        print(abstract.lower())

        abstract_cleaned = re.sub(r"\([^)]*\)", "", abstract.lower())

        word_tokens = word_tokenize(abstract_cleaned)
        filtered_sentence = []

        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)

        print(filtered_sentence)

# Close the driver
driver.quit()
