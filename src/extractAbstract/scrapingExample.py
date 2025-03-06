import requests
import json
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def get_abstract(paper_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }

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
        #options.add_argument("--headless")  # REMOVE this line to see browser actions
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
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
                    EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'Abstracts')]"))
                ).text
            except Exception as e:
                print("ScienceDirect abstract not found:", e)

        # Wiley Online Library
        elif "wiley.com" in paper_url:
            try:
                abstract = wait.until(
                    EC.presence_of_element_located((By.XPATH, "//div[@class='article-section__content en main']/p"))
                ).text
            except Exception as e:
                print("Wiley abstract not found:", e)

        # IEEE Xplore
        elif "ieee.org" in paper_url:
            try:
                abstract = wait.until(
                    EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'abstract-text')]"))
                ).text
            except Exception as e:
                print("IEEE abstract not found:", e)

        driver.quit()
        return abstract

    except Exception as e:
        print("Selenium method failed:", e)
        return None

# Test URLs
urls = [
    "https://arxiv.org/abs/2401.12345",  # arXiv
    "https://www.sciencedirect.com/science/article/abs/pii/009830049390090R",  # ScienceDirect
    "https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/wics.101",  # Wiley
    "https://ieeexplore.ieee.org/document/9895422"  #IEEE
]

for url in urls:
    abstract = get_abstract(url)
    print(f"\n📄 Abstract from {url}:\n{abstract if abstract else 'Not Found'}")

