from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import os
import pandas as pd
import time


def fetch_edx_courses(keywords):
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in headless mode for efficiency
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    driver_path = '/usr/local/bin/chromedriver'  # Update this to your actual path
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    results = []

    for keyword in keywords:
        print(f"Searching for courses related to: {keyword}")
        try:
            driver.get(f'https://www.edx.org/search?q={keyword}')

            # Wait until the courses are loaded
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='search-card']"))
            )

            courses = driver.find_elements(By.CSS_SELECTOR, "div[data-testid='search-card']")

            for course in courses:
                try:
                    title = course.find_element(By.CSS_SELECTOR, "h3").text
                    description = course.find_element(By.CSS_SELECTOR, "p").text
                    link = course.find_element(By.TAG_NAME, "a").get_attribute("href")

                    results.append({
                        "Keyword": keyword,
                        "Title": title,
                        "Description": description,
                        "URL": link
                    })
                except Exception as e:
                    print(f"Failed to retrieve course details: {e}")

        except Exception as e:
            print(f"Failed to retrieve results for {keyword}: {e}")

    driver.quit()
    return results


def load_keywords():
    with open("keywords.txt", "r") as file:
        return [line.strip() for line in file if line.strip()]


def main():
    keywords = load_keywords()
    if not keywords:
        print("No keywords found in keywords.txt.")
        return

    print('Fetching edX courses...')
    edx_courses = fetch_edx_courses(keywords)

    if edx_courses:
        df = pd.DataFrame(edx_courses)
        output_file = "EdX_Resources.csv"
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    else:
        print('No courses found.')


if __name__ == "__main__":
    main()

