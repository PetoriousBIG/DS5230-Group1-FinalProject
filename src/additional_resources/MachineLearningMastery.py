import os
import sys
import pandas as pd
import requests
from bs4 import BeautifulSoup

COMMENT_THRESHOLD = 100
BASE_URL = 'https://machinelearningmastery.com/blog/'
INDEX_OF_LAST_PAGE= -2

# Set up headers to mimic a real browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

# Dataframe set up
df = pd.DataFrame(columns=['URL', 'Title', 'Text'])
index = 0

# Get number of pages of articles to scan
response = requests.get(BASE_URL, headers=headers)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    last_page = soup.find_all('a', {"class": "page-numbers"})[INDEX_OF_LAST_PAGE].text.replace(',', '')
    try:
        last_page = int(last_page)
    except Exception as e:
        sys.exit(f"Error converting {last_page} to integer. ", e)

else:
    sys.exit(f"Failed to retrieve number of pages. Status code: {response.status_code}")

print(f"Scanning through {last_page} pages of articles...")

# For every page, check if each article meets the engagement threshold and if so, save its contents to dataframe.
for page_num in range(1, last_page+1):
    print(f"Starting scan of page {page_num}...")
    page_url = f'https://machinelearningmastery.com/blog/page/{page_num}/'
    page_response = requests.get(page_url, headers=headers)
    
    if page_response.status_code == 200:
        page_soup = BeautifulSoup(page_response.text, 'html.parser')
        articles = page_soup.find_all('article')
        
        for article in articles:
            num_comments = int(article.find('span', {'class': 'post-comments'}).text.replace(',', ''))

            if num_comments < COMMENT_THRESHOLD:
                continue

            article_title = article.find('h2', {'class': 'title entry-title'}).text
            article_url = article.find('a', href=True)['href']
            article_text = ''
            article_response = requests.get(article_url, headers=headers)
            if article_response.status_code == 200:
                article_soup = BeautifulSoup(article_response.text, 'html.parser')
                paragraphs = article_soup.find('section', {'class': 'entry'}).find_all('p')
                for paragraph in paragraphs:
                    article_text += paragraph.text
                
                df.loc[index] = [article_url, article_title, article_text]
                index += 1

            else:
                print(f"Failed to retrieve article {article_title}. Status code: {article_response.status_code}")

    else:
        print(f"Failed to retrieve page {page_num}. Status code: {page_response.status_code}")

# Output to CSV
print("Saving data to csv.")
df.to_csv('MachineLearningMastery.csv', index=False)
