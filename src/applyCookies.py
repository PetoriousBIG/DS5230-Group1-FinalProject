import pickle
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

"""
After saving cookies, we need to load it and refresh the page to apply cookies. 
This is one time process. After the cookies is added, you are free to go. 
"""
# Open Google Scholar
driver_path = open("txt/driver_path.txt", "r").read().rstrip()
driver = webdriver.Chrome(service=Service(driver_path))
driver.get("https://scholar.google.com")

# Load saved cookies
cookies = pickle.load(open("cookies.pkl", "rb"))
for cookie in cookies:
    driver.add_cookie(cookie)

# Refresh the page to apply cookies
driver.get("https://scholar.google.com")

print("Cookies loaded! You can now scrape without solving CAPTCHA.")
