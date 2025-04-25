import pickle
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

"""
Reason to do this is because Google Scholar prevents a CAPTCHA challenge to verify user is a robot or human. Using python
to access the description or details of the paper needs to receive the access first. 

Online resources about CAPTCHA: 
https://support.google.com/a/answer/1217728?hl=en#:~:text=CAPTCHA%20(Completely%20Automated%20Public%20Turing,known%20as%20challenge%2Dresponse%20authentication.

Is scraping legal?
https://www.octoparse.com/blog/is-web-crawling-legal-well-it-depends
"""

# Open browser
driver_path = (
    open("driver_path.txt", "r").read().rstrip()
)  # Change this to your actual chromedriver path

service = Service(driver_path)
driver = webdriver.Chrome(service=service)
driver.get("https://scholar.google.com")

# Pause to manually solve CAPTCHA
input("🚀 Solve the CAPTCHA in the browser, then press Enter to continue...")

# Save cookies after solving CAPTCHA
cookies = driver.get_cookies()
pickle.dump(cookies, open("cookies.pkl", "wb"))

print("✅ Cookies saved!")
driver.quit()  # Close browser after saving cookies
