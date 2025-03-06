import pickle
from selenium import webdriver

# Open browser
driver = webdriver.Chrome()
driver.get("https://scholar.google.com")

# Pause to manually solve CAPTCHA
input("🚀 Solve the CAPTCHA in the browser, then press Enter to continue...")

# Save cookies after solving CAPTCHA
cookies = driver.get_cookies()
pickle.dump(cookies, open("cookies.pkl", "wb"))

print("✅ Cookies saved! You won’t need to solve CAPTCHA again.")
driver.quit()  # Close browser after saving cookies
