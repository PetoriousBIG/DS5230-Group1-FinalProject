# Install Required Libraries in VSCode:
# requests: Run `pip install requests` in the terminal

import requests  # For making HTTP requests to the Coursera API
import pandas as pd  # For handling tabular data
import json  # For parsing JSON responses
from time import sleep  # To introduce delays and avoid rate limiting

def fetch_coursera_courses(query="machine learning", max_results=100):
    """
    Fetches course data from Coursera based on a search query.
    """
    api_url = "https://api.coursera.org/api/courses.v1"  # Coursera API endpoint
    params = {
        "q": "search",  # Query parameter to search for courses
        "query": query,  # The search term (e.g., "machine learning")
        "fields": "name,description,partnerIds",  # Fields to retrieve
        "limit": max_results  # Maximum number of results to fetch
    }

    response = requests.get(api_url, params=params)  # Send a GET request to the API

    if response.status_code == 200:  # Check if the request was successful
        data = response.json()  # Parse the JSON response
        courses = data.get("elements", [])  # Extract course elements

        results = []  # List to store processed course data
        for course in courses:
            course_id = course.get("id", "N/A")  # Get course ID (default to "N/A" if missing)
            partner_ids = course.get("partnerIds", [])  # Get partner IDs (list)

            course_info = {
                "query": query,  # Store the search query
                "course_name": course.get("name", "N/A"),  # Course name or "N/A" if missing
                "description": course.get("description", "No description available"),  # Course description
                "difficulty": fetch_course_difficulty(course_id),  # Fetch difficulty level
                "provider_name": fetch_provider_name(partner_ids[0]) if partner_ids else "Unknown",  # Fetch provider name
                "course_link": f"https://www.coursera.org/course/{course_id}"  # Construct course link
            }
            results.append(course_info)  # Add course data to the list

        return results  # Return all fetched course details
    else:
        print(f"Failed to retrieve data: {response.status_code}")  # Print error if request fails
        return []  # Return an empty list in case of failure

def fetch_course_difficulty(course_id):
    """
    Fetches the difficulty level of a course using its ID.
    """
    details_url = f"https://api.coursera.org/api/courses.v1?ids={course_id}&fields=level"  # API endpoint for course details
    response = requests.get(details_url)  # Send request to get difficulty level

    if response.status_code == 200:  # Check if request was successful
        data = response.json()  # Parse the JSON response
        elements = data.get("elements", [{}])  # Extract course elements
        return elements[0].get("level", "Unknown") if elements else "Unknown"  # Return difficulty level or "Unknown"
    
    return "Unknown"  # Return "Unknown" if request fails

def fetch_provider_name(partner_id):
    """
    Fetches the provider (organization/university) name using the partner ID.
    """
    partner_url = f"https://api.coursera.org/api/partners.v1?ids={partner_id}"  # API endpoint for partner details
    response = requests.get(partner_url)  # Send request to get provider name

    if response.status_code == 200:  # Check if request was successful
        data = response.json()  # Parse the JSON response
        elements = data.get("elements", [{}])  # Extract partner elements
        return elements[0].get("name", "Unknown") if elements else "Unknown"  # Return provider name or "Unknown"
    
    return "Unknown"  # Return "Unknown" if request fails

# List of AI/ML-related search queries
queries = [
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "computer vision",
    "reinforcement learning",
    "natural language processing"
]

# Fetch data for multiple queries
all_courses = []  # Initialize an empty list to store all courses
for query in queries:
    print(f"Fetching courses for: {query}")  # Display current search query
    all_courses.extend(fetch_coursera_courses(query))  # Fetch and append course data
    sleep(1)  # Pause for 1 second to avoid rate limiting

# Convert results to DataFrame
df = pd.DataFrame(all_courses)

# Save results to a CSV file
csv_filename = "coursera_ml_ai_data.csv"
df.to_csv(csv_filename, index=False)  # Save DataFrame as CSV without index column

# Print success message with file path
print(f"CSV file saved successfully: {csv_filename}")
