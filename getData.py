import pandas as pd
import requests

#GET DATA:
#    Use API to get data from google jobs. Stored in csv file for easy acces, here command around it so its not run everytime the programm is run
## SerpAPI Job Listing parameters

#### Not perfect. If no more jobs available, it still keeps requesting until 100 results, using up all the available capacities
def get_job_listings(query, location, api_key, total_results=100, results_per_page=10):
    num_pages = total_results // results_per_page
    all_jobs_results = []

    for page in range(num_pages):
        start = page * results_per_page
        url = f"https://serpapi.com/search.json?q={query}&engine=google_jobs&hl=en&api_key={api_key}&start={start}&location={location}"

        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            jobs_results = data.get("jobs_results")

            # Check if jobs_results is not None before extending the list
            if jobs_results is not None:
                all_jobs_results.extend(jobs_results)  # Append the current page's job results to the list
            else:
                print(f"No job results found for {query} in {location}")
        else:
            print(f"Request for {query} in {location} failed with status code:", response.status_code)

    df = pd.DataFrame(all_jobs_results)
    file_name = f"{query.replace(' ', '_')}_{location.replace(', ', '_')}.csv"
    df.to_csv(f'D:/Organisation/Uni/Wien/PFF II/Project/{file_name}', index=False)
    print(f"Data for {query} in {location} saved as {file_name}")

## Get data
api_key = '757748fdc785c6292d45826b8f79e99d8ee094d6887e4befcf6d44a233dff4e0'

queries = ['Data Analyst', 'Data Scientist', 'Data Engineer', 'Data Consultant', 'Business Analyst', 'Business Intelligence Analyst', 'Machine Learning Engineer', 'Data Visualization Specialist', 'AI Analyst']
locations = ['Vienna, Austria']

for query in queries:
    for location in locations:
        get_job_listings(query, location, api_key)
        
        
       