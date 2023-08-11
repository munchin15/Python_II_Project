import pandas as pd
import requests

#### GET DATA #################################################################

#### SerpAPI ##################################################################
# Im using SerpAPI to get data from google jobs. Stored in csv file for easy access
# The API is limited to 100 requests per month for the free plan, as the project was over the span of ~2 months, I manages to get 173 observations for the analysis
# This is by far not enough to get great resuts, but I could not get more data this month. I have to keep adding data every month from now on to make the results more meaningful
# The following code is also not perfect. If no more jobs available are avalable for a certain keyword, it still keeps requesting up to 100 results, using up all the available capacities
# But as I don't have more requests left anyway, it was not worth to improve this code (:
    
# Define Job Listing request inside function for better use later
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
                all_jobs_results.extend(jobs_results)
            else:
                print(f"No job results found for {query} in {location}")
        else:
            print(f"Request for {query} in {location} failed with status code:", response.status_code)

    # Store all the jobs in one dataframe and then save it to a csv to use later on 
    df = pd.DataFrame(all_jobs_results)
    file_name = f"{query.replace(' ', '_')}_{location.replace(', ', '_')}.csv"
    df.to_csv(f'D:/Organisation/Uni/Wien/PFF II/Project/{file_name}', index=False)
    print(f"Data for {query} in {location} saved as {file_name}")

###############################################################################

#### Get data #################################################################
# Use private API key
api_key = '757748fdc785c6292d45826b8f79e99d8ee094d6887e4befcf6d44a233dff4e0'

# For now I have concentrated on Data related job postings. Although the analysis would work for any field
queries = ['Data Analyst', 'Data Scientist', 'Data Engineer', 'Data Consultant', 'Business Analyst', 'Business Intelligence Analyst', 'Machine Learning Engineer', 'Data Visualization Specialist', 'AI Analyst']

# If I would have had more requests left, I would also look at other places to increase the data size
locations = ['Vienna, Austria']

# Run function for every job title and location
for query in queries:
    for location in locations:
        get_job_listings(query, location, api_key)
        
###############################################################################
        
       