# Capstone Project: Interactive Instagram Dashboard for MINI

## 1. Project Definition

### Project Overview
This project is a web tool for marketers at MINI. 
It takes in Instagram data from MINI and its main competitors, analyses and visualizes the data in various interactive charts.
These help the global Brand Management team at MINI to monitor relevant developments on Instagram.

### Problem Statement
Continuous monitoring, analysis and deriving insights is important in the global steering role of the Brand Management team at MINI.  
The problem is however, that the Instagram has discontinued their official API and as the medium is ever-evolving, it is very resource-intense to keep track in a manual fashion.  
Retrieving relevant data from the platform by accessing Instagram in an automated fashion, such as using typical scraping techniques however violates the platforms' ToS.

### Proposed Solution
The proposed solution involves automating the process as much as (legally) possible, i.e. the analysis and visualization of data.
This will make it much more efficient and less error-prone.  
Hence the original data for setting up the tool (date range from Jan 1, 2021 to Jun 15, 2021) was collected through browsing the relevant profiles on Instagram as a regular logged-in user in Chrome and downloading the resulting .har files from Google's Developer Tools.  
The .har files are then read in, parsed, cleaned and relevant data snippets saved to a database.  
Using this method is also expected to be a more resilient solution than using methods that circumvent security measures by the platform, e.g. using fake IP addresses, VPN, cookies, session ids and the like.  

The data is displayed in a web-based dashboard using Dash and Plotly in the background. The user can make a global date selection through a date picker that automatically updates all outputs.  
Drop downs are available at each visualization to individually select relevant competitors.  

### Metrics
Key metrics to measure the social media performance of the channels are visualized in a dashboard and regard current followers, engagement (likes & comments) and sentiment.  

Based on the date selection following metrics are displayed:
- total posts in date range
- total likes in date range
- total comments in date range
- avg. sentiment in date range (using VaderSentiment compound)
- avg. posts per week (frequency)
- avg. likes per post
- avg. comments per post
- top-performing post regarding likes
- top-performing post regarding comments
- least-performing post regarding likes
- least-performing post regarding comments  
  
The metrics allow the user to see, for example, how well the audience perceives individual posts, whether higher posting frequency leads to better overall performance, etc.

## 2. Analysis

### Data Exploration & Visualization
As mentioned in the project definition, the original dataset (date range from Jan 1, 2021 to Jun 15, 2021) was collected through browsing the relevant profiles on Instagram in numerous sessions and reading in the resulting .har files (total more than 3.5 GB of data) using the process_data.py file.
After extracting and cleaning the relevant data, this results in 391 rows of data related to individual timeline posts of MINI, Fiat and Audi on their global profiles. These rows also contain the average sentiment score of more than 28k comments.
In addition, the current follower numbers are saved in an extra SQL table, as there is no history available.
Additional insights from data exploration are discussed in section 4 Results.
The corresponding data visualizations are available in the interactive dashboard to allow users to easily grasp the development and metrics for the key competitors.

## 3. Methodology

### Data Pre-processing, Implementation and Refinement
The most time-consuming part of the project was retrieving and preparing the data for analysis, as depending on how a user browses Instagram, e.g. just visiting a profile timeline vs. individual posts vs. clicking on "+" button on post to view more comments etc., the data structure in the resulting .har file differs in its composition.  
Hence extracting all relevant entries from the .har file required various techniques, including try & except constructs to check for different encodings (utf-8 vs. base64), regex-expressions to match only relevant patterns and catering for different data structures within the file, e.g. JSON paths differ for timeline vs. individual posts.  
Further pre-processing steps included removing all data that is loaded in the browsing history, but is not from MINI, Fiat or Audi, and also dropping duplicate entries e.g. when same post is visited twice.  
To display the average sentiment of comments on a post, VADER (https://github.com/cjhutto/vaderSentiment) was chosen as an initial solution as it is specifically trained on social media data. The resulting sentiment score is normalized to be between -1 (most extreme negative) and +1 (most extreme positive).
Next step in this project, given more time and resources, is manually scoring retrieved comments and using the resulting data for training a custom algorithm for the automotive industry.  

## 4. Results
The retrieved data for the first half of 2021 show a high variability of how the competitors MINI, Fiat and Audi manage their global Instagram channels.
- The charts on engagement development over time show peaks in post activity and user engagement around the premieres of new vehicles e.g. Jan 27 for announcement of several new MINI models, Feb 9 (Audi RS e-tron GT) and April 14 (Audi Q4 e-tron) for new Audi models, not that pronounced for Fiat (TipoCitySport on April 7).
- MINI was the most active on Instagram, posting significantly more posts (261) than Fiat (44) and Audi (86) and also receiving more likes (4M vs. 2.8M Audi 226k Fiat) and comments (13k vs. 9k Audi and 6k Fiat) overall as a result. Also the sentiment of comments was more positive (.26), however not by much (.18 for Audi and .20 for Fiat). 
- Audi on the other hand is able to get more likes per post (33k vs. 16k MINI, 5k Fiat).
- Fiat posts quite rarely (only 1.9 posts per week vs. 3.6 for Audi and 11 for MINI) and spread unevenly (when they post, they often post several posts on one day). They still achieve quite some engagement, especially comments (135 per post vs. 106 Audi and 52 MINI) also given the much lower number of followers (470k vs. 1.8M MINI and 2.4M Audi)

As a result of this analysis, MINI could try to test limiting the number of posts per week and check whether the engagement per post will rise (similar to Audi). If this is the case, then MINI could save some budget on social media content production and management.

## 5. Conclusion

### Reflection
What started as an idea to optimize and accelerate workflows in social media performance monitoring, ended up in a much bigger project than expected.  
Data collection approaches using Scraping or other automated means turned out as being legally non-compliant.  
Finding a workaround through the parsing .har files finally ended up in working well, but reading in only relevant data was more difficult than originally expected.
It was also important to not process personal data such as user names or profile images of commenting users.

### Opportunities for future improvement
The following aspects would make the dashboard even more insightful, however it would have required additional capacities and legal clearance, which was not possible within the time frame of the capstone project.
- Include images / screenshot of video of posts directly in the dashboard as a preview (extra legal clearance required)
- Include additional social media platforms such as Facebook, LinkedIn, Twitter
- Manually classify comments in dataset and replace VADER with custom algorithm trained on automotive comments on Instagram
- Use NLP or image recognition algorithms to extract more insights from actual contents of the posts and their influence on engagement  
- Host app on Heroku or the like and implement direct upload functionality for new .har files to allow direct online user interaction
All these improvements would result in a broader view of developments or more in-depth understanding.

### Instructions
The project uses the following libraries:  
base64, datetime, dash, dash_core_components, dash_html_components, dash_bootstrap_components, json, numpy, pandas, plotly, re, sqlalchemy, sys, vaderSentiment  
Please check requirements.txt for details.  

1. Run the following command in the project's root directory to read in your .har file (example provided), process it and add cleaned data to the database.  
    `python data/process_data.py data/audi_example.har data/database.db`  
  
2. Run the following command in the project's root directory to run the web app.  
    `flask run`  
    
3. Go to http://127.0.0.1:5000/  

To create your own .har file, please follow these steps:  
- Open Chrome Browser  
- Navigate to the respective profile on Instagram, e.g. https://www.instagram.com/audiofficial/
- Right-click and select "Inspect" (or press Ctrl+Shift+I)
- Navigate to "Network" tab
- Ensure recording is activated as well as "Preserve log" and "Disable cache"
- Browse all posts that you would like to include in dashboard
- Click on the downward-pointing arrow (Export HAR...) in "Network" tab to download .har file and save it in the project folder "data"
- Follow instructions above to read in data and display it in the web app
 
Please note: .har files can accumulate large amounts of data fairly quickly, so it is recommended to repeat this process after a couple of minutes of browsing.

### File Tree
The project consists of the following files:
```
MINI_Dashboard/  
│  
├── assets/  
│   ├── logo.png  # logo for styling the web app
│   ├── style.css  # custom styles for the web app  
│  
├── data/  
│   ├── audi_example.har # exemplary .har file for reading in new data to the dashboard 
│   ├── database.db   # SQLite database to store cleaned posts data  
│   ├── process_data.py   # python file to process .har files 
│  
├── app.py  # python file for running the web app
├── Procfile  # file for running app in production  
├── README.md  # this Readme file  
├── requirements.txt  # file containing all required libraries for installing on web server  
│  
```
### Reference
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

### Acknowledgements
Thanks to BMW for believing in my aspiration and determination to tackle this program as a full-time marketeer with hobbyist interest in programming and sponsoring this Udacity Nanodegree in Data Science. It's been truly worth investing the time.  
Thanks to Udacity for creating such a challenging and intense program. It went way beyond other online video platforms and broadened and deepened my understanding of Data Science at the same time. I've learned a tremendous amount and improved my skills to a whole new level in the process.