# Capstone Project: Interactive Instagram Dashboard for MINI

## 1. Project Definition

### Project Overview
This project is a web tool for marketers at MINI. 
It takes in Instagram data from MINI and its main competitors, analyses (using an automotive-specific sentiment 
algorithm) and visualizes the data in various interactive charts.
These help the global Brand Management team at MINI to monitor relevant developments on Instagram.

### Problem Statement
Continuous monitoring, analysis and deriving insights from social media platforms is important in the global steering 
role of the Brand Management team at MINI.  
The problem is however that Instagram has discontinued their official API and as the medium is ever-evolving, it is 
very resource-intense to keep track in a manual fashion.  
Retrieving relevant data from the platform by accessing Instagram in an automated fashion, such as using typical 
scraping techniques however violates the platforms' ToS.
Also, understanding sentiment from the community through the comments is of key interest, but there are no satisfactory 
standard solutions due to multi-lingual text and specificity of automotive comments.

### Proposed Solution
The proposed solution involves automating the process as much as (legally) possible, i.e. the analysis and 
visualization of data.
This will make it much more efficient and less error-prone.  
Hence the original data for setting up the tool (date range from Jan 1, 2021 to Jun 15, 2021) was collected through 
browsing the relevant profiles on Instagram as a regular logged-in user in Chrome and downloading the resulting .har 
files from Google's Developer Tools.  
The .har files are then read in, parsed, cleaned and relevant data snippets saved to a database.  
Using this method is also expected to be a more resilient solution than using methods that circumvent security measures 
by the platform, e.g. using fake IP addresses, VPN, cookies, session ids and the like.  

In order to obtain insights on the sentiment, the collected comments data was manually scored regarding polarity 
(positive, neutral, negative) and different approaches were tested to do this in an automated fashion.

The data is displayed in a web-based dashboard using Dash and Plotly in the background. The user can make a global date 
selection through a date picker that automatically updates all outputs.  
Drop downs are available at each visualization to individually select relevant competitors.  

![Screenshot 1](screenshot_1.jpg)

### Metrics
In order to satisfy the requirements of the Capstone project, the sentiment analysis piece was selected as a 
multi-class classification problem to apply a custom-trained machine learning algorithm.
Accuracy of correctly classifying the sentiment of a comment was selected as the primary metric to evaluate 
different approaches.

Additional features to measure the social media performance of the automotive profiles are visualized in a dashboard 
and regard current followers, engagement (likes & comments) and sentiment.  

Besides the metrics related to the sentiment analysis, the following information is displayed to the end-user of the 
dashboard:
- total posts, likes, and comments in date range 
- avg. sentiment in date range (using VaderSentiment compound)
- avg. posts per week (frequency)
- avg. likes and comments per post
- top-performing post regarding likes or comments
- least-performing post regarding likes or comments

These allow the user to see, for example, how well the audience perceives individual posts, whether higher posting 
frequency leads to better overall performance, etc.

## 2. Analysis

### Data Exploration & Visualization
As mentioned in the project definition, the original dataset (date range from Jan 1, 2021 to Jun 15, 2021) was 
collected through browsing the relevant profiles on Instagram in numerous sessions and reading in the resulting .har 
files (total more than 3.5 GB of data) using the process_data.py file.
After extracting and cleaning the relevant data, this results in 391 rows of data related to individual timeline posts 
of MINI, Fiat and Audi on their global profiles. These rows also contain the sentiment classification of more than 10k 
comments.
In addition, the current follower numbers are saved in an extra SQL table, as there is no history available through 
Instagram directly.
Additional insights from data exploration are discussed in section 4: Results.
The corresponding data visualizations are available in the interactive dashboard to allow users to easily grasp the 
development and metrics for the key competitors.

In order to explore the sentiment of the comments, an additional Jupyter Notebook (EDA.ipynb) is available. It shows 
that the standard VADER algorithm achieved an accuracy of 65.3% of the manually scored comments. 

## 3. Methodology

### Data Pre-processing, Implementation and Refinement
Initially, the most time-consuming part of the project was retrieving and preparing the data for analysis, as depending 
on how a user browses Instagram, e.g. just visiting a profile timeline vs. individual posts vs. clicking on "+" button 
on post to view more comments etc., the data structure in the resulting .har file differs in its composition.  
Hence extracting all relevant entries from the .har file required various techniques, including try & except constructs 
to check for different encodings (utf-8 vs. base64), regex-expressions to match only relevant patterns and catering for 
different data structures within the file, e.g. JSON paths differ for timeline vs. individual posts.  
Further pre-processing steps included removing all data that is loaded in the browsing history, but is not from MINI, 
Fiat or Audi, and also dropping duplicate entries e.g. when same post is visited twice.  
To display the average sentiment of comments on a post, VADER (https://github.com/cjhutto/vaderSentiment) was chosen as 
an initial solution as it is specifically trained on social media data. The resulting sentiment score is normalized to 
be between -1 (most extreme negative) and +1 (most extreme positive).

After receiving initial reviewer feedback, the next step in this project was to manually score retrieved comments (6.2k 
deduplicated entries) and using the resulting data for training a custom algorithm for the automotive industry.  
Checking the classifications by VADER against the manual labels showed that VADER was only able to classify 65.3% 
correctly.  

After an exploratory data analysis (see EDA.ipynb), some issues of the data appeared. Besides being limited in size, the
dataset also is very imbalanced, with 73% positive comments, 19% neutral and only 8% negative.  
The manually scored comments were then pre-processed for Machine Learning. Comments were pre-processed by translating
all comments into English, replacing irrelevant urls, normalizing, tokenizing and removing stopwords and lemmatization.
The cleaned tokens were then used to train a classifier using a machine learning pipeline involving a CountVectorizer, 
a TfidfTransformer and initially a weighted DecisionTreeClassifier to better balance the classes.
The results were refined using GridSearch for optimizing the hyperparameters and validated using cross-validation.
This process helped to get the overall (mean) accuracy to 72.4%.
As a refinement, a weighted RandomForestClassifier was implemented and here, GridSearch helped to get the overall (mean)
accuracy to 72.9%. 

The model however is still very highly biased towards the predominant 'positive' class.
Hence with more time on hand, acquiring more training data by additional manual rating is required to provide better
learning possibilities for the model.

## 4. Results

Initially the project did not entail a model, but this has now been included based on reviewer feedback.

### Model Evaluation, Validation & Justification
The project now uses a custom trained algorithm based on a dataset of 6.2k manually scored comments.
In order to classify new comments regarding polarity (positive, neutral, negative) a machine learning pipeline 
including a CountVectorizer, a TfidfTransformer and a weighted RandomForestClassifier is used.
The results were refined using GridSearch for optimizing the hyperparameters.

The best results were achieved by using the following hyperparameters:
- min samples before node is split: 2
- max features considered for splitting a node: auto
- n estimators, i.e. trees in the forest = 10
- max depth, i.e. maximum depth of the tree = 3

By using this model and the discussed parameters the accuracy of sentiment classification was improved to 72.9%, 
compared to 65.3% of the VADER algorithm. However to make the model more robust increasing the training dataset by 
manually scoring more comments would be beneficial. This may also then enable Deep Learning approaches.

### Additional Findings
The retrieved data for the first half of 2021 show a high variability of how the competitors MINI, Fiat and Audi manage 
their global Instagram channels.
- The manual rating of almost 6.2k unique comments resulted in 73% of all comments as positive, 19% as neutral 
and only 8% as negative. This shows the overall tendency of positive sentiment on the automotive channels.
- The charts on engagement development over time show peaks in post activity and user engagement around the premieres 
of new vehicles e.g. Jan 27 for announcement of several new MINI models, Feb 9 (Audi RS e-tron GT) and April 14 
(Audi Q4 e-tron) for new Audi models, not that pronounced for Fiat (TipoCitySport on April 7).
- MINI was the most active on Instagram, posting significantly more posts (261) than Fiat (44) and Audi (86) and also 
receiving more likes (4.1M vs. 2.9M Audi 226k Fiat) and comments (13k vs. 9k Audi and 6k Fiat) overall as a result. 
TODO ADAPT:
Also 
the sentiment distribution of comments was better with 84% positive comments, but not by much (81% for Audi and 83% for Fiat). 
- Audi on the other hand is able to get more likes per post (33k vs. 16k MINI, 5k Fiat).
- Fiat posts quite rarely (only 1.9 posts per week vs. 3.6 for Audi and 11.1 for MINI) and spread unevenly (when they 
post, they often post several posts on one day). They still achieve quite some engagement, especially comments (135 per 
post vs. 106 Audi and 52 MINI) also given the much lower number of followers (470k vs. 1.8M MINI and 2.4M Audi)

As a result of this analysis, MINI could try to test limiting the number of posts per week and check whether the 
engagement per post will rise (similar to Audi). If this is the case, then MINI could save some budget on social media 
content production and management.

## 5. Conclusion

### Reflection
What started as an idea to optimize and accelerate workflows in social media performance monitoring, ended up in a much 
bigger project than expected.  
Data collection approaches using scraping or other automated means turned out as being legally non-compliant.  
Finding a workaround through the parsing .har files finally ended up in working well, but reading in only relevant data 
was more difficult than originally expected.
It was also important to not process personal data such as user names or profile images of commenting users.
After the initial submission, the reviewer feedback led to a manual scoring of comments data regarding sentiment to 
train a custom algorithm that can classify comments accordingly and not simply rely on VADER.
While the custom model improved the accuracy, it is still rather low with 72.9% and not very robust as the dataset was 
small and imbalanced.

### Opportunities for future improvement
The following aspects would make the dashboard even more insightful, however it would have required additional 
capacities and legal clearance, which was not possible within the time frame of the capstone project.
- First and foremost: Rating more comments to improve machine learning model and/or enable deep learning
- Include images / screenshot of video of posts directly in the dashboard as a preview (extra legal clearance required)
- Include additional social media platforms such as Facebook, LinkedIn, Twitter
- Use image recognition algorithms to extract more insights from actual contents of the posts and their 
influence on engagement  
- Host app on Heroku or the like and implement direct upload functionality for new .har files to allow direct online 
user interaction
All these improvements would result in a broader view of developments or more in-depth understanding.

## 6. Instructions
The project uses the following libraries:
base64, datetime, dash, dash_core_components, dash_html_components, dash_bootstrap_components, joblib, json, nltk, 
numpy, pandas, pickle, plotly, re, sklearn, sqlalchemy, sys, vaderSentiment  
Please check requirements.txt for details.  

1. Run the following command in the project's root directory to read in your .har file (example provided), process it 
and add cleaned data to the database.  
    `python data/parse_data.py data/audi_example.har data/database.db models/model.pkl`  
  
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
- Click on the downward-pointing arrow (Export HAR...) in "Network" tab to download .har file and save it in the 
project folder "data"
- Follow instructions above to read in data and display it in the web app
 
Please note: .har files can accumulate large amounts of data fairly quickly, so it is recommended to repeat this 
process after a couple of minutes of browsing.

## 7. File Tree
The project consists of the following files:
```
Capstone [MINI_Dashboard]/  
│  
├── assets/  
│   ├── logo.png  # logo for styling the web app
│   ├── style.css  # custom styles for the web app  
│  
├── data/  
│   ├── audi_example.har  # exemplary .har file for reading in new data to the dashboard 
│   ├── database.db   # SQLite database to store cleaned posts data  
│   ├── parse_data.py   # python file to process .har files 
│  
├── models/  
│   ├── comments_for_rating.csv  # initial file to create training and testing data for the classifier
│   ├── comments_rated.csv  # csv file incl. all labels for training and testing the classifier
│   ├── model.pkl  # pickle file containing trained model in binary ###(saved in Git LFS)  
│   ├── process_comments.py  # python file to load data from database to create comments_for_rating.csv
│   ├── train_classifier.py  # python file to train classifier model 
│   ├── train_classifier_results.txt  # txt file with training results and classification report
│  
├── app.py  # python file for running the web app
├── EDA.ipynb  # Jupyter notebook containing Exploratory Data Analysis of comments
├── Procfile  # file for running app in production  
├── README.md  # this Readme file  
├── requirements.txt  # file containing all required libraries for installing on web server  
├── screenshot_1.jpg  # screenshot of dashboard for README  
│  
```
## 8. Reference
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. 
Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

## 9. Acknowledgements
Thanks to BMW for believing in my aspiration and determination to tackle this Udacity Nanodegree in Data Science 
program as a full-time marketeer with hobbyist interest in programming. It's been truly worth investing the time.  
Thanks to Udacity for creating such a challenging and intense program. It went way beyond other online learning 
platforms and broadened and deepened my understanding of Data Science at the same time. I've learned a tremendous 
amount and the program has increased my skills to a whole new level.  
And last, but most definitely not least: thanks a million to my husband - for always having my back and taking over 
countless hours of looking after our baby boy.