# Capstone Project: Instagram Dashboard for MINI

## 1. Project Definition

### Project Overview
This project is a web tool for marketers at MINI. 
It takes in Instagram data, analyses, classifies and visualizes the data in various charts.
These help the global Brand Management team at MINI to monitor developments on Instagram and to draw conclusions.

### Problem Statement
Continuous monitoring, analysis and deriving insights is important in the global steering role of the Brand Management team at MINI.
The problem is however, that the medium is ever-evolving and it is very resource-intense to keep track of this in a manual fashion.
Retrieving relevant data from the platform by accessing Instagram in an automated fashion, such as using typical scraping techniques however violates the platforms' ToS.

### Proposed Solution
The proposed solution involves automating the process as much as (legally) possible, i.e. the analysis, classification and visualization of data.
This will make it much more efficient and less error-prone.
Hence the original data for setting up the tool, was collected through browsing the relevant profiles on Instagram as a regular logged-in user in Chrome and downloading the resulting .har file from Google's Developer Tools.
The .har file is then read in, parsed, cleaned and relevant data snippets saved to a database.
Using this method is also expected to be a more resilient / lasting solution than using methods that circumvent security measures by the platform, e.g. using fake IP addresses / VPN, cookies, session ids and the like.

TO DO The original data used for training the model can be accessed in the data folder // here (due to size)

### Metrics
TO DO: Metrics used to measure performance of a model or result are clearly defined. Metrics are justified based on the characteristics of the problem.

## 2. Analysis

### Data Exploration
TO DO: Features and calculated statistics relevant to the problem have been reported and discussed related to the dataset, and a thorough description of the input space or input data has been made. Abnormalities or characteristics about the data or input that need to be addressed have been identified.

### Data Visualization
TO DO: Build data visualizations to further convey the information associated with your data exploration journey. Ensure that visualizations are appropriate for the data values you are plotting.

## 3. Methodology
### Data Preprocessing
TO DO: All preprocessing steps have been clearly documented. Abnormalities or characteristics about the data or input that needed to be addressed have been corrected. If no data preprocessing is necessary, it has been clearly justified.
- cleaning captions (extracting hashtags) stemming / lemmatization...
- special: deleting emissions data


### Implementation
TO DO: The process for which metrics, algorithms, and techniques were implemented with the given datasets or input data has been thoroughly documented. Complications that occurred during the coding process are discussed.

### Refinement
TO DO: The process of improving upon the algorithms and techniques used is clearly documented. Both the initial and final solutions are reported, along with intermediate solutions, if necessary.

## 4. Results
### Model Evaluation and Validation
TO DO: If a model is used, the following should hold: The final model’s qualities — such as parameters — are evaluated in detail. Some type of analysis is used to validate the robustness of the model’s solution.
Alternatively a student may choose to answer questions with data visualizations or other means that don't involve machine learning if a different approach best helps them address their question(s) of interest.

### Justification
TO DO: The final results are discussed in detail.
Exploration as to why some techniques worked better than others, or how improvements were made are documented.

## 5. Conclusion
### Reflection
TO DO Student adequately summarizes the end-to-end problem solution and discusses one or two particular aspects of the project they found interesting or difficult.

### Opportunities for future improvement
The following aspects would make the dashboard even more insightful, however it would have required additional time and legal clearance, which was not possible within the time frame of the capstone project.
- Include data on competitors' activities (e.g. Volkswagen, Fiat)
- Include sentiment analysis of comments (needs to ensure that any personally identifiable information is removed from comments)
- Include images / screenshot of video of posts directly in the dashboard as a preview
- Include images in the data for the ML classification algorithm
- Include additional social media platforms
All these improvements would result in a broader view of developments or more in-depth understanding.

### Instructions
The project uses the following libraries:
- TODO

### File Tree
TO DO

### Acknowledgements
Thank you BMW for sponsoring my Udacity Nanodegree in Data Science. It's been truly worth investing the time.
Thanks also to Udacity for creating such a challenging and intense program. It definitely broadened and deepened my understanding of Data Science and I've learned a lot on the way.