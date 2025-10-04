**INTRODUCTION:**
The project aims to develop a web application using Python and Flask that incorporates 
advanced sentiment analysis, review visualization, and product evaluation capabilities. 
The primary goal is to empower organizations to gain valuable insights into customer 
sentiments towards their products. By leveraging sentiment analysis algorithms, the 
platform will process and categorize customer reviews, providing an intuitive 
visualization of sentiments, with a focus on identifying and addressing negative 
feedback. Additionally, the application will implement interactive data visualization 
techniques to present an aggregated overview of customer sentiments, enabling 
organizations to make data-driven decisions and prioritize areas for product enhancement.

**PROPOSED SYSTEM NEED:**
1) Customer Sentiment Analysis
2) Data Visualization
3) Proactive Feedback Management
4) Product Evaluation and Enhancement 
5) Decision Support and Strategy Formulation

<img width="827" height="627" alt="image" src="https://github.com/user-attachments/assets/c5ea907a-8c91-4031-b663-bf8a81265250" />


**FREMEWORK AND LIBRARIES:**
1) Flask – Handles web routing, requests, templates, and navigation.
2) pandas – Reads, writes, and processes CSV files (reviews, ratings).
3) transformers – Uses the pretrained RoBERTa model for sentiment classification.
4) AutoTokenizer – Converts review text into tokens.
5) AutoModelForSequenceClassification – Predicts sentiment scores.
6) NumPy & SciPy (softmax) – Process model outputs into probabilities.
7) statistics.mean – Calculates average rating per dish.
8) matplotlib – Generates and saves bar charts of dish ratings.
9) os – Handles file paths and environment settings.

**MODEL**
cardiffnlp/twitter-roberta-base-sentiment-latest
→ Used to convert reviews into sentiment-based ratings (1–3).
