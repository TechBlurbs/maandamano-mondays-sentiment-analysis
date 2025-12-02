# Maandamano-mondays-sentiment-analysis
A project to gauge the sentiments of Kenyans on the Monday Maandamano's in the country.

# 🎯 Objectives

* To accurately extract people's opinions from a large number of unstructured texts and classify them into sentiment classes, i.e. positive, negative, or neutral.
* To determine the economic setbacks arising from the close of businesses due to the demonstrations.
* To evaluate model performance using comprehensive metrics including precision, recall, and F1 score.

# 📊 Model Evaluation

This project includes a comprehensive model evaluation system that assesses the sentiment analysis model's performance using various metrics:

- **Precision, Recall, and F1 Score** for each sentiment class
- **Overall Accuracy** across all predictions
- **Confusion Matrix** for detailed error analysis
- **Class Distribution Analysis** to understand data balance

See `EVALUATION_GUIDE.md` for detailed instructions on running the evaluation.

# What is Sentiment Analysis

Sentiment analysis is the process of analyzing text data to determine the emotional tone behind it, whether it is positive, negative, or neutral. Sentiment analysis has become an increasingly important area of study in recent years, with applications ranging from customer service to social media monitoring.

Here are some reasons why sentiment analysis is a great project for a data scientist portfolio:

## 1. High demand in industry
Sentiment analysis is widely used across different industries, including e-commerce, finance, healthcare, and more. It is an essential tool for businesses to understand their customers' opinions and make informed decisions. Therefore, demonstrating your proficiency in sentiment analysis through a project in your portfolio can make you a desirable candidate for potential employers.

## 2. Real-world applications
Sentiment analysis has many practical applications, such as predicting customer behavior, identifying emerging trends, and tracking brand reputation. Developing a sentiment analysis project that can accurately analyze and predict sentiment in real-world scenarios can help showcase your ability to apply data science techniques to solve real-world problems.

## 3. Large amounts of data
Sentiment analysis typically involves processing large amounts of text data, which can be challenging to analyze manually. As a data scientist, you can leverage your skills to develop automated sentiment analysis models that can quickly and accurately process large volumes of data. Demonstrating this ability through a sentiment analysis project can help set you apart from other candidates.

## 4. Variety of techniques
Sentiment analysis involves a variety of techniques, including natural language processing (NLP), machine learning, and deep learning. Developing a sentiment analysis project that leverages these techniques can help showcase your technical skills and experience working with these tools.

## 5. Accessibility of data
There is a wide range of publicly available data sets that can be used for sentiment analysis projects, such as Twitter data, product reviews, and news articles. This accessibility to data allows you to experiment with different approaches to sentiment analysis and showcase your creativity and problem-solving skills.

Overall, sentiment analysis is a great project for a data scientist portfolio as it demonstrates your ability to apply data science techniques to solve real-world problems, work with large amounts of data, and showcase your technical skills. As sentiment analysis becomes increasingly important in many industries, having a project in your portfolio that showcases your proficiency in this area can help set you apart from other candidates and increase your chances of success in the job market.

# Installation Guide
* Clone the repo
* Create python environment
```sh
python -m venv env
```
* Activate python environment
```sh
source env/bin/activate
```
* Collect the data
```sh
python data_collection/sns_scrape.py
```
* Preprocess data
```sh
python data_processing/add_labels.py
```
* Run jupyter notebook
```sh
jupyter notebook
```
* Evaluate model performance
```sh
python model_evaluation.py
```