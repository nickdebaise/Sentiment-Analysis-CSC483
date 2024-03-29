# Sentiment Analysis on Financial News Headlines

- _Read Nick's research paper [here](<Research Paper - Nick DeBaise.pdf>)_
- _Read Jeremy's research paper [here](<Reserach Paper - Jeremy Perez.pdf>)_

### 1. Background and Goals
Our project's goal is to perform a sentiment analysis on news headlines pertaining to specific stocks. From this, we will use these sentiments to evaluate and predict whether a stock should be bought or sold on that day. We're hoping to learn more about the different ways to apply BERT models and investigate what it looks like to build an application that attempts to predict real-world movements based on pre-trained models. The task of predicting share prices is ultimately much more complex than the project itself, however we focus on extracting the non-quantifiable information within the news headlines and seeing how they correlate (if at all) to the real markets.

### 2.  Prerequisites to Running
If external corpus is wanted, override the raw_analyst_ratings.csv file in the main directory and follow the format specified [`below`](#corpus).

The corpus can be downloaded from [kaggle](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests). The corpus used in the project comes from the raw_analyst_ratings.csv file in Kaggle.

### 3. How to Install and Run

    pip install -r requirements.txt 
    python3 stock_prediction.py

#### Understanding Output
The application outputs the models' evaluation based on how successful your model was in predicting the stocks' movement. The evaluation consist of calculating the models: 

- Accuracy
- Precision
- Recall
- F-score

### 4. File Overview

#### stock_prediction.py
This file is responsible for training and testing the neural models behind making the actual predictions. This is the main file of the project and this is where configurations can be found.

#### stock_price.py
Stores internal models needed to work with stocks and containing the information surrounding them. Internalized, they make use of Yahoo Finance API to get ticker information about a given stock and stores it.

#### constants.py
Contains constants for project.

#### mock_finBERT.py
Mini implementation of Hugging Face's FinBert model which classifies headline data into positive, negative, and neutral states.

<h4 id="corpus"> raw_analyst_ratings.csv </h4> 

This is the location of the corpus needed to train the classifiers within the predictor.
If external / different corpus is wanted, the CSV format must correspond to the CSV_COLUMN constant within constants.py.

Entries/Columns required to run program:
 - INDEX: An integer corresponding to each row of data
 - HEADLINE: Headline of the article
 - URL: Corresponds to the article URL
 - PUBLISHER: Article Publisher
 - DATE: Date article was published
 - TICKER: The symbol of the companies covered in the 


#### scripts.py
Bucket of useful functions which are used all throughout the project. 

### 5. Program Configuration
Configuration parameters for the predictor can be found within stock_prediction.py. Main neural structure and learning phases can be found and edited here as well

### 6. About Project
This group project expands upon the concepts discussed in Natural Language Processing Courses and applies them to the domain of Economics.

#### Contributors to building this project

- Nicholas DeBaise
- Hawkeye Nadel
- Jeremy Perez
