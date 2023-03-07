# Sentiment Analysis on Financial News Headlines

###1. Background and Goals
Our project's goal is to perform a sentiment analysis on news headlines pertaining to specific stocks. From this, we will use these sentiments to evaluate and predict whether a stock should be bought or sold on that day. We're hoping to learn more about the different ways to apply BERT models and investigate what it looks like to build an application that attempts to predict real-world movements based on pre-trained models. The task of predicting share prices is ultimately much more complex than the project itself, however we focus on extracting the non-quantifiable information within the news headlines and seeing how they correlate (if at all) to the real markets.


###2. Program Configuration


###3. How to run
pip install -r requirements.txt
python3 stock_prediciton.py

###4. File Overview

####stock_prediction.py
The main file in our project 

####stock_price.py
Stores internal models representing stocks and the information surrounding them. These models use the Yahoo Finance API to get ticker information about a given stock and stores it.

####constants.py
Contains project constants
####mock_finBERT.py

####raw_analyst_ratings.txt
####scripts.py


