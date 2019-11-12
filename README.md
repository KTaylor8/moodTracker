# Mood Tracker

I created this sentiment analyzer for my final project in my Computer Programming class in Fall of 2018 in order to learn about natural language processing. The mood tracker asks you about your day, determines the general mood of the response (positive, negative, or neutral), provides an appropriate reaction (currently limited), stores entries over time, and displays your mood graphically over the course of many entries. In its current state, the mood tracker works best with in simple sentences in simple present tense. Also, it is only configured for responses in English.

![Mood Tracker flowchart]
Mood Tracker program flowchart

The example data that I am currently training the classifier on is sourced from Stanford graduate students' [Sentiment140's training data](http://help.sentiment140.com/for-students) and [OpenData Stack Exchange Twitter Sentiment Analysis data](https://old.datahub.io/dataset/twitter-sentiment-analysis/resource/091d6b4b-22e9-4a64-85c4-bdc8028183ac). In the future, I may potentially add back in the data from the [Twitter US Airline Dataset](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) from Kaggle. I use MatPlotLib for the graph and Natural Language Toolkit for the machine learning classifier. I used a Naive Bayes classifier, which assumes strong (naive) independence between attributes.
