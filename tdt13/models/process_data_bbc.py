import os
import pandas as pd
import numpy as np
import pickle
import spacy
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from functools import reduce
import seaborn as sns

# Function to gather and structure the bbc data into one file
def gather_data():
    data = []
    articles_path = '/tdt13/Data/bbc'
    for topic in os.listdir(articles_path):
        if not topic.startswith('.') and not topic.startswith("README"):
            topic_path = os.path.join(articles_path, topic)
            for file_path in os.listdir(topic_path):
                try:
                    with open(f'{topic_path}/{file_path}', 'r') as f:
                        data.append((f.read(), topic_path.split('/')[-1]))
                except UnicodeDecodeError:
                    print(f'file {file_path} in {topic_path} is not readable')

    df = pd.DataFrame(data, columns=['text', 'label'])
    df.to_csv("/Users/eskilriibe/PycharmProjects/NaturalLanguage/tdt13/Data/bbc/articles_raw.csv", index=False)
    print(df.head())

def make_numerical_labels(df):
    df.loc[df.label == "business", "label"] = 0
    df.loc[df.label == "sport", "label"] = 1
    df.loc[df.label == "entertainment", "label"] = 2
    df.loc[df.label == "tech", "label"] = 3
    df.loc[df.label == "politics", "label"] = 4
    return df

#Visualize the frequency of the words
def plot_wordcloud(documents, dataset="bbc"):

    comment_words = ''
    stopwords = set(STOPWORDS)
    labels = ["business", "sport", "entertainment", "tech", "politics"]
    # iterate through the csv file
    for label in labels:
        new_documents = documents.loc[documents.label == label]
        for val in new_documents.text:

            # typecaste each val to string
            val = str(val)

            # split the value
            tokens = val.split()

            # Converts each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()

            comment_words += " ".join(tokens) + " "

        wordcloud = WordCloud(width=800, height=800,
                              background_color='white',
                              stopwords=stopwords,
                              min_font_size=10).generate(comment_words)

        # plot the WordCloud image
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)

        plt.show()

#See the 30 most frequent words
def plot_frequent_words(df):
    nlp = spacy.load("en_core_web_sm")
    df["text"] = df['text'].str.lower()
    new_texts = []
    for index, row in df.iterrows():
        doc = nlp(row['text'])
        doc = [token.text for token in doc if token.is_stop == False and token.text.isalpha() == True]
        new_texts.append(doc)
    new_texts = pd.Series(new_texts)

    for i, label in enumerate(df['label'].unique()):
        all_tokens_class = pd.Series(reduce(lambda x, y: x + y, new_texts[df['label'] == label]))
        tokens_freq_class = all_tokens_class.value_counts()[:30]
        sns.barplot(x=tokens_freq_class.values, y=tokens_freq_class.index,
                    orient='h', color="grey")
        plt.show()

#Pre-process the data using the described preprocessing pipeline and through the spacy framework
def process_df(df, process = True):
    if process == True:
        nlp = spacy.load("en_core_web_sm")
        # 1. making caracters to lower
        df["text"] = df['text'].str.lower()
        # 2. Tokenization, 3. Lemmatization and 4. removing stop-words and characters/numbers
        for index, row in df.iterrows():
            doc = nlp(row['text'])
            doc = [token.lemma_ for token in doc if token.is_stop == False and token.text.isalpha() == True]
            new_text = ""
            for token in doc:
                new_text += " " + token
            df.at[index, 'text'] = new_text
    return df


#Vectorize the BoW tokens
def vectorize_data(train_processed_df, test_processed_df, vectorizer = "tfidf"):
    #Preparing train/test split and making numeric values of data

    if vectorizer == "tfidf":
        Tfidf_vect = TfidfVectorizer(max_features=5000)
        Tfidf_vect.fit(train_processed_df.astype(str))

        Train_X_Tfidf = Tfidf_vect.transform(train_processed_df.astype(str))
        Test_X_Tfidf = Tfidf_vect.transform(test_processed_df.astype(str))
        return Train_X_Tfidf, Test_X_Tfidf

    elif vectorizer == "count":
        Tfidf_vect = CountVectorizer()
        Tfidf_vect.fit(train_processed_df.astype(str))

        Train_X_Tfidf = Tfidf_vect.transform(train_processed_df.astype(str))
        Test_X_Tfidf = Tfidf_vect.transform(test_processed_df.astype(str))
        return Train_X_Tfidf, Test_X_Tfidf

    return None

