import pandas as pd
import numpy as np
import pickle
import spacy
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt


#Read the test_train data into a dataframe
def read_test_train_csv(NAMEOFTESTFILE,NAMEOFTHETRAINFILE):
    df_test = pd.read_csv(NAMEOFTESTFILE, delimiter="\t", names=["label", "text", "answer","nan"])
    df_train = pd.read_csv(NAMEOFTHETRAINFILE, delimiter="\t", names=["label", "text", "answer"])
    df_test = df_test.drop("nan", axis = 1)
    labels = ["GEOGRAPHY", "MUSIC", "LITERATURE", "HISTORY", "SCIENCE"]
    for row_number, text in enumerate(df_train.text):
        if labels.count(text) > 0:
            df_train = df_train.drop(row_number, axis = "index")
    df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)
    df_train, df_test = make_numerical_labels(df_train), make_numerical_labels(df_test)
    return df_train, df_test

def make_numerical_labels(df):
    df.loc[df.label == "GEOGRAPHY", "label"] = 0
    df.loc[df.label == "MUSIC", "label"] = 1
    df.loc[df.label == "LITERATURE", "label"] = 2
    df.loc[df.label == "HISTORY", "label"] = 3
    df.loc[df.label == "SCIENCE", "label"] = 4
    return df

#plot most frequent words
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
        print(label)
        all_tokens_class = pd.Series(reduce(lambda x, y: x + y, new_texts[df['label'] == label]))
        tokens_freq_class = all_tokens_class.value_counts()[:30]
        sns.barplot(x=tokens_freq_class.values, y=tokens_freq_class.index,
                    orient='h', color="grey")
        plt.show()

#Preprocess the data using the preprocessing pipeline through the spacy framework
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

#vectorize the BoW tokens
def vectorize_data(train_processed_df, test_processed_df, vectorizer = "tfidf"):
    #Preparing train/test split and making numeric values of data
    if vectorizer=="tfidf":
        Tfidf_vect = TfidfVectorizer(max_features=5000)
    elif vectorizer == "count":
        Tfidf_vect = CountVectorizer()

    Tfidf_vect.fit(train_processed_df.values.astype('U'))
    Train_X_Tfidf = Tfidf_vect.transform(train_processed_df.astype(str))
    Test_X_Tfidf = Tfidf_vect.transform(test_processed_df.astype(str))
    return Train_X_Tfidf, Test_X_Tfidf