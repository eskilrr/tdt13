from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
import numpy as np
import pickle
import process_data_jeopardy
import process_data_bbc
from sklearn.metrics import precision_recall_fscore_support

def plot_learning_curve(model, Train_X_Tfidf, y_train, model_name):
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(model, Train_X_Tfidf, y_train, cv=5, scoring="accuracy",
                                                            train_sizes=np.linspace(0.05, 1, 50), verbose=1)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    if model_name != "random forest": plt.plot(train_sizes, train_mean, label='training score')
    plt.plot(train_sizes, test_mean, label='cross-validation score')
    plt.grid()
    plt.legend(loc="best")
    plt.title("Learning Curve")
    plt.xlabel("samples")
    plt.ylabel("accuracy")
    plt.show()

def get_train_test_data():
    #Preparing train/test split and making numeric values of data
    processed_df = pd.read_csv("processed.csv")

    dev_processed_df = pd.read_csv("dev_processed.csv")
    Test_X, y_test = dev_processed_df['text'], dev_processed_df['label']
    Train_X, y_train = processed_df['text'], processed_df['label']

    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(processed_df['text'])

    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    return Train_X_Tfidf, Test_X_Tfidf, y_train, y_test, processed_df, dev_processed_df

#See the distribution of the labels
def plot_data(processed_df, dev_processed_df):
    sns.countplot(processed_df['label'])
    plt.show()

    sns.countplot(dev_processed_df["label"])
    plt.show()

#Create naive bayes classifier and check test results
def naive_bayes_classifier(Train_X_Tfidf, Test_X_Tfidf, y_train, y_test):
    print("-----USING NAIVE BAYES------")
    nb = naive_bayes.ComplementNB()
    #plot_learning_curve(nb, Train_X_Tfidf=Train_X_Tfidf, y_train=y_train, model_name="naive bayes")
    nb.fit(Train_X_Tfidf, y_train)
    predictions_nb = nb.predict(Test_X_Tfidf)
    print("CONFUSION MATRIX:")
    matrix = confusion_matrix(y_test, predictions_nb)
    print(matrix)
    print("accuracy for each class: ")
    print(matrix.diagonal() / matrix.sum(axis=1))
    print("naive bayes total score -> ", accuracy_score(predictions_nb, y_test) * 100)

#Create RF classifier and check test results
def random_forest_classifier(Train_X_Tfidf, Test_X_Tfidf, y_train, y_test):
    ##----USING THE BEST PARAMS OF RANDOM FOREST----
    print("\n----USING RANDOM FOREST CLASSIFIER------")
    rf = RandomForestClassifier(max_depth=15, n_estimators=200, max_features=5, criterion="gini")
    #plot_learning_curve(rf, Train_X_Tfidf=Train_X_Tfidf, y_train=y_train, model_name="random forest")
    rf.fit(Train_X_Tfidf, y_train)
    predictions_rf = rf.predict(Test_X_Tfidf)
    print("CONFUSION MATRIX:")
    matrix = confusion_matrix(y_test, predictions_rf)
    print(matrix)
    print("accuracy for each class: ")
    print(matrix.diagonal() / matrix.sum(axis=1))
    print("random forest Accuracy Score -> ", accuracy_score(predictions_rf, y_test) * 100)

#Hyper parameter tuning for RF
def tune_random_forest(Train_X_Tfidf, y_train):
    parameters = {'min_samples_split': [10,20,40],'n_jobs' : [-1] ,'n_estimators' : [100,200,500], 'max_features' : [2,5,10]}
    svc = RandomForestClassifier()
    clf = GridSearchCV(svc, parameters, cv=5, return_train_score=False)
    clf.fit(Train_X_Tfidf, y_train)
    results = pd.DataFrame(clf.cv_results_)
    print(results[['param_min_samples_split', 'param_n_estimators', 'param_max_features', 'mean_test_score']].head(50))

# Read bbc data and train model
def main_bbc():
    df = pd.read_csv("/Users/eskilriibe/PycharmProjects/NaturalLanguage/tdt13/Data/bbc/articles_processed.csv")
    X, y = df["text"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
    Train_X_Tfidf, Test_X_Tfidf = process_data_bbc.vectorize_data(train_processed_df=X_train, test_processed_df=X_test, vectorizer="tfidf")
    naive_bayes_classifier(Train_X_Tfidf=Train_X_Tfidf, Test_X_Tfidf=Test_X_Tfidf, y_train=y_train, y_test=y_test)
    random_forest_classifier(Train_X_Tfidf = Train_X_Tfidf, Test_X_Tfidf = Test_X_Tfidf, y_train=y_train, y_test=y_test)
    process_data_bbc.plot_frequent_words(df)

#Read jeopardy data and train models
def main_jeopardy():
    processed_data = "/Users/eskilriibe/PycharmProjects/NaturalLanguage/tdt13/Data/jeopardy/processed_data.csv"

    df = pd.read_csv(processed_data)
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], random_state=42, test_size=0.2)


    Train_X_Tfidf, Test_X_Tfidf = process_data_jeopardy.vectorize_data(train_processed_df=X_train, test_processed_df=X_test, vectorizer="count")
    naive_bayes_classifier(Train_X_Tfidf=Train_X_Tfidf, Test_X_Tfidf=Test_X_Tfidf, y_train=y_train, y_test=y_test)
    random_forest_classifier(Train_X_Tfidf = Train_X_Tfidf, Test_X_Tfidf = Test_X_Tfidf, y_train=y_train, y_test=y_test)
    process_data_jeopardy.plot_frequent_words(df)



#Process the bbc data and choose whether preprocessing pipeline should be used
def process_bbc():
    file = "/Users/eskilriibe/PycharmProjects/NaturalLanguage/tdt13/Data/bbc/articles_raw.csv"
    df = pd.read_csv(file)
    df_processed = process_data_bbc.process_df(df, process=False)
    df_processed.to_csv("/Users/eskilriibe/PycharmProjects/NaturalLanguage/tdt13/Data/bbc/articles_processed.csv", index=False)
    print("done")

#Process the jeopardy data and choose whether preprocessing pipeline should be used.
def process_jeopardy():
    file1 = "/Users/eskilriibe/PycharmProjects/NaturalLanguage/tdt13/Data/jeopardy/clean1.txt"
    file2 = "/Users/eskilriibe/PycharmProjects/NaturalLanguage/tdt13/Data/jeopardy/clean2.txt"

    data1, data2 = process_data_jeopardy.read_test_train_csv(NAMEOFTESTFILE=file1, NAMEOFTHETRAINFILE=file2)
    df = pd.concat([data1, data2], ignore_index=True)
    df_processed = process_data_jeopardy.process_df(df, process=False)


    df_processed.to_csv("/Users/eskilriibe/PycharmProjects/NaturalLanguage/tdt13/Data/jeopardy/processed_data.csv", index=False)
    print("done")


if __name__ == '__main__':
    main_bbc()