# -*- coding: utf-8 -*-
"""
Applied Data Mining - Final assignment
"""


"""
1. Load and read the dataset
Keep only the text and airline_sentiment (label) columns 
"""

import pandas as pd
import numpy as np

df = pd.read_csv('Tweets.csv')
#df = df.reindex(np.random.permutation(df.index))
df = df[['text', 'airline_sentiment']]
print(df['airline_sentiment'].value_counts())


"""
2. Function to generate heatmap for the classification results
"""
#Heatmap function
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

def confusion_matrix_heatmap(cm, index):
    cmdf = pd.DataFrame(cm, index = index, columns=index)
    dims = (5, 5)
    fig, ax = plt.subplots(figsize=dims)
    sns.heatmap(cmdf, annot=True, cmap="coolwarm", center=0)
    ax.set_ylabel('Actual')    
    ax.set_xlabel('Predicted')
    


"""
3. Functions to pre-process the data - remove unwanted features
"""
#Preprocess
import ftfy
import nltk
import re

nltk.download('punkt')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')

hashtag_re = re.compile(r"#\w+")
mention_re = re.compile(r"@\w+")
url_re = re.compile(r"(?:https?://)?(?:[-\w]+\.)+[a-zA-Z]{2,9}[-\w/#~:;.?+=&%@~]*")


def preprocess(text):
    p_text = hashtag_re.sub("[hashtag]",text)
    p_text = mention_re.sub("[mention]",p_text)
    p_text = url_re.sub("[url]",p_text)
    p_text = ftfy.fix_text(p_text)
    return p_text


"""
5. Tokenization methods
"""
tokenise_re = re.compile(r"(\[[^\]]+\]|[-'\w]+|[^\s\w\[']+)") #([]|words|other non-space)
def custom_tokenise(text):
    return tokenise_re.findall(text.lower())

def nltk_twitter_tokenise(text):
    twtok = nltk.tokenize.TweetTokenizer()
    return twtok.tokenize(text.lower())

def nltk_word_tokenizer(text):
    #wtok = nltk.tokenize.word_tokenize()
    return nltk.tokenize.word_tokenize(text.lower())

def nltk_regex_tokenizer(text):
    return nltk.tokenize.regexp_tokenize(text.lower(), r"\w+|[^\w\s]+") #pattern that matches both mentions and hashtags


"""
5. Set X, y values
X = tweets (text)
y = label (positive, negative, neutral)
"""
#Define X,y
X = df['text']
y = df.airline_sentiment


"""
6. Split the the data into training and test data
"""
#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)



"""
7. Create classification model pipeline - Naive Bayes and Logistic Regression
Create Models and evaluate for each tokenizer
Extract Features under CountVectorizer - ngrams, word analyzer, max_features
Perform K fold vross validation under GridSearchcv
Evaluate model on test data
Calculate accuracy and other performance metrics
"""
#CLassification model
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Binarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_validate, StratifiedKFold


#Custom tokenizer
model = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='word',tokenizer=custom_tokenise, preprocessor=preprocess, max_features=1000, ngram_range=(1,2))),
    ('norm', Binarizer()),
    ('norm2', TfidfTransformer(norm=None)),    
    ('selector', SelectKBest(score_func = chi2)),
    ('clf', LogisticRegression(solver='liblinear', random_state=0)),
])

search = GridSearchCV(model, cv=StratifiedKFold(n_splits=5, random_state=0), 
                      return_train_score=False, 
                      scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                      refit = 'f1_weighted',
                      param_grid={
                          'selector__k': [10, 50, 100, 250, 500, 1000],
                          'clf': [MultinomialNB(), LogisticRegression(solver='liblinear', random_state=0)],
                      })


search.fit(X_train.values, y_train.values)



#Results
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

predictions = search.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

confusion_matrix_heatmap(confusion_matrix(y_test,predictions), search.classes_)


#Tweet tokenizer
model = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='word',tokenizer=nltk_twitter_tokenise, preprocessor=preprocess, max_features=1000, ngram_range=(1,2))),
    ('norm', Binarizer()),
    ('norm2', TfidfTransformer(norm=None)),    
    ('selector', SelectKBest(score_func = chi2)),
    ('clf', LogisticRegression(solver='liblinear', random_state=0)),
])

search = GridSearchCV(model, cv=StratifiedKFold(n_splits=5, random_state=0), 
                      return_train_score=False, 
                      scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                      refit = 'f1_weighted',
                      param_grid={
                          'selector__k': [10, 50, 100, 250, 500, 1000],
                          'clf': [MultinomialNB(), LogisticRegression(solver='liblinear', random_state=0)],
                      })


search.fit(X_train.values, y_train.values)



#Results

predictions = search.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

confusion_matrix_heatmap(confusion_matrix(y_test,predictions), search.classes_)


#Word tokenizer
model = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='word',tokenizer=nltk_word_tokenizer, preprocessor=preprocess, max_features=1000, ngram_range=(1,2))),
    ('norm', Binarizer()),
    ('norm2', TfidfTransformer(norm=None)),    
    ('selector', SelectKBest(score_func = chi2)),
    ('clf', LogisticRegression(solver='liblinear', random_state=0)),
])

search = GridSearchCV(model, cv=StratifiedKFold(n_splits=5, random_state=0), 
                      return_train_score=False, 
                      scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                      refit = 'f1_weighted',
                      param_grid={
                          'selector__k': [10, 50, 100, 250, 500, 1000],
                          'clf': [MultinomialNB(), LogisticRegression(solver='liblinear', random_state=0)],
                      })


search.fit(X_train.values, y_train.values)



predictions = search.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

confusion_matrix_heatmap(confusion_matrix(y_test,predictions), search.classes_)




#Regex tokenizer
model = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='word',tokenizer=nltk_regex_tokenizer, preprocessor=preprocess, max_features=1000, ngram_range=(1,2))),
    ('norm', Binarizer()),
    ('norm2', TfidfTransformer(norm=None)),    
    ('selector', SelectKBest(score_func = chi2)),
    ('clf', LogisticRegression(solver='liblinear', random_state=0)),
])

search = GridSearchCV(model, cv=StratifiedKFold(n_splits=5, random_state=0), 
                      return_train_score=False, 
                      scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                      refit = 'f1_weighted',
                      param_grid={
                          'selector__k': [10, 50, 100, 250, 500, 1000],
                          'clf': [MultinomialNB(), LogisticRegression(solver='liblinear', random_state=0)],
                      })


search.fit(X_train.values, y_train.values)



predictions = search.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

confusion_matrix_heatmap(confusion_matrix(y_test,predictions), search.classes_)
  
