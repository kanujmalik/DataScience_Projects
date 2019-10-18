# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 04:10:15 2019

@author: Sita Guest
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import time
start_time = time.time()
import textdistance
import pandas as pd
import re
from fuzzywuzzy import fuzz
from numpy import mean
import jellyfish as j
#from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def find_ngrams(text: str, number: int=3) -> set:
    """
    returns a set of ngrams for the given string
    :param text: the string to find ngrams for
    :param number: the length the ngrams should be. defaults to 3 (trigrams)
    :return: set of ngram strings
    """

    if not text:
        return set()

    words = [f'  {x} ' for x in re.split(r'\W+', text.lower()) if x.strip()]

    ngrams = set()

    for word in words:
        for x in range(0, len(word) - number + 1):
            ngrams.add(word[x:x+number])

    return ngrams


def trigram_score(text1: str, text2: str, number: int=3) -> float:
    """
    Finds the similarity between 2 strings using ngrams.
    0 being completely different strings, and 1 being equal strings
    """

    ngrams1 = find_ngrams(text1, number)
    ngrams2 = find_ngrams(text2, number)

    num_unique = len(ngrams1 | ngrams2)
    num_equal = len(ngrams1 & ngrams2)

    return float(num_equal) / float(num_unique)

def name_swap_score(name1, name2):
     
    from collections import Counter
        
    name1_splits = name1.split()
    name2_splits = name2.split()
    name_swap_score = len(Counter(name2_splits) & Counter(name1_splits)) / float(max(len(name1_splits), len(name2_splits)))
    
    return name_swap_score

def first_three(name):
    newstr = name.replace(' ', '')
    return newstr[:3]


def last_three(name):
    newstr = name.replace(' ', '')
    return newstr[-2:]

def check_substring(name1, name2):
    name1_split = name1.split()
    name2_split = name2.split()

    if (set(name2_split) <= set(name1_split)) or (set(name1_split) <= set(name2_split)):
        return 1
    else:
        return 0

def substring_text(name1, name2):
    name1_split = name1.replace(' ', '')
    name2_split = name2.replace(' ', '')
    
    if (name1_split in name2_split) or (name2_split in name1_split):
        return 1
    else:
        return 0


def get_features(name1, name2):
    features = {}
    features['levenshtein'] = textdistance.levenshtein.normalized_similarity(name1, name2) 
    features['jaro'] = textdistance.jaro.normalized_similarity(name1, name2)
    features['jaro_winkler'] = textdistance.jaro_winkler.normalized_similarity(name1, name2)
    features['jaccard'] = textdistance.jaccard.normalized_similarity(name1, name2)
    features['ratcliff_obershelp'] = textdistance.ratcliff_obershelp.normalized_similarity(name1, name2)
    features['hamming'] = textdistance.hamming.normalized_similarity(name1, name2)
    features['needleman_wunsch'] = textdistance.needleman_wunsch.normalized_similarity(name1, name2)
    features['smith_waterman'] = textdistance.smith_waterman.normalized_similarity(name1, name2)
    features['sorensen'] = textdistance.sorensen.normalized_similarity(name1, name2)
    features['tversky'] = textdistance.tversky.normalized_similarity(name1, name2)
    features['overlap'] = textdistance.overlap.normalized_similarity(name1, name2)
    features['cosine'] = textdistance.cosine.normalized_similarity(name1, name2)
    features['bag'] = textdistance.bag.normalized_similarity(name1, name2)
    features['lcsseq'] = textdistance.lcsseq.normalized_similarity(name1, name2)
    features['lcsstr'] = textdistance.lcsstr.normalized_similarity(name1, name2)
    features['mra'] = textdistance.mra.normalized_similarity(name1, name2)
    features['editex'] = textdistance.editex.normalized_similarity(name1, name2)
    features['damerau_levenshtein'] = textdistance.damerau_levenshtein.normalized_similarity(name1, name2)
    features['fuzz_wratio'] = fuzz.WRatio(name1, name2)/100
    features['trigram'] = trigram_score(name1, name2)
    features['name_swap'] = name_swap_score(name1, name2)
    features['nysiis'] = textdistance.levenshtein.normalized_similarity(j.nysiis(name1), j.nysiis(name2))
    features['metaphone'] = textdistance.levenshtein.normalized_similarity(j.metaphone(name1), j.metaphone(name2))
    features['soundex'] = textdistance.levenshtein.normalized_similarity(j.soundex(name1), j.soundex(name2))
    #lev_nospace = textdistance.levenshtein.normalized_similarity(name1_split, name2_split)
#    first_three1 = first_three(name1)
#    first_three2 = first_three(name2)
#    last_three1 = last_three(name1)
#    last_three2 = last_three(name2)
#    is_substring = check_substring(name1, name2)
#    is_subtext = substring_text(name1, name2)
    #avg'] = metrics[['levenshtein', 'jaro', 'jaro_winkler', 'jaccard', 'ratcliff_obershelp', 'hamming', 'needleman_wunsch', 'smith_waterman', 'sorensen', 'tversky', 'overlap', 'cosine', 'bag', 'lcsseq', 'lcsstr', 'mra', 'editex', 'damerau_levenshtein', 'fuzz_wratio', 'trigram', 'name_swap']].mean()
    #avg = np.mean(levenshtein, jaro, jaro_winkler, jaccard, ratcliff_obershelp, hamming, needleman_wunsch, smith_waterman, sorensen, tversky, overlap, cosine, bag, lcsseq, lcsstr, mra, editex, damerau_levenshtein, fuzz_wratio, trigram, name_swap)
    
    #return pd.Series(metrics, index = ['levenshtein', 'jaro', 'jaro_winkler', 'jaccard', 'ratcliff_obershelp', 'hamming', 'needleman_wunsch', 'smith_waterman', 'sorensen', 'tversky', 'overlap', 'cosine', 'bag', 'lcsseq', 'lcsstr', 'mra', 'editex', 'damerau_levenshtein', 'fuzz_wratio', 'trigram', 'name_swap', 'first_three1', 'first_three2', 'last_three1', 'last_three2'])
    #return metrics
    #avg = (levenshtein+ jaro+ jaro_winkler+ jaccard+ ratcliff_obershelp+ hamming+ needleman_wunsch+ smith_waterman+ sorensen+ tversky+ overlap+ cosine+ bag+ lcsseq+ lcsstr+ mra+ editex+ damerau_levenshtein+ fuzz_wratio+ trigram+ name_swap)/21
    #a = [levenshtein, jaro, jaro_winkler, jaccard, ratcliff_obershelp, needleman_wunsch, smith_waterman, sorensen, tversky, overlap, cosine, bag, lcsseq, lcsstr, editex, damerau_levenshtein, fuzz_wratio, trigram, name_swap, nysiis, metaphone, soundex, lev_nospace]
    #avg = mean(a)
    return features

def feature_engineering(df):
    feats_list = df.apply(lambda x: get_features(x['name1'], x['name2']), axis=1).values.tolist()
    df_cl_feats = pd.DataFrame(feats_list)
        
    return df_cl_feats.values, df_cl_feats.to_dict('records'), df_cl_feats.columns




df_cl_main = pd.read_csv('C:\\SITA\\train_final.csv')  #Loading training data
#df_cl_test = pd.read_csv('../input/test.csv')   #Loading test data
df_cl_train = df_cl_main[['name1', 'name2', 'label']]
#df_cl_test = df_cl_train.sample(frac=0.2)


X_train, train_dict, cols = feature_engineering(df_cl_train)    #training set data
y_train = df_cl_train.label.values    #training set labels

#X_test = feature_engineering(df_cl_test)      #test set

#ss = StandardScaler()  
#X_train = ss.fit_transform(X_train)
#X_test = ss.transform(X_test)

dt = DecisionTreeClassifier()  #Using default parameters.
dt.fit(X_train, y_train)    #training the model with X_train, y_train

feat_imp = (dict(zip(cols, dt.feature_importances_)))
print(feat_imp)

"""
Predict
"""

#first_name1 = "Kanuj"
#last_name1 = "Malik"
#
#first_name2 = "Paulo"
#last_name2 = "Rossi"
#
#name1 = (first_name1 + " " + last_name1)
#name2 = (first_name2 + " " + last_name2)
name1 = 'Alistair Baron'
name2 = 'Kanuj Malik'

lst_name1 = []
lst_name2 = []
lst_name1.append(name1)
lst_name2.append(name2)


df_cl_test = pd.DataFrame()
df_cl_test['name1'] = lst_name1
df_cl_test['name2'] = lst_name2


X_test, test_dict_lst, test_cols = feature_engineering(df_cl_test)
test_dict = {k:v for element in test_dict_lst for k,v in element.items()}
y_pred = dt.predict(X_test)
sim_score = sum(feat_imp[k]*test_dict[k] for k in feat_imp)
pred_prob = dt.predict_proba(X_test)
if y_pred == 1:
    res = "Yes"
else:
    res = "No"
#sim_score = score_sum/len(test_dict)
print("prediction: ", y_pred)
print("Is likely a match ?: ", res)
print("similarity score: ", sim_score)
print("predict probability: ", pred_prob)

from sklearn.externals import joblib
joblib.dump(dt, 'dt_model_maintrain.pkl')

dt1 = joblib.load('dt_model_maintrain.pkl')
name1 = 'Alistair Baron'
name2 = 'Kanuj Malik'

lst_name1 = []
lst_name2 = []
lst_name1.append(name1)
lst_name2.append(name2)


df_cl_test = pd.DataFrame()
df_cl_test['name1'] = lst_name1
df_cl_test['name2'] = lst_name2


X_test, test_dict_lst, test_cols = feature_engineering(df_cl_test)
test_dict = {k:v for element in test_dict_lst for k,v in element.items()}
y_pred = dt1.predict(X_test)
sim_score = sum(feat_imp[k]*test_dict[k] for k in feat_imp)
pred_prob = dt1.predict_proba(X_test)
if y_pred == 1:
    res = "Yes"
else:
    res = "No"
#sim_score = score_sum/len(test_dict)
print("prediction: ", y_pred)
print("Is likely a match ?: ", res)
print("similarity score: ", sim_score)
print("predict probability: ", pred_prob)



end_time = time.time()
print(end_time-start_time)