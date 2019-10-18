# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:42:51 2019

@author: kanujmalik
"""

import time
start_time = time.time()
import textdistance
import pandas as pd
import re
from fuzzywuzzy import fuzz
from numpy import mean
import jellyfish as j
import seaborn as sns
from sklearn.cluster import KMeans
import math

def calculate_wcss(data):
        wcss = []
        for n in range(2, 15):
            kmeans = KMeans(n_clusters=n)
            kmeans.fit(X=data)
            wcss.append(kmeans.inertia_)
    
        return wcss
    
    
def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2

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


def similarity(df):
    name1 = df['name1']
    name2 = df['name2']
    name1_split = name1.replace(' ', '')
    name2_split = name2.replace(' ', '')
    levenshtein = textdistance.levenshtein.normalized_similarity(name1, name2) 
    jaro = textdistance.jaro.normalized_similarity(name1, name2)
    jaro_winkler = textdistance.jaro_winkler.normalized_similarity(name1, name2)
    jaccard = textdistance.jaccard.normalized_similarity(name1, name2)
    ratcliff_obershelp = textdistance.ratcliff_obershelp.normalized_similarity(name1, name2)
    hamming = textdistance.hamming.normalized_similarity(name1, name2)
    needleman_wunsch = textdistance.needleman_wunsch.normalized_similarity(name1, name2)
    smith_waterman = textdistance.smith_waterman.normalized_similarity(name1, name2)
    sorensen = textdistance.sorensen.normalized_similarity(name1, name2)
    tversky = textdistance.tversky.normalized_similarity(name1, name2)
    overlap = textdistance.overlap.normalized_similarity(name1, name2)
    cosine = textdistance.cosine.normalized_similarity(name1, name2)
    bag = textdistance.bag.normalized_similarity(name1, name2)
    lcsseq = textdistance.lcsseq.normalized_similarity(name1, name2)
    lcsstr = textdistance.lcsstr.normalized_similarity(name1, name2)
    mra = textdistance.mra.normalized_similarity(name1, name2)
    editex = textdistance.editex.normalized_similarity(name1, name2)
    damerau_levenshtein = textdistance.damerau_levenshtein.normalized_similarity(name1, name2)
    fuzz_wratio = fuzz.WRatio(name1, name2)/100
    trigram = trigram_score(name1, name2)
    name_swap = name_swap_score(name1, name2)
    nysiis = textdistance.levenshtein.normalized_similarity(j.nysiis(name1), j.nysiis(name2))
    metaphone = textdistance.levenshtein.normalized_similarity(j.metaphone(name1), j.metaphone(name2))
    soundex = textdistance.levenshtein.normalized_similarity(j.soundex(name1), j.soundex(name2))
    #lev_nospace = textdistance.levenshtein.normalized_similarity(name1_split, name2_split)
    first_three1 = first_three(name1)
    first_three2 = first_three(name2)
    last_three1 = last_three(name1)
    last_three2 = last_three(name2)
    return levenshtein, jaro, jaro_winkler, jaccard, ratcliff_obershelp, hamming, needleman_wunsch, smith_waterman, sorensen, tversky, overlap, cosine, bag, lcsseq, lcsstr, mra, editex, damerau_levenshtein, fuzz_wratio, trigram, name_swap, nysiis, metaphone, soundex, first_three1, first_three2, last_three1, last_three2

df_names_concat_merged_pairs = pd.read_csv('C:\\SITA\\names_concat_merged_pairs.csv')

df = df_names_concat_merged_pairs[['id', 'name1', 'name2']]



df_nohash = df

df_nohash['name1'] = df_nohash['name1'].str.upper()
df_nohash['name2'] = df_nohash['name2'].str.upper()
df_nohash['name1'] = df_nohash['name1'].str.replace('[^A-Z\s]+', '')
df_nohash['name2'] = df_nohash['name2'].str.replace('[^A-Z\s]+', '')
remove_words = ['FNU', 'LNU', 'GNU', 'MNU', 'UNK', 'UNKNOWN']
pattern = r'\b(?:{})\b'.format('|'.join(remove_words))
df_nohash['name1'] = df_nohash['name1'].str.replace(pattern, '')
df_nohash['name2'] = df_nohash['name2'].str.replace(pattern, '')

df_nohash = df_nohash[df_nohash['name1'].str.strip().astype(bool)]
df_nohash = df_nohash[df_nohash['name2'].str.strip().astype(bool)]

df_distance =   df_nohash       
df_distance[['levenshtein', 'jaro', 'jaro_winkler', 'jaccard', 'ratcliff_obershelp', 'hamming', 'needleman_wunsch', 'smith_waterman', 'sorensen', 'tversky', 'overlap', 'cosine', 'bag', 'lcsseq', 'lcsstr', 'mra', 'editex', 'damerau_levenshtein', 'fuzz_wratio', 'trigram', 'name_swap', 'nysiis', 'metaphone', 'soundex', 'first_three1', 'first_three2', 'last_three1', 'last_three2']] = df_distance.apply(similarity, axis=1, result_type="expand")

df_cluster = df_distance[['id', 'name1', 'name2', 'levenshtein', 'jaro', 'jaro_winkler', 'jaccard', 'ratcliff_obershelp', 'needleman_wunsch', 'smith_waterman', 'sorensen', 'tversky', 'overlap', 'cosine', 'bag', 'lcsseq', 'lcsstr', 'editex', 'damerau_levenshtein', 'fuzz_wratio', 'trigram', 'name_swap', 'nysiis', 'metaphone', 'soundex']]
df_cluster = df_cluster.set_index(['id', 'name1', 'name2'])

sum_of_squares = calculate_wcss(df_cluster)
n = optimal_number_of_clusters(sum_of_squares)
kmeans = KMeans(n_clusters=n)
clusters = kmeans.fit_predict(df_cluster)
df_cluster['Cluster'] = clusters

df_cluster.to_csv('C:\\SITA\\traindata_kmeans2.csv', header=True)


"""
Labeling
"""

import re
def remove_title(name):
    replace_list = ['MRS', 'MISS', 'MS', 'MR']
    for cur_word in replace_list:
        name = name.replace(cur_word, '')
    
    #re.sub(r"MRS|MISS|MS|MR", "", name)
    name = name.strip()
    return name

def remove_space(name):
    name = name.replace(' ', '')
    return name

def first_two_space(name):
    return name[:2]

def last_two_space(name):
    return name[-2:]

def first_two_nospace(name):
    name = remove_space(name)
    return name[:2]

def check_subs(name1, name2):
    name1 = remove_title(name1)
    name2 = remove_title(name2)
    return check_substring(name1, name2)

def check_subs_nospace(name1, name2):
    name1 = remove_title(name1)
    name2 = remove_title(name2)
    return substring_text(name1, name2)
    

df_label= df_cluster.reset_index()
import numpy as np
df_label['first_two_nospace1'] = np.vectorize(first_two_nospace)(df_label['name1'])
df_label['first_two_nospace2'] = np.vectorize(first_two_nospace)(df_label['name2'])
df_label['last_two_nospace1'] = np.vectorize(last_three)(df_label['name1'])
df_label['last_two_nospace2'] = np.vectorize(last_three)(df_label['name2'])
df_label['first_two_space1'] = np.vectorize(first_two_space)(df_label['name1'])
df_label['first_two_space2'] = np.vectorize(first_two_space)(df_label['name2'])
df_label['last_two_space1'] = np.vectorize(last_two_space)(df_label['name1'])
df_label['last_two_space2'] = np.vectorize(last_two_space)(df_label['name2'])
df_label['is_substring'] = np.vectorize(check_subs)(df_label['name1'], df_label['name2'])
df_label['is_subs_nospace'] = np.vectorize(check_subs_nospace)(df_label['name1'], df_label['name2'])

df_label['label']=0

df_label.loc[(df_label['Cluster']==0) | (df_label['Cluster']==1) | (df_label['Cluster']==4), 'label'] = 1
df_label.loc[df_label['Cluster']==3, 'label'] = 0
#df_label['label'] = df_label.apply(lambda row: 1 if row['Cluster']==3 & ((first_two_nospace(row['name1'])==first_two_nospace(row['name2']) | first_two_space(row['name1'])==first_two_space(row['name2'])) | (last_three(row['name1'])==last_three(row['name2']) | last_two_space(row['name1'])==last_two_space(row['name2'])) | (check_substring(row['name1'], row['name2'])==1 | substring_text(row['name1'], row['name2'])==1)), axis=1)
#df_label.loc[(df_label['Cluster']==3) & (((first_two_nospace(df_label['name1'])==first_two_nospace(df_label['name2'])) | (first_two_space(df_label['name1'])==first_two_space(df_label['name2']))) & ((last_three(df_label['name1'])==last_three(df_label['name2'])) | (last_two_space(df_label['name1'])==last_two_space(df_label['name2']))) | ((check_subs(df_label['name1'], df_label['name2'])==1) | (check_subs_nospace(df_label['name1'], df_label['name2'])==1))), 'label'] = 1
df_label.loc[(df_label['Cluster']==2) & (((df_label['first_two_nospace1']==df_label['first_two_nospace2']) | (df_label['first_two_space1']==df_label['first_two_space2'])) & ((df_label['last_two_nospace1']==df_label['last_two_nospace2']) | (df_label['last_two_space1']==df_label['last_two_space2'])) | ((df_label['is_substring']==1) | (df_label['is_subs_nospace']==1))), 'label'] = 1

df_train = df_label[['id', 'name1', 'name2', 'label']]

df_train.to_pickle('C:\\SITA\\train_label.pkl')
df_train.to_csv('C:\\SITA\\train_label.csv', header=True)


    
    

end_time = time.time()
print(end_time-start_time)		