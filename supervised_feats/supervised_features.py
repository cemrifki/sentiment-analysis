from collections import Counter
import math
import numpy as np

delta_idf_scores = None

import unsuperv

use_unsuperv = True
UNSUPERV_SCORE = 0.2

def get_corp_count(revs):
    
    word_counts = Counter()
    for rev in revs:
        rev_set_words = set(rev)
        for word in rev_set_words:
            word_counts[word] += 1
            
    return word_counts

def get_all_corpus_words(pos_revs, neg_revs):

    
    all_words = set()
    
    all_words = all_words.union(*pos_revs)
    all_words = all_words.union(*neg_revs)
    return all_words
    
def get_all_revs(pos_revs, neg_revs):
    all_revs = []
    all_revs.extend(pos_revs)
    all_revs.extend(neg_revs)
    return all_revs

def sign(n):
    return 1 if n > 0 else -1

def get_avg_pol_scores(dict1, dict2):
    all_words = set(dict1) | set(dict2)
    
    new_pol_scores = {}
    for word in all_words:
        if word not in dict1:
            new_pol_scores[word] = dict2[word]
        elif word not in dict2:
            new_pol_scores[word] = 0
        else:
            if sign(dict1[word]) == sign(dict2[word]):
                new_pol_scores[word] = (UNSUPERV_SCORE * dict1[word] + (1 - UNSUPERV_SCORE) * dict2[word]) / 1.0
            else:
                new_pol_scores[word] = (1 - UNSUPERV_SCORE) * dict2[word]
                
    return new_pol_scores

def get_delta_idf_vecs(revs, labels):
    
    pos_indices = np.where(np.char.lower(labels) == "p")
    neg_indices = np.where(np.char.lower(labels) == "n")
    
    
    pos_revs = revs[pos_indices]
    neg_revs = revs[neg_indices] 


    unsup_scores = unsuperv.get_unsupervised_scores()

    delta_idf_scores = extract_delta_idf_scores(pos_revs, neg_revs)

    avg_pols = get_avg_pol_scores(unsup_scores, delta_idf_scores)
    if use_unsuperv == True:
        return avg_pols
    else:
        return delta_idf_scores
    #return delta_idf_scores
    
def extract_delta_idf_scores(pos_revs, neg_revs):
    
    
    delta_idf_scores = {}

    pos_counts = get_corp_count(pos_revs)
    neg_counts = get_corp_count(neg_revs)
    
    all_words = get_all_corpus_words(pos_revs, neg_revs)
    
    
    
    for word in all_words:
        pos_val = 0
        neg_val = 0
        if word in pos_counts:
            pos_val = pos_counts[word]
        if word in neg_counts:
            neg_val = neg_counts[word]
        delta_idf_scores[word] = math.log((pos_val / len(pos_revs) + 0.001) /
                              (neg_val / len(neg_revs) + 0.001))
    return delta_idf_scores

def get_all_context_words(all_revs, context_size=5):
    contexts_of_words = {}
    for rev in all_revs:
        
        for i in range(0, len(rev)):
            target_word = rev[i]
            start_index = max(int(i - context_size / 2), 0)
            end_index = min(int(i + context_size / 2), len(rev))
            
            
            left_hand_side_els = {}
            right_hand_side_els = {}
            
            if i > 0:
                left_hand_side_els = set(rev[start_index:i])
            if i < len(rev) - 1:
                right_hand_side_els = set(rev[i + 1:end_index])
            
            word_context_words = set([])
            word_context_words = word_context_words.union(left_hand_side_els).union(right_hand_side_els)
            if target_word in word_context_words:
                word_context_words.remove(target_word)
            
            if target_word not in contexts_of_words:
                contexts_of_words[target_word] = set()
            contexts_of_words[target_word] = contexts_of_words[target_word].union(word_context_words)
            
    return contexts_of_words

def extract_context_delta_idf_scores(context_words, all_delta_idf_scores):
    return [all_delta_idf_scores[word] if word in all_delta_idf_scores else 0.0 for word in context_words]

def extract_all_context_delta_idf_scores(target_and_context_words, all_delta_idf_scores):
    
    all_words_and_context_delta_idfs = {}
    for target_word, context_words in target_and_context_words.items():
        
        cont_delta_idfs = extract_context_delta_idf_scores(context_words, all_delta_idf_scores)
        all_words_and_context_delta_idfs[target_word] = cont_delta_idfs
        
    return all_words_and_context_delta_idfs
        
def get_max_context_delta_idf_score(context_idf_scores):
    return max(context_idf_scores) 
  
def get_min_context_delta_idf_score(context_idf_scores):
    return min(context_idf_scores)

def get_avg_context_delta_idf_score(context_idf_scores):
    return sum(context_idf_scores) / len(context_idf_scores)

def get_context_4_delta_idf_scores(target_word, word_context_delta_idf_scores, delta_idf_scores):
    supervised_4_scores = []
    
    
    supervised_4_scores.append(get_max_context_delta_idf_score(word_context_delta_idf_scores))
    supervised_4_scores.append(get_min_context_delta_idf_score(word_context_delta_idf_scores))
    supervised_4_scores.append(get_avg_context_delta_idf_score(word_context_delta_idf_scores))
    supervised_4_scores.append(delta_idf_scores[target_word])

    return supervised_4_scores

def get_all_context_4_delta_idf_scores(pos_revs, neg_revs):
    global delta_idf_scores
    if delta_idf_scores == None:
        delta_idf_scores = extract_delta_idf_scores(pos_revs, neg_revs)
    all_revs = get_all_revs(pos_revs, neg_revs)
    all_context_words = get_all_context_words(all_revs=all_revs)
    
    context_delta_idf_scores = extract_all_context_delta_idf_scores(all_context_words, delta_idf_scores)
    
    all_context_4_delta_idf_scores = {}


    for target_word, word_context_delta_idf_scores in context_delta_idf_scores.items():
        
        all_context_4_delta_idf_scores[target_word] = get_context_4_delta_idf_scores(target_word, word_context_delta_idf_scores, delta_idf_scores)
        
    return np.array(all_context_4_delta_idf_scores)

def get_4_delta_idf_vecs(revs, labels):
    
    #Düzelt!!!
    
    revs_np = np.array(revs)
    
    pos_indices = np.where(np.char.lower(labels) == "p")
    neg_indices = np.where(np.char.lower(labels) == "n")
    
    pos_revs = revs_np[pos_indices]
    neg_revs = revs_np[neg_indices]
    
    
    return np.array(get_all_context_4_delta_idf_scores(pos_revs, neg_revs))

#The below functions are on a review basis
def get_review_polarity_scores(rev, delta_idf_scores):
    rev_delta_idf_scores = []
    for word in rev:
        
        if word in delta_idf_scores:
            rev_delta_idf_scores.append(delta_idf_scores[word])
    return rev_delta_idf_scores

def get_reviews_polarity_scores(revs, delta_idf_scores):
    revs_delta_idf_scores = []
    for rev in revs:
        rev_delta_idf_scores = []
        for word in rev:
            if word in delta_idf_scores:
                rev_delta_idf_scores.append(delta_idf_scores[word])
            else:
                rev_delta_idf_scores.append(0.0)
        revs_delta_idf_scores.append(rev_delta_idf_scores)
    return rev_delta_idf_scores


def get_review_3_polarity_scores(rev, delta_idf_scores):
    review_polarity_scores = get_review_polarity_scores(rev, delta_idf_scores)
    
    if len(review_polarity_scores) == 0:
        return np.array([0] * 3)#np.random.rand(1).tolist()
    
    min_pol = max(review_polarity_scores)
    max_pol = min(review_polarity_scores)
    mean_pol = sum(review_polarity_scores) / len(review_polarity_scores)
    
    review_3_polarity_scores = []
    
    review_3_polarity_scores.append(min_pol)
    review_3_polarity_scores.append(max_pol)
    review_3_polarity_scores.append(mean_pol)
    
    return review_3_polarity_scores

'''
def get_all_revs_3_polarity_scores(revs):
    return [get_review_3_polarity_scores(rev, delta_idf_scores) for rev in revs]
'''

def generate_revs_with_3_polarity_scores(revs, avg_vex_matr, *labels):
    revs_three_pol_scores = get_all_revs_3_polarity_scores(revs, *labels)
    
    return np.concatenate((avg_vex_matr, revs_three_pol_scores), axis=1)

def get_all_revs_3_polarity_scores(revs, *labels):
    global delta_idf_scores
    #The below works for the training data only.
    if delta_idf_scores == None:
        delta_idf_scores = get_delta_idf_vecs(revs, *labels)  
    
    return [get_review_3_polarity_scores(rev, delta_idf_scores) for rev in revs]
 