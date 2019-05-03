#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:18:37 2019

@author: cem
"""

#Kodu yetiştiremedim Cambridge için.

#Tunga hocaya söylenecekler:
#Handled negation
#pos. cnt = neg. cnt (Neden, açıkla)
#Twitter & Movie (ama eşit ağırlıklı; corpus size'larına ayrı ayrı doğru orantılı değil)

#Negation'ı kaldır sonradan
#Morpheme'leri sort ettim.

# (I) ve (II)'leri handle et! !!
import re
from collections import Counter
import math

import sys


sys.path.append('../supervised_feats')
import supervised_features

sys.path.append('../preprocessing')

import read_normalized

sys.path.append('../const')

import constants #0: English; #1: Turkish


POS_PATTERN = r'\[(.*?)\]'

import numpy as np

TOP_SENT_PERC = 1.0

#Use TOP_SENT_PERC.
def get_equal_top_pos_and_neg(deltas):
    
    positive_pos = {}
    negative_pos = {}
    
    for k, v in deltas.items():
        if v >= 0.0:
            positive_pos[k] = v
        else:
            negative_pos[k] = v
    positive_pos = sorted(positive_pos.items(), key=lambda kv: abs(kv[1]), reverse=True)
    negative_pos = sorted(negative_pos.items(), key=lambda kv: abs(kv[1]), reverse=True)


    min_sent_pos = min(len(positive_pos), len(negative_pos))
    top_score_size = int(TOP_SENT_PERC * min_sent_pos)

#    positive_pos = dict(itertools.islice(positive_pos.items(), top_score_size))
#    negative_pos = dict(itertools.islice(negative_pos.items(), top_score_size))
    
    positive_pos = conv_list_2_dict(positive_pos[:top_score_size])
    negative_pos = conv_list_2_dict(negative_pos[:top_score_size])

    
    return set(positive_pos.keys()).union(set(negative_pos.keys()))
    
    
    #return {**positive_pos, **negative_pos}

def conv_list_2_dict(l):
    d = {}
    for it in l:
        d[it[0]] = it[1]
    return d




def get_top_sent_tags(data, labels):

    #data = read_normalized.remove_labels_of_revs(revs)
    #labels = read_normalized.get_labels(revs)

    pos_deltas = get_pos_tags_delta_scores(data, labels)

    top_delta_pos = get_equal_top_pos_and_neg(pos_deltas)

    return top_delta_pos
    


def split_into_pos_tags(morpho_analysed):
    all_pos_tags_of_a_word = re.findall(POS_PATTERN, morpho_analysed)
    return all_pos_tags_of_a_word
    
#değil, yok; bunları da hesaba kat
def get_pos_tags_from_a_rev(rev):
    rev_pos_tags = set({})
    for word in rev:
        spl = split_into_pos_tags(word)
        neg_concat = ""
        if "Neg" in spl:
            neg_concat = "_"
        for pos in spl: 
            if pos == "Neg":
                rev_pos_tags.add(pos)
            else:
                rev_pos_tags.add(pos + neg_concat)
    return list(rev_pos_tags)    

'''
def get_pos_tags_from_revs(revs):
    all_revs_pos_tags = []
    for rev in revs:
        rev_pos_tags = get_pos_tags_from_a_rev(rev)
        all_revs_pos_tags.append(rev_pos_tags)
    return all_revs_pos_tags
'''
sents = ["P", "N"]
def map_rev_pos_tags_2_pol(rev, pol):
    pos_pols = {}
    rev_pos_tags = get_pos_tags_from_a_rev(rev)
    
    '''
    if "Neg" in rev:
        ind = (sents.index(pol) + 1) & 1
        pol = sents[ind]
    ''' 
    for pos in rev_pos_tags:
        if pos is "Neg":
            continue
        if pos not in pos_pols:
            pos_pols[pos][pol] = Counter()
        if pos[-1] == "_":
            shifted_pol = sents[(sents.index(pol) + 1) & 1]
            pos_pols[pos][shifted_pol] += 1
        else:
            pos_pols[pos][pol] += 1
    return pos_pols
   
#NEGATION (SONUNA _ EKLENMESİ) NE ALAKA!?!?!?!??!?!
def map_revs_pos_tags_2_pols(revs, pols):
    
    positive_cnts = {}
    negative_cnts = {}
    
    all_cnts = [positive_cnts, negative_cnts]
    
    ind = 0
    for rev in revs:
        pol = pols[ind]
        
        sent_ind = sents.index(pol)
        #sent_cnts = all_cnts[sent_ind]
            
        pos_tags = get_pos_tags_from_a_rev(rev)
        for pos_tag in pos_tags:
            if pos_tag[-1] == "_":
                pos_tag = pos_tag[:-1]
                sent_cnts = all_cnts[(sent_ind + 1) & 1]
                if pos_tag not in sent_cnts:
                    sent_cnts[pos_tag] = 0
                sent_cnts[pos_tag] = sent_cnts[pos_tag] + 1
            else:
                sent_cnts = all_cnts[sent_ind]
                if pos_tag not in sent_cnts:
                    sent_cnts[pos_tag] = 0
            
                sent_cnts[pos_tag] = sent_cnts[pos_tag] + 1
            
        ind += 1
    
    return all_cnts

# Yok, değil, vs.'yi de hesaba kat
def root(revs):
    '''
    Include the second only
    '''
    root_revs = []
    for rev in revs:
        root_rev = []
        
        len_rev = len(rev)
        for i in range(0, len_rev):
            token = rev[i]
            
            if i > 0:
                if "değil" in token or "yok" in token:
                    if root_rev[-1][-1] == "_":
                        root_rev[-1] = root_rev[-1][:-1] 
                    else:
                        root_rev[-1] += "_"
                
            token_root = token.split()[1][:token.split()[1].index("[")]
            if "[Neg]" in token:
                token_root += "_"
            root_rev.append(token_root)
        root_revs.append(root_rev)
    return root_revs

def surface(revs):
    '''
    Include the first only
    '''
    surf_revs = []
    for rev in revs:
        surf_rev = []
        for token in rev:
            token_surf = token.split()[0]
            surf_rev.append(token_surf)
        surf_revs.append(surf_rev)
    return surf_revs
    
#Top 20 percent, top 30 percent, top 50 percent, top 80 percent; try them all
def morphos(data, *labels):
    top_morphemes = get_max_pos_tags(data, *labels)
    filtered_revs = filter_morphos(data, top_morphemes)
    
    return top_morphemes, filtered_revs
    '''
    Include both
    
    return 1. training revs, 2. top morphemes
    '''

#Negation handling'i hocaya anlat; sonuna '_' koymayız; önceki aşamnada - ile çarptığımız için zaten handling etniş oluruz.
#Birinci POS tag'i yok say!!
#Yok, değil gibi kelimelerde "Neg" var mı; bir kontrol et. Varsa ona göre düzenle.
def filter_morphos(revs, top_morphos):
    filtered_revs = []
    for rev in revs:
        modified_rev = []
        for i in range(len(rev)):
            word = rev[i]
            morpho_analysed = word.split()[1]
            root = morpho_analysed.split("[")[0]
            all_pos_tags = re.findall(POS_PATTERN, morpho_analysed)[1:]
            common_pos_tags = set(all_pos_tags) & set(top_morphos)
            common_pos_tags = sorted(common_pos_tags)
            word_w_top_morphemes = root + ''.join(common_pos_tags)
            
            if i > 0:
                if "değil" in root or "yok" in root:
                    if modified_rev[-1][-1] == "_":
                        modified_rev[-1] = modified_rev[-1][:-1] 
                    else:
                        modified_rev[-1] += "_"
                    continue
                
            if "[Neg]" in all_pos_tags:
                word_w_top_morphemes += "_"
                
            
            modified_rev.append(word_w_top_morphemes)
        filtered_revs.append(modified_rev)
    return filtered_revs

def all_morphos(revs, *pol):
    return root(revs), surface(revs), morphos(revs, *pol)[1]    

def test_morphos(revs, top_morphos):
    return root(revs), surface(revs), filter_morphos(revs, top_morphos)




def get_all_morpheme_tags(pol_list):

    all_morph = set([])
    for l in pol_list:
        all_morph = all_morph.union(l.keys())
        
    return all_morph
#NORMALİZE ETMEK ZORUNDASIN!!
def get_pos_tags_delta_scores(revs, pols):
        
    pos_size, neg_size = read_normalized.get_pos_and_neg_cnt(pols)
    
    #print(pos_size, neg_size)
    
    pos_2_sents = map_revs_pos_tags_2_pols(revs, pols)
    delta_idf_scores = {}

    pos_counts = pos_2_sents[0]
    neg_counts = pos_2_sents[1]
    
    
    all_morph = get_all_morpheme_tags(pos_2_sents)
    
    for morph in all_morph:
        
        pos_val = 0
        neg_val = 0
        if morph in pos_counts:
            pos_val = pos_counts[morph]
        if morph in neg_counts:
            neg_val = neg_counts[morph]
        delta_idf_scores[morph] = math.log((pos_val / pos_size + 0.001) /
                              (neg_val / neg_size + 0.001))
    supervised_features.delta_idf_scores = None
    return delta_idf_scores

def sign(x): return 1 if x >= 0 else -1
#For two corpora: INCOMPLETE!!
def get_avg_pos_tags_delta_scores(main_corp_data, main_corp_labels, other_corp_tag_deltas):
    
    new_tag_scores = {}
    
    

    main_deltas = get_pos_tags_delta_scores(main_corp_data, main_corp_labels)
    
    for tag, score in main_deltas.items():
        if tag in other_corp_tag_deltas:
            other_corp_score = other_corp_tag_deltas[tag]
            new_score = (score + other_corp_score) / 2.0
#            if sign(score) != sign(other_corp_score):
#                new_score = 0
#            else:
#                new_score = (score + other_corp_score) / 2.0
        else:
            new_score = score
        new_tag_scores[tag] = new_score
        
    top_avg_tags = get_equal_top_pos_and_neg(new_tag_scores)
    filtered_revs = filter_morphos(main_corp_data, top_avg_tags)
    
#    top_avg_tags = get_equal_top_pos_and_neg(other_corp_tag_deltas)
#    filtered_revs = filter_morphos(main_corp_data, top_avg_tags)
    
    
    return top_avg_tags, filtered_revs
        

def get_max_pos_tags(data, labels):
#    revs = read_normalized.generate_rev_lists()
    max_sent_pos_tags = get_top_sent_tags(data, labels)
    return max_sent_pos_tags


morphological = [
        root,
        surface,
        morphos]

if __name__ == "__main__":
    revs = read_normalized.generate_rev_lists(constants.POLARITY_FILES_DIR)
    data = read_normalized.remove_labels_of_revs(revs)
    labels = read_normalized.get_labels(revs)

    root, surface, morphos = all_morphos(data, labels)
    
    top_tags = get_max_pos_tags(data, labels)
    revs = filter_morphos(data, top_tags)
#    print(revs)
#    print(root)
#    print(surface)
#    print(morphos)
    pass#get_revs_and_max_pos_tags()

