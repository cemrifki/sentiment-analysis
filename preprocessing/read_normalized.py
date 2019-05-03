#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:56:44 2019

@author: cem
"""


import os
from collections import Counter, OrderedDict

import numpy as np
import subprocess
import re
import shutil

from random import shuffle
turk_corp_word_freqs = Counter()
eng_corp_word_freqs = Counter()

import sys
sys.path.append('../const')

import constants #0: English; #1: Turkish


rev_separator = "separatorX"
sep = os.path.sep




def get_pos_and_neg_cnt(revs):
    
    pos_cnt, neg_cnt = 0, 0
    for rev in revs:
        if rev[-1] == "P":
            pos_cnt += 1
        elif rev[-1] == "N":
            neg_cnt += 1
    return pos_cnt, neg_cnt

def tokenize_rev(rev):
    
    rev = re.sub(r"([,.;:!?\)\()]+)", r" \1 ", rev)
    
    rev = re.sub(r"([a-zA-ZçöşığüÇÖŞİĞÜ+)(\'[a-zA-ZçöşığüÇÖŞIĞÜ]+)", r"\1", rev)
    
    #emoticon handling
    rev = re.sub(r"(:) (\)+)", r"\1\2", rev)
    rev = re.sub(r"(:) (\(+)", r"\1\2", rev)
    
    rev = re.sub(r"(\() (:+)", r"\1\2", rev)
    rev = re.sub(r"(\)) (:+)", r"\1\2", rev)
    
    
    rev = rev.rstrip()
    if constants.LANG == 1:
        if rev_separator not in rev:
            rev = rev + " " + rev_separator
    
    rev = re.sub(r"[ ]+", " ", rev)
    
    return rev

def tokenize_revs_and_update_files(inp_folder_path):
    
    
    
    if not os.path.exists(inp_folder_path):
        raise FileNotFoundError
    
    os.chdir(inp_folder_path)
    
    pol_revs = {}
    
    for file in files("."):#os.listdir(inp_folder_path):
    
        revs = []
        with open(file, "r") as f:
            for line in f:
                tokenized = tokenize_rev(line)
                revs.append(tokenized)
                
        pol_revs[file] = revs
        
        

        with open(file, "w") as f:
            f.truncate(0)
            for rev in revs:
                f.write(rev + "\n")
            revs = []
          
    go_up_directories(len(constants.LANG_CORP_NAME))
            
def go_up_directories(num):
      
    dir_ = str(os.path.sep).join([".."] * num)
    os.chdir(dir_)
    
    
def up_directory_str(num):
    dir_ = str(os.path.sep).join([".."] * num)
    return dir_
            

def replace_last(pat, sub, str_):
    k = str_.rfind(pat)
    return str_[:k] + sub + str_[k+1:]

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

def parse_and_disamb_turk_revs(file_path):
    
    
    tokenize_revs_and_update_files(file_path)
    
    #corp_name = file_path.split(os.sep)[-1] 
    
    os.chdir(file_path)
    if not os.path.exists("parsed"):
        os.makedirs("parsed")
    if not os.path.exists("disamb"):
        os.makedirs("disamb")
       
    
    dir_up = up_directory_str(len(constants.LANG_CORP_NAME))
     

    to_be_parsed_dir = os.path.join(dir_up, "Turkce Morphological Analyzer-Disambiguator",
                                    "MP", "files")
    if not os.path.exists(to_be_parsed_dir):
        os.makedirs(to_be_parsed_dir)
        

    
    
    for file in files("."):
        shutil.copyfile(file, os.path.join(to_be_parsed_dir, file))
        

    os.chdir(os.path.join(to_be_parsed_dir, ".."))    
    
    if not os.path.exists("parsed"):
        os.makedirs("parsed")
        
    for file in files("files"):
        
        
        #subprocess.call(["python2.7", "parse_corpus.py", "files" + sep + file, ">", "parsed" + sep + file])

        parsed_file_name = replace_last(".", ".parsed.", file)
        subprocess.call("python2.7 parse_corpus.py files" + sep + file + " > " + " parsed" + sep + parsed_file_name, shell=True)
        
        #subprocess.call("parse_corpus.py " + file + " > " + "parsed" + os.path.sep + replace_last(".", ".parsed.", file))
        #os.system("parse_corpus.py " + file + " > " + "parsed" + os.path.sep + replace_last(".", ".parsed.", file)) 
        
    
       
    md_file_dir = os.path.join("..", "MD-2.0") + os.path.sep + "parsed"
    if  os.path.exists(md_file_dir):
        os.system("rm -rf " + md_file_dir)


    os.system("mv parsed " + os.path.join("..", "MD-2.0") + os.path.sep)
    
    os.system("rm -rf files")
    
    os.chdir(os.path.join("..", "MD-2.0"))



    print(os.getcwd())
    
    for file in os.listdir("parsed"):
        disamb_name = file.replace("parsed", "disamb")
        
        #os.system("perl md.pl -disamb model.txt " + file + os.path.join("..", "..", file_path, disamb_name))
        
        #subprocess.call(["perl", "md.pl", "-disamb", "model.txt", file, os.path.join("..", "..", file_path, disamb_name)])
        
        
        #subprocess.call("perl md.pl -disamb model.txt parsed" + sep + file + " " + os.path.join("..", "..", file_path, "disamb", disamb_name))

        subprocess.call(["perl", "md.pl", "-disamb", "model.txt", "parsed" + sep + file, os.path.join("..", "..", file_path, "disamb",disamb_name)])


    os.chdir(os.path.join("..", ".."))
    
    #return out_disamb




def generate_morpho_turk_rev_lists(file_path): #data/turk/movie
    """
    Polarity files should end in .pos or .neg.
    """
    
    fnd = False
#    print(os.getcwd())
#    print(file_path)
    
    for file in os.listdir(file_path):
        if file == "disamb":
            fnd = True
    if not fnd:
        parse_and_disamb_turk_revs(file_path)
    

    

    revs = []
    rev = []
    pol = ""

    disamb_fold = file_path + sep + "disamb"
    os.chdir(disamb_fold)
    
    for file in os.listdir("."):
        
        
        if "pos" in file:
            pol = "P"
        elif "neg" in file:
            pol = "N"
        with open(file, "r") as f:
            
                
            for line in f:

                if "@url" in line or "@hashtag" in line or "@mention" in line:
                    continue

                if "[" not in line:
                    continue


                if line[0].isspace():
                    continue

                if "smiley" in line:
                    emot = re.search(r'\[(.*?)\]', line).group(1)
                    line = emot + " " + emot + "[Unknown]"
                
                line_spl = line.split()
                
                root = ' '.join(line_spl[0:2])#line_spl_first_morph.split("[")[0]
        
        
                turk_corp_word_freqs[root] += 1 
        
                
                
                if rev_separator in line:
                    rev.append(pol)
                    revs.append(rev)
                    rev = []
                    continue
                
                rev.append(root)

    
    #revs = get_list_except_least_freq(turk_corp_word_freqs, revs)
    #revs = get_list_except_most_and_least_freq(turk_corp_word_freqs, revs)

    # Düzgün kod yazarken aşağıdakini değiştir
    go_up_directories(len(constants.LANG_CORP_NAME) + 1)
    return revs


def generate_turk_rev_lists(file_path): #data/turk/movie
    """
    Polarity files should end in .pos or .neg.
    """
    
    fnd = False
    for file in os.listdir(file_path):
        if file == "disamb":
            fnd = True
    if not fnd:
        parse_and_disamb_turk_revs(file_path)
    

    

    revs = []
    rev = []
    pol = ""

    disamb_fold = file_path + sep + "disamb"
    os.chdir(disamb_fold)
    for file in os.listdir("."):
        
        
        if "pos" in file:
            pol = "P"
        elif "neg" in file:
            pol = "N"
        with open(file, "r") as f:
            
                
            for line in f:
                if "[" not in line:
                    continue

                if line[0].isspace():
                    continue
                
                line_spl = line.split()
                
                line_spl_first_morph = line_spl[1]
                root = line_spl_first_morph.split("[")[0]
        
                print(root)
        
                turk_corp_word_freqs[root] += 1 
        
                
                if len(rev) > 0:
                    if "değil" in line or "yok" in line:
                        if rev[-1][-1] == "_":
                            rev[-1] = rev[-1][:-1] 
                        else:
                            rev[-1] += "_"
                        continue
                    elif "[Neg]" in line_spl[1]:
                        root += "_"
                
                if rev_separator in line:
                    rev.append(pol)
                    revs.append(rev)
                    rev = []
                    continue
                
                rev.append(root)

    
    #revs = get_list_except_most_and_least_freq(turk_corp_word_freqs, revs)

    # Düzgün kod yazarken aşağıdakini değiştir
    go_up_directories(len(const.constants.LANG_CORP_NAME) + 1)
    return revs
                
def generate_eng_rev_lists(file_path): #data/eng/movie
    """
    Polarity files should end in .pos or .neg.
    """
    
    revs = []
    pol = ""

    os.chdir(file_path)

    for file in os.listdir("."):
        
        if "pos" in file:
            pol = "P"
        elif "neg" in file:
            pol = "N"
        with open(file, "r") as f:
            
                
            for line in f:
                line = line.lower()
                line_spl = tokenize_rev(line).split()
                line_spl.append(pol)
                
                for word in line_spl:
                    eng_corp_word_freqs[word] += 1
                revs.append(line_spl)
                
    revs = remove_noise(revs)
    #revs = get_list_except_most_and_least_freq(eng_corp_word_freqs, revs)

    go_up_directories(3)
    return revs

def generate_rev_lists(corp_path):
    #corp_path = constants.POLARITY_FILES_DIR
    revs = []
    if constants.LANG == 0:
        revs = generate_eng_rev_lists(corp_path)
    else:
        revs = generate_morpho_turk_rev_lists(corp_path)
    shuffle(revs)
    
    return revs
        
def remove_labels_of_revs(revs):
    return [rev[:-1] for rev in revs]

def get_labels(revs):
    labels = []
    for rev in revs:
        labels.append(rev[-1])
    return labels
#    return [rev[-1] for rev in revs]

#last_index is sentiment; therefore, discard it
#delta idf mantığı çok daha iyi olurdu. Ya da, en sık olanlardan "the"
#gibi artikelleri veya punctuation'ları ele sadece, nltk'i kullanarak.
def remove_noise(revs):
    
    cnt = Counter()
    
    for rev in revs:
        rev_words = set(rev)
        for word in rev_words:
            cnt[word] += 1
            
    
    min_count = (1 / 1000) * len(revs)
    max_count = (1 / 100) * len(revs)
    
    res_revs = []
    
    for rev in revs:
        new_rev = []
        
        for i in range(0, len(rev) - 1):
            word = rev[i]
            if cnt[word] > min_count and cnt[word] < max_count:
                new_rev.append(word)
        
        if len(new_rev) == 0:
            continue
        new_rev.append(rev[-1])
        
        res_revs.append(new_rev)
        
        
    return res_revs


def remove_noise_from_dict(dict_):
    
    cnt = Counter()
    
    for k, v in dict_.items():
        vals = set(v)
        for val in vals:
            cnt[val] += 1
            
    
    #min_count = (1 / 1000) * len(dict_)
    max_count = (1 / 100) * len(dict_)
    
    res_dict = {}
    
    for k, v in dict_.items():
        new_v = []
        
        for word in v:
            if cnt[word] < max_count:
                new_v.append(word)
        res_dict[k] = new_v
        
        
    return res_dict



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


def get_most_and_least_freq_words(frequency_dict_):
    
    size = len(frequency_dict_)
    frequency_el = int(size / 100)
    
    
    
    most_freq = frequency_dict_.most_common(frequency_el)
    least_freq = frequency_dict_.most_common()[:-frequency_el-1:-1]


    union_set = set([k[0] for k in most_freq]).union(set(k[0] for k in least_freq))


    return union_set


def get_list_except_least_freq(frequency_dict, revs):
    
    noise_thr = int(len(revs) / 1000)
    
    res_revs = []
    
    for rev in revs:
        new_rev = []
        for i in range(0, len(rev) - 1):
            word = rev[i]
            if frequency_dict[word] > noise_thr:
                new_rev.append(word)
        new_rev.append(rev[-1])
        res_revs.append(new_rev)
        
    return res_revs


def get_list_except_most_and_least_freq(frequency_dict_, revs):
    
    most_and_least_freq_words = get_most_and_least_freq_words(frequency_dict_)
    
    
    res_revs = []
    
    for rev in revs:
        new_rev = []
        for word in rev:
            if word not in most_and_least_freq_words:
                new_rev.append(word)
        res_revs.append(new_rev)
        
    return res_revs

def get_dict_except_most_and_least_freq(frequency_dict_, dict_defs):
    
    
    most_and_least_freq_words = get_most_and_least_freq_words(frequency_dict_)
    
    res_dict = {}
    
    for word, dict_def in dict_defs.items():
        res_dict[word] = []
        for dict_def_word in dict_def:
            if dict_def_word not in most_and_least_freq_words:
                res_dict[word].append(dict_def_word)
            
    return res_dict

def get_corp_vocab(revs):
    
    return set([word for rev in revs for word in rev])


def generate_keys_and_np_matr_from_dict(dict_):
    
    keys = list(dict_.keys())
    
    matr = dict_.values()
    
    matr = np.array(list(matr))
    
    return keys, matr


if __name__ == "__main__":
    
    
    revs = generate_rev_lists(constants.POLARITY_FILES_DIR)

'''
from string import punctuation
import pandas as pd

# 1. Read and preprocess titles from HN posts.
punctrans = str.maketrans(dict.fromkeys(punctuation))
def tokenize(title):
    x = title.lower() # Lowercase
    x = x.encode('ascii', 'ignore').decode() # Keep only ascii chars.
    x = x.translate(punctrans) # Remove punctuation
    return x.split() # Return tokenized.

t0 = time()
df = pd.read_csv(inpFileName, usecols=['title'], encoding='utf-8') #../input/HN_posts_year_to_Sep_26_2016.csv
texts_tokenized = df['title'].apply(tokenize)
print(texts_tokenized[:10])
print('%.3lf seconds (%.5lf / iter)' % (time() - t0, (time() - t0) / len(df)))

'''
