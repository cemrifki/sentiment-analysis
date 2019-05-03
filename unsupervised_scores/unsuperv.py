#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 23:01:55 2019

@author: cem
"""

#wordPairs = {};//{{"tatlı", "acı"}, {"harika", "berbat"}, {"sevgi", "nefret"}, {"olumlu", "olumsuz"}, {"başyapıt", "vasat"}, {"güzel", "çirkin"}, {"doğru", "yanlış"}, {"zevkli", "sıkıcı"}, {"iyi", "kötü"}, {"sevimli", "sevimsiz"},};
import os


def get_unsupervised_scores():

    unsupervised_scores = {}

    word_pair_cnt = 0
    for filename in os.listdir("."):
        word_pair_cnt += 1
        if filename.endswith("txt"):
            with open(filename, "r") as f:
                for line in f:
                    word = line.split("->")[0][1:-1]
                    res = float(line.split("result:")[1])
                    if word not in unsupervised_scores:
                        unsupervised_scores[word] = res
                    else:
                        unsupervised_scores[word] += res
    for word in unsupervised_scores:
        unsupervised_scores[word] /= word_pair_cnt
    return unsupervised_scores

if __name__ == "__main__":
    uns_scores = get_unsupervised_scores()
    print(len(uns_scores))