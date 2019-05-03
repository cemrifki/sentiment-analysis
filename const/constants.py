# -*- coding: utf-8 -*-
import os

import sys
sys.path.append('../morphology')

import morphology

REDUCED_DIM_SIZE = 200 #10000 filan da olmasın abuk sabuk


#TEK SATIR VEYA İKİ SATIR OLMAK ZORUNDA, DİL İLE İLGİLİ PARAMETRE!!
#Command line argument bir .csv olsun sadece sonradan
LANG = 1 #0: English, #1: Turkish

CORPORA = ["movie", "twitter"]
CORP_IND = 0
LANG_CORP_NAME = ["data", "turk", CORPORA[CORP_IND]]

TW_CORP_NAME = "data/turk/" + str(CORPORA[(CORP_IND + 1) & 1])
if LANG == 0:
    LANG_CORP_NAME[1] = "eng"
else:
    LANG_CORP_NAME[1] = "turk"

POLARITY_FILES_DIR = os.path.sep.join(LANG_CORP_NAME)

CONTEXT_SIZE = 5

EMBED_APPROACH_IND = 3

#Aşağıdaikini silip yukarıdakini 0-9 arasında bir değer yap; daha iyi olur.
USE_3_REV_POL_SCORES = True


MORPHO = 2 #0: root, 1: surface, 2: partial surface

#MORPHOLOGICAL_FUNCTION = [morphology.root, morphology.surface, morphology.morphos]