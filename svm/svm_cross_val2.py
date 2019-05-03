# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import numpy as np
from sklearn.model_selection import KFold


import sys
sys.path.append('../morphology')
import morphology

sys.path.append('../preprocessing')
import read_normalized


sys.path.append('../supervised_feats')
import supervised_features


def run_svm_10_fold_cross():

    clf = svm.SVC(kernel='linear', C=1)

#    data = read_normalized.remove_labels_of_revs(revs)
#    labels = read_normalized.get_labels(revs)
#
#    root, surface, morphos = all_morphos(data, labels)
#    
#    top_tags = get_max_pos_tags(data, labels)
#    revs = filter_morphos(data, top_tags)


    root_accuracy = 0
    surface_accuracy = 0
    morpho_accuracy = 0
    
    morpho_names = ["root", "surface", "morpho"]
    all_accuracies = [root_accuracy, surface_accuracy, morpho_accuracy]    
    
    
    
    fold_no = 10
    kf = KFold(n_splits=fold_no)
            
    
    
    revs = read_normalized.generate_rev_lists()
    
    data = np.array(read_normalized.remove_labels_of_revs(revs))
    labels = np.array(read_normalized.get_labels(revs))
    
    root, surface, morphos = morphology.all_morphos(data.tolist(), labels.tolist())
    all_forms = [root, surface, morphos]

    keep_morpho_data = data
    # for döngüleri tersine olmak zorunda!! !

    for train, test in kf.split(data):
        for i in range(len(all_forms)):
            form = all_forms[i]
        
            data = np.array(form)
            X_train, X_test, y_train, y_test = data[train], data[test], labels[train], labels[test]
        
            #tr_morpho = morphology.morphos(keep_morpho_data[train].tolist(), labels[train].tolist())
            #X_train = np.array(tr_morpho)
            
            if i == 2:
                
                morphology.TOP_SENT_PERC = 0.9
                
                top_tags, X_train = morphology.morphos(keep_morpho_data[train], y_train)
                #top_tags = morphology.get_max_pos_tags(keep_morpho_data[train], y_train)
 
                #top_tags = morphology.get_max_pos_tags(keep_morpho_data[train], y_train)
                X_test = morphology.filter_morphos(keep_morpho_data[test], top_tags)
                #X_test = np.array(X_test)
            
            X_train = np.array(X_train)
            X_train = supervised_features.get_all_revs_3_polarity_scores(X_train, y_train)
            X_train = np.array(X_train)
            
            
            
            X_test = supervised_features.get_all_revs_3_polarity_scores(X_test)
            X_test = np.array(X_test)
            
            
            #The below works for the training data only.    
            
            clf.fit(X_train, y_train)
        
            y_pred = clf.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
#            print('Accuracy: ' + str(accuracy))
    #        print("F1 score: " + str(f1_score(y_test, y_pred, average="macro")))
    #        print("Precision: " + str(precision_score(y_test, y_pred, average="macro")))
    #        print("Recall: " + str(recall_score(y_test, y_pred, average="macro"))) 
                
            all_accuracies[i] += accuracy
    
    #        if iteration_no == 3:
    #            break
            supervised_features.delta_idf_scores = None
    for i in range(len(all_accuracies)):    
        print("Average accuracy score for " + morpho_names[i] + ": " + str(all_accuracies[i] / fold_no))

if __name__ == "__main__":
    run_svm_10_fold_cross()