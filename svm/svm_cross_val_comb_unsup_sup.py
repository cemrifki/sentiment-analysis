# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy import stats

from collections import OrderedDict

import numpy as np
from sklearn.model_selection import KFold


import sys
sys.path.append('../morphology')
import morphology

sys.path.append('../preprocessing')
import read_normalized


sys.path.append('../supervised_feats')
import supervised_features

sys.path.append('../const')
import constants


def run_svm_10_fold_cross():

    
    
    clf = svm.SVC(kernel='linear', C=1)

#    data = read_normalized.remove_labels_of_revs(revs)
#    labels = read_normalized.get_labels(revs)
#
#    root, surface, morphos = all_morphos(data, labels)
#    
#    top_tags = get_max_pos_tags(data, labels)
#    revs = filter_morphos(data, top_tags)


    
    tw_revs = read_normalized.generate_rev_lists(constants.TW_CORP_NAME)
    
    tw_data = np.array(read_normalized.remove_labels_of_revs(tw_revs))
    tw_labels = np.array(read_normalized.get_labels(tw_revs))
    
    tw_root, tw_surface, tw_morphos = morphology.all_morphos(tw_data.tolist(), tw_labels.tolist())

    tw_main_deltas = morphology.get_pos_tags_delta_scores(tw_data.tolist(), tw_labels.tolist())
    
 
    
    supervised_features.delta_idf_scores = None


    TOP_SENT_PERCS = [x * 0.1 for x in range(1, 10)]    
    root_accuracy = 0
    surface_accuracy = 0
    morpho_accuracy = [0] * len(TOP_SENT_PERCS)
    
    tw_avg_morpho_accuracy = [0] * len(TOP_SENT_PERCS)
    
    morpho_pers = ["morpho" + str(perc) for perc in TOP_SENT_PERCS]
    
    tw_avg_morpho_pers = ["morpho_tw" + str(perc) for perc in TOP_SENT_PERCS]
    
    
    morpho_names = ["root", "surface"] + morpho_pers + tw_avg_morpho_pers
    
    all_accuracies = [root_accuracy, surface_accuracy] + morpho_accuracy + tw_avg_morpho_accuracy

    
    
    fold_no = 10
    kf = KFold(n_splits=fold_no)
            
    
    sup_vs_uns_comb = [0.0, 0.0]
    
    revs = read_normalized.generate_rev_lists(constants.POLARITY_FILES_DIR)
    
    data = np.array(read_normalized.remove_labels_of_revs(revs))
    labels = np.array(read_normalized.get_labels(revs))
    
    root, surface, morphos = morphology.all_morphos(data.tolist(), labels.tolist())
    all_forms = [root, surface, morphos]

    keep_morpho_data = data
    # for döngüleri tersine olmak zorunda!! !

    cro_val_no = 1
    
    
    sup_scores = []
    sup_and_unsup_scores = OrderedDict({})
    
    for train, test in kf.split(data):
        print(cro_val_no)
        cro_val_no += 1
        for i in range(len(all_forms)):
            if i > 0:
                continue
            form = all_forms[i]
        
            data = np.array(form)
            X_train, X_test, y_train, y_test = data[train], data[test], labels[train], labels[test]
        
            X_train_ = X_train
            X_test_ = X_test
            #tr_morpho = morphology.morphos(keep_morpho_data[train].tolist(), labels[train].tolist())
            #X_train = np.array(tr_morpho)
            
            if i == 2:
                for j in range(len(TOP_SENT_PERCS)):
                    X_train, X_test, y_train, y_test = data[train], data[test], labels[train], labels[test]


  
                    morphology.TOP_SENT_PERC = TOP_SENT_PERCS[j]
                    
                    #top_tags, X_train = morphology.get_avg_pos_tags_delta_scores(keep_morpho_data[train], y_train, tw_main_deltas)
                    top_tags, X_train = morphology.morphos(keep_morpho_data[train], y_train)
                    #top_tags = morphology.get_max_pos_tags(keep_morpho_data[train], y_train)
     
                    #top_tags = morphology.get_max_pos_tags(keep_morpho_data[train], y_train)
                    X_test = morphology.filter_morphos(keep_morpho_data[test], top_tags)
                    #X_test = np.array(X_test)
            
                    X_train = np.array(X_train)
                    X_train = supervised_features.get_all_revs_3_polarity_scores(X_train, y_train)
                    X_train = np.array(X_train)
            
            


#                    morphology.TOP_SENT_PERC = TOP_SENT_PERCS[j]
#                    top_tags = morphology.get_max_pos_tags(keep_morpho_data[train], y_train)
#                    X_test = morphology.filter_morphos(keep_morpho_data[test], top_tags)
#
#                            
#                    X_train = supervised_features.get_all_revs_3_polarity_scores(X_train, y_train)
#                    X_train = np.array(X_train)
                    
                    
                    
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
                        
                    all_accuracies[i + j] += accuracy
            
            #        if iteration_no == 3:
            #            break
                    supervised_features.delta_idf_scores = None
                for k in range(0, len(TOP_SENT_PERCS)):
                    X_train, X_test, y_train, y_test = data[train], data[test], labels[train], labels[test]


  
                    morphology.TOP_SENT_PERC = TOP_SENT_PERCS[k]
                    
                    top_tags, X_train = morphology.get_avg_pos_tags_delta_scores(keep_morpho_data[train], y_train, tw_main_deltas)
                    #top_tags, X_train = morphology.morphos(keep_morpho_data[train], y_train)
                    #top_tags = morphology.get_max_pos_tags(keep_morpho_data[train], y_train)
     
                    #top_tags = morphology.get_max_pos_tags(keep_morpho_data[train], y_train)
                    X_test = morphology.filter_morphos(keep_morpho_data[test], top_tags)
                    #X_test = np.array(X_test)
            
                    X_train = np.array(X_train)
                    X_train = supervised_features.get_all_revs_3_polarity_scores(X_train, y_train)
                    X_train = np.array(X_train)
            
            


#                    morphology.TOP_SENT_PERC = TOP_SENT_PERCS[j]
#                    top_tags = morphology.get_max_pos_tags(keep_morpho_data[train], y_train)
#                    X_test = morphology.filter_morphos(keep_morpho_data[test], top_tags)
#
#                            
#                    X_train = supervised_features.get_all_revs_3_polarity_scores(X_train, y_train)
#                    X_train = np.array(X_train)
                    
                    
                    
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
                        
                    all_accuracies[i + j + k + 1] += accuracy
            
            #        if iteration_no == 3:
            #            break
                    supervised_features.delta_idf_scores = None

            else:
                supervised_features.use_unsuperv = False
                X_train = supervised_features.get_all_revs_3_polarity_scores(X_train_, y_train)
                
#                print(X_train)
                X_train = np.array(X_train)
                
                
                
                X_test = supervised_features.get_all_revs_3_polarity_scores(X_test_)
                X_test = np.array(X_test)
                
                
                #The below works for the training data only.    
                
                clf.fit(X_train, y_train)
            
                y_pred = clf.predict(X_test)
                
                accuracy = f1_score(y_test, y_pred, average='macro')
                
    #            print('Accuracy: ' + str(accuracy))
        #        print("F1 score: " + str(f1_score(y_test, y_pred, average="macro")))
        #        print("Precision: " + str(precision_score(y_test, y_pred, average="macro")))
        #        print("Recall: " + str(recall_score(y_test, y_pred, average="macro"))) 
                    
                all_accuracies[i] += accuracy
                sup_vs_uns_comb[0] += accuracy
                sup_scores.append(accuracy)
        
        #        if iteration_no == 3:
        #            break
                supervised_features.delta_idf_scores = None

                print(accuracy)
               
                rng = [i * 0.1 for i in range(0, 11)]
                
                for it in rng:
                    supervised_features.UNSUPERV_SCORE = it
                    
                    supervised_features.use_unsuperv = True
                    X_train = supervised_features.get_all_revs_3_polarity_scores(X_train_, y_train)
                    
    #                print(X_train)
                    X_train = np.array(X_train)
                    
                    
                    
                    X_test = supervised_features.get_all_revs_3_polarity_scores(X_test_)
                    X_test = np.array(X_test)
                    
                    
                    #The below works for the training data only.    
                    
                    clf.fit(X_train, y_train)
                
                    y_pred = clf.predict(X_test)
                    
                    accuracy = f1_score(y_test, y_pred, average='macro')
                    
                    sup_vs_uns_comb[1] += accuracy
                    
                    
                    if it not in sup_and_unsup_scores:
                        sup_and_unsup_scores[it] = []
                    sup_and_unsup_scores[it].append(accuracy)
        #            print('Accuracy: ' + str(accuracy))
            #        print("F1 score: " + str(f1_score(y_test, y_pred, average="macro")))
            #        print("Precision: " + str(precision_score(y_test, y_pred, average="macro")))
            #        print("Recall: " + str(recall_score(y_test, y_pred, average="macro"))) 
                        
                    #all_accuracies[i] += accuracy
            
            #        if iteration_no == 3:
            #            break
                    supervised_features.delta_idf_scores = None
                    print(accuracy)
                    print('___')                
    for i in range(len(all_accuracies)):    
        print("Average accuracy score for " + morpho_names[i] + ": " + str(all_accuracies[i] / fold_no))

    print("sup.: " + str(sup_vs_uns_comb[0] / fold_no))
    #print("sup + unsup.: " + str(sup_vs_uns_comb[1] / fold_no))
    
    print("All supervised: ", sup_scores)
    for k, v in sup_and_unsup_scores.items():
        print("sup + unsup - ", k, " = ", np.array(v).mean())
    
        print(stats.ttest_ind(sup_scores, v))
    
    
if __name__ == "__main__":
    run_svm_10_fold_cross()
