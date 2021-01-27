import numpy as np
import pandas as pd
from DimensionReduction import *
from ConvertSubConceptToCore import *
from Models import Decision_Tree

def main():
    # files
    window_matrix_file = ".\OutputDir\window_matrix.csv"
    window_terms_file = ".\OutputDir\window_matrix_terms.txt"
    sub_verb_matrix_file = ".\OutputDir\sub_verb_matrix.csv"
    subverb_terms_file = ".\OutputDir\sub_verb_matrix_terms.txt"
    abstract_concepts_file = ".\csv_docs\TP_CS_CoreConceptIn2Level.csv"
    ontology_file = ".\OutputDir\goldsorted.csv"
    
    
    print("===================================================")
    print("======= Decision Tree On sub-core concepts ========")
    print("===================================================")
    
    print(
        """
            ===================================
            ========= sub_verb_matrix =========
            ===================================
        """
    )
    
    # load the data
    X = np.loadtxt(sub_verb_matrix_file, delimiter=",", dtype=float)
    Y = eliminate_non_existing_terms(subverb_terms_file, ontology_file)
    
    # compute frequency of classes in order to reduce complexity of classification
    frequency = repartition(Y, ploting=True)
    
    # reduce dataset
    X, Y, old_labels = eliminate_non_frequent_class(X, Y, frequency, 3)
    
    # Compute Matrix Sparcity
    Sparsity(X, "sub_verb_matrix", "sub_verb_matrix Sparcity")
    print("   > Number of individual after pre-processing : {}".format(X.shape[0]))
    
    # Compute PCA
    X_pca = PCA_reduction(X, keep_n_component=3, n_print=20)
    # Compute T-SNE
    X_tsne = TSNE_reduction(X, keep_n_component=3)
    
    # Compute Decision Tree
    gs_dtree_pca = Decision_Tree(X_pca, Y, kfold=5)
    gs_dtree_tsne = Decision_Tree(X_tsne, Y, kfold=5)
    
    print("    > With PCA reduction :")
    print("       > Best estimator : \n{}".format(gs_dtree_pca.best_estimator_))
    print("       > Accuracy : {}".format(gs_dtree_pca.best_score_))
    print("    > With T-SNE reduction :")
    print("       > Best estimator : \n{}".format(gs_dtree_tsne.best_estimator_))
    print("       > Accuracy : {}".format(gs_dtree_tsne.best_score_))
    
    print(
        """
            ===================================
            ========= window_matrix =========
            ===================================
        """
    )
    
    # load the data
    X = np.loadtxt(window_matrix_file, delimiter=",", dtype=float)
    Y = eliminate_non_existing_terms(window_terms_file, ontology_file)
    
    # compute frequency of classes in order to reduce complexity of classification
    frequency = repartition(Y, ploting=True)
    
    # reduce dataset
    X, Y, old_labels = eliminate_non_frequent_class(X, Y, frequency, 3)
    
    # Compute Matrix Sparcity
    Sparsity(X, "window_matrix", "window_matrix Sparcity")
    print("   > Number of individual after pre-processing : {}".format(X.shape[0]))
    
    # Compute PCA
    X_pca = PCA_reduction(X, keep_n_component=5, n_print=20)
    # Compute T-SNE
    X_tsne = TSNE_reduction(X, keep_n_component=2)
    
    # Compute Decision Tree
    gs_dtree_pca = Decision_Tree(X_pca, Y, kfold=5)
    gs_dtree_tsne = Decision_Tree(X_tsne, Y, kfold=5)
    
    print("    > With PCA reduction :")
    print("       > Best estimator : \n{}".format(gs_dtree_pca.best_estimator_))
    print("       > Accuracy : {}".format(gs_dtree_pca.best_score_))
    print("    > With T-SNE reduction :")
    print("       > Best estimator : \n{}".format(gs_dtree_tsne.best_estimator_))
    print("       > Accuracy : {}".format(gs_dtree_tsne.best_score_))
    
    
    print("===================================================")
    print("========= Decision Tree On core concepts ==========")
    print("===================================================")
    
    print(
        """
            ===================================
            ========= sub_verb_matrix =========
            ===================================
        """
    )
    
    
    # load the data
    X = np.loadtxt(sub_verb_matrix_file, delimiter=",", dtype=float)
    Y = eliminate_non_existing_terms(subverb_terms_file, ontology_file)
    Y, core_concepts = convert_sub_concepts_to_core(Y, abstract_concepts_file)
    
    # compute frequency of classes in order to reduce complexity of classification
    frequency = repartition(Y, ploting=True)
    
    # reduce dataset
    X, Y, old_labels = eliminate_non_frequent_class(X, Y, frequency, 3)
    
    # Compute Matrix Sparcity
    Sparsity(X, "sub_verb_matrix", "sub_verb_matrix Sparcity")
    print("   > Number of individual after pre-processing : {}".format(X.shape[0]))
    
    # Compute PCA
    X_pca = PCA_reduction(X, keep_n_component=3, n_print=20)
    # Compute T-SNE
    X_tsne = TSNE_reduction(X, keep_n_component=3)
    
    # Compute Decision Tree
    gs_dtree_pca = Decision_Tree(X_pca, Y, kfold=5)
    gs_dtree_tsne = Decision_Tree(X_tsne, Y, kfold=5)
    
    print("    > With PCA reduction :")
    print("       > Best estimator : \n{}".format(gs_dtree_pca.best_estimator_))
    print("       > Accuracy : {}".format(gs_dtree_pca.best_score_))
    print("    > With T-SNE reduction :")
    print("       > Best estimator : \n{}".format(gs_dtree_tsne.best_estimator_))
    print("       > Accuracy : {}".format(gs_dtree_tsne.best_score_))
    
    print(
        """
            ===================================
            ========= window_matrix =========
            ===================================
        """
    )
    
    # load the data
    X = np.loadtxt(window_matrix_file, delimiter=",", dtype=float)
    Y = eliminate_non_existing_terms(window_terms_file, ontology_file)
    Y, core_concepts = convert_sub_concepts_to_core(Y, abstract_concepts_file)
    
    # compute frequency of classes in order to reduce complexity of classification
    frequency = repartition(Y, ploting=True)
    
    # reduce dataset
    X, Y, old_labels = eliminate_non_frequent_class(X, Y, frequency, 3)
    
    # Compute Matrix Sparcity
    Sparsity(X, "window_matrix", "window_matrix Sparcity")
    print("   > Number of individual after pre-processing : {}".format(X.shape[0]))
    
    # Compute PCA
    X_pca = PCA_reduction(X, keep_n_component=5, n_print=20)
    # Compute T-SNE
    X_tsne = TSNE_reduction(X, keep_n_component=2)
    
    # Compute Decision Tree
    gs_dtree_pca = Decision_Tree(X_pca, Y, kfold=5)
    gs_dtree_tsne = Decision_Tree(X_tsne, Y, kfold=5)
    
    print("    > With PCA reduction :")
    print("       > Best estimator : \n{}".format(gs_dtree_pca.best_estimator_))
    print("       > Accuracy : {}".format(gs_dtree_pca.best_score_))
    print("    > With T-SNE reduction :")
    print("       > Best estimator : \n{}".format(gs_dtree_tsne.best_estimator_))
    print("       > Accuracy : {}".format(gs_dtree_tsne.best_score_))


if __name__ == '__main__':
    main()

