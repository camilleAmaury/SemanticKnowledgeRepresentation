import numpy as np
import pandas as pd
from DimensionReduction import *
from ConvertSubConceptToCore import *
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score, v_measure_score

def main():
    # files
    window_matrix_file = ".\OutputDir\window_matrix.csv"
    window_terms_file = ".\OutputDir\window_matrix_terms.txt"
    sub_verb_matrix_file = ".\OutputDir\sub_verb_matrix.csv"
    subverb_terms_file = ".\OutputDir\sub_verb_matrix_terms.txt"
    abstract_concepts_file = ".\csv_docs\TP_CS_CoreConceptIn2Level.csv"
    ontology_file = ".\OutputDir\goldsorted.csv"
    
    
    print("===================================================")
    print("=========== DBSCAN On sub-core concepts ===========")
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
    
    # Compute PCA
    X_pca = PCA_reduction(X, keep_n_component=2, n_print=20)
    # Compute T-SNE
    X_tsne = TSNE_reduction(X, keep_n_component=2)    
    
    # Compute DBSCAN
    db = DBSCAN(eps=1.2, min_samples=3)
    gs_dbscan_pca = db.fit_predict(X)
    gs_dbscan_tsne = db.fit_predict(X_tsne)
    
    print("> With PCA reduction:")
    print("    > Random score :", adjusted_rand_score(Y[:,1], gs_dbscan_pca))
    print("    > Macro-precision score :", homogeneity_score(Y[:,1], gs_dbscan_pca))
    print("    > Micro-precision score :", completeness_score(Y[:,1], gs_dbscan_pca))
    print("    > V-measure score :", v_measure_score(Y[:,1], gs_dbscan_pca))
    print("> With TSNE reduction:")
    print("    > Random score :", adjusted_rand_score(Y[:,1], gs_dbscan_tsne))
    print("    > Macro-precision score :", homogeneity_score(Y[:,1], gs_dbscan_tsne))
    print("    > Micro-precision score :", completeness_score(Y[:,1], gs_dbscan_tsne))
    print("    > V-measure score :", v_measure_score(Y[:,1], gs_dbscan_tsne))
    
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=gs_dbscan_pca)
    # plt.show()
    # display_clusters(gs_dbscan_pca, Y)
    
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
    
    # Compute PCA
    X_pca = PCA_reduction(X, keep_n_component=2, n_print=20)
    # Compute T-SNE
    X_tsne = TSNE_reduction(X, keep_n_component=2)
    
    # Compute DBSCAN
    db = DBSCAN(eps=1.2, min_samples=3)
    gs_dbscan_pca = db.fit_predict(X)
    gs_dbscan_tsne = db.fit_predict(X_tsne)
    
    print("> With PCA reduction:")
    print("    > Random score :", adjusted_rand_score(Y[:,1], gs_dbscan_pca))
    print("    > Macro-precision score :", homogeneity_score(Y[:,1], gs_dbscan_pca))
    print("    > Micro-precision score :", completeness_score(Y[:,1], gs_dbscan_pca))
    print("    > V-measure score :", v_measure_score(Y[:,1], gs_dbscan_pca))
    print("> With TSNE reduction:")
    print("    > Random score :", adjusted_rand_score(Y[:,1], gs_dbscan_tsne))
    print("    > Macro-precision score :", homogeneity_score(Y[:,1], gs_dbscan_tsne))
    print("    > Micro-precision score :", completeness_score(Y[:,1], gs_dbscan_tsne))
    print("    > V-measure score :", v_measure_score(Y[:,1], gs_dbscan_tsne))
    
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=gs_dbscan_pca)
    # plt.show()
    # display_clusters(gs_dbscan_pca, Y)
    
    print("===================================================")
    print("============= DBSCAN On core concepts =============")
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
    
    # Compute PCA
    X_pca = PCA_reduction(X, keep_n_component=2, n_print=20)
    # Compute T-SNE
    X_tsne = TSNE_reduction(X, keep_n_component=3)
    
    # Compute DBSCAN
    db = DBSCAN(eps=1.2, min_samples=3)
    gs_dbscan_pca = db.fit_predict(X)
    gs_dbscan_tsne = db.fit_predict(X_tsne)
    
    print("> With PCA reduction:")
    print("    > Random score :", adjusted_rand_score(Y[:,1], gs_dbscan_pca))
    print("    > Macro-precision score :", homogeneity_score(Y[:,1], gs_dbscan_pca))
    print("    > Micro-precision score :", completeness_score(Y[:,1], gs_dbscan_pca))
    print("    > V-measure score :", v_measure_score(Y[:,1], gs_dbscan_pca))
    print("> With TSNE reduction:")
    print("    > Random score :", adjusted_rand_score(Y[:,1], gs_dbscan_tsne))
    print("    > Macro-precision score :", homogeneity_score(Y[:,1], gs_dbscan_tsne))
    print("    > Micro-precision score :", completeness_score(Y[:,1], gs_dbscan_tsne))
    print("    > V-measure score :", v_measure_score(Y[:,1], gs_dbscan_tsne))
    
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
    
    # Compute PCA
    X_pca = PCA_reduction(X, keep_n_component=2, n_print=20)
    # Compute T-SNE
    X_tsne = TSNE_reduction(X, keep_n_component=2)
    
    # Compute DBSCAN
    db = DBSCAN(eps=1.2, min_samples=3)
    gs_dbscan_pca = db.fit_predict(X)
    gs_dbscan_tsne = db.fit_predict(X_tsne)
    
    print("> With PCA reduction:")
    print("    > Random score :", adjusted_rand_score(Y[:,1], gs_dbscan_pca))
    print("    > Macro-precision score :", homogeneity_score(Y[:,1], gs_dbscan_pca))
    print("    > Micro-precision score :", completeness_score(Y[:,1], gs_dbscan_pca))
    print("    > V-measure score :", v_measure_score(Y[:,1], gs_dbscan_pca))
    print("> With TSNE reduction:")
    print("    > Random score :", adjusted_rand_score(Y[:,1], gs_dbscan_tsne))
    print("    > Macro-precision score :", homogeneity_score(Y[:,1], gs_dbscan_tsne))
    print("    > Micro-precision score :", completeness_score(Y[:,1], gs_dbscan_tsne))
    print("    > V-measure score :", v_measure_score(Y[:,1], gs_dbscan_tsne))
    
    
    
    
def display_clusters(dbscan, Y):
    clusters = dict()
    nbClusters = len(set(dbscan))
    for i in range(0, nbClusters):
        clusters[i] = []
    print("NB", nbClusters)
    #get the clusters
    for j in range(Y.shape[0]):
        term = Y[j,0]
        clusterNb = dbscan[j]+1
        clusters[clusterNb].append(term)

    for cluster_id, cluster  in clusters.items():
        print("cluster: " + str(cluster_id))
        print(cluster)
    
    
    
if __name__ == '__main__':
    main()