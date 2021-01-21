from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

"""
    Function which compares a matrix_term_file with the labeled ontology and delete the non-existing of the first in the second file.
    example : matrix_term_file = {x;0} and labeled = {x:0, y:1} then returns {x;0}
"""
def eliminate_non_existing_terms(matrix_term_file, terms_labeled_file):
    # ===> Compose the set of the two files <=== #
    # ==> Matrix Term <== #
    matrix_set = set()
    with open(matrix_term_file, "r") as f:
        for line in f:
            matrix_set.add(line.replace("\n", ""))
    # ==> Ontology <== #
    Y = pd.read_csv(terms_labeled_file, sep=",", header=0)
    ontology_set = set(Y["Term"])
    
    # ===> Look over the dataset if every word exist in both datasets  <=== #
    indices_to_keep = []
    for i, w in enumerate(ontology_set):
        if w in matrix_set:
            indices_to_keep.append(i)
    
    # ===> Eliminate the non-found words  <=== #
    return Y.values[indices_to_keep, 1:]


"""
    Function which computes the frequency of classes and plot it
"""
def repartition(Y, ploting=False):
    # ===> Compute frequency <=== #
    frequency = defaultdict(int)
    for label in Y[:,1]:
        frequency[label] += 1
        
    # ===> Plot frequency <=== #
    if ploting:
        plt.bar([k for k in frequency], height=[frequency[k] for k in frequency], width=0.8)
        plt.xlabel('Class')
        plt.ylabel('Frequency') 
        plt.title("Class Frequency")
        plt.show()
    
    # ===> Returns frequency <=== #
    return frequency

"""
    Function which truncate dataset regarding a threshold of class frequency, and returns the new class label as a dictionnary.
"""
def eliminate_non_frequent_class(X, Y, frequency, threshold = 1):
    # ===> Compose the set of the two files <=== #
    # ==> Keeping frequency higher or equal to threshold <== #
    indices = []
    classes = []
    for k in frequency:
        if frequency[k] >= threshold:
            classes.append(k)
    # ==> Keeping indices <== #
    for i in range(Y.shape[0]):
        if Y[i,1] in classes:
            indices.append(i)
    # ==> Getting the function of old label into new labels <== #
    dic_new_label = {}
    for i, oldlabel in enumerate(classes):
        dic_new_label[oldlabel] = i
    
    # ==> Changing old labels <== #
    Y = Y[indices, :]
    for i in range(Y.shape[0]):
        Y[i,1] = dic_new_label[Y[i,1]]
        
    
    return X[indices, :], Y, {v: k for k, v in dic_new_label.items()}


def Sparsity(M, title, titleplot):
    print(" > Matrix Sparsity of {} = {}" .format(title, 1.0 - np.count_nonzero(M) * 1.0 / M.size))
    plt.matshow(M)
    plt.title(titleplot)
    plt.show()

"""
    Function which computes PCA
"""
def PCA_reduction(X, keep_n_component=2, n_print=-1):
    # ===> Compute the pca and reduce dimensions <=== #
    # ==> PCA plot <== #
    if n_print != -1 and n_print > 1:
        pca = PCA(n_components=n_print)
        pca.fit(X)
        plt.plot(pca.explained_variance_ratio_)
        plt.xticks(range(len(pca.explained_variance_ratio_)))
        plt.xlabel("Dimension of the PCA")
        plt.ylabel("Variance Explained")
        plt.title("Explained variance of the PCA's dimension")
        plt.show()
    
    # ==> PCA <== #
    pca = PCA(n_components=keep_n_component)
    return pca.fit_transform(X)

"""
    Function which computes TSNE
"""
def TSNE_reduction(X, keep_n_component=2):
    # ===> Compute the TSNE and reduce dimensions <=== #
    return TSNE(n_components=keep_n_component).fit_transform(X)
