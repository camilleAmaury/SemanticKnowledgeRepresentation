import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd

def main():
    #load the data and labels
        #load the data and labels
    X = np.loadtxt(".\OutputDir\sub_verb_matrix.csv", delimiter=",", dtype=float)
    print(X)
    
    from collections import defaultdict
    temp = set()
    with open(".\OutputDir\sub_verb_matrix_terms.txt", "r") as f:
        for line in f:
            temp.add(line.replace("\n", ""))
    terms_labeled = pd.read_csv(".\OutputDir\goldsorted.csv", sep=",", header=0)["Term"]
    indices = list(range(terms_labeled.shape[0]))
    df = set(terms_labeled)
    
    to_keep = []
    for i,w in enumerate(df):
        if w in temp:
            to_keep.append(i)
    indices = to_keep
    
    Y = pd.read_csv(".\OutputDir\goldsorted.csv", sep=",", header=0).values[indices, 1:]
    
    terms_labeled = pd.read_csv(".\OutputDir\goldsorted.csv", sep=",", header=0).values[indices,2]
    #print(len(set(terms_labeled)))
    
    
    # plot of repartition
    dico = defaultdict(int)
    for label in terms_labeled:
        dico[label] += 1
    
    threshold = 5
    indices = []
    classes = []
    for k in dico:
        if dico[k] >= threshold:
            classes.append(k)
    for i in range(Y.shape[0]):
        if Y[i,1] in classes:
            indices.append(i)
    Y = Y[indices, :]
    dic_new_label = {}
    for i, oldlabel in enumerate(classes):
        dic_new_label[oldlabel] = i
        
    for i in range(Y.shape[0]):
        Y[i,1] = dic_new_label[Y[i,1]]
        
    X = X[indices, :]

    #split the data between training and testing
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    pca = PCA(n_components=20)
    pca.fit(X)
    # plt.plot(pca.explained_variance_ratio_)
    # plt.xticks(range(len(pca.explained_variance_ratio_)))
    
    pca = PCA(n_components=4)
    X = pca.fit_transform(X)
    
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    
    # data_train, data_test, labels_train, labels_test, couples_train, couples_test = train_test_split(X, Y[:, 1].astype(int), Y[:, 0], test_size=0.20)
    # print(couples_test)


    #build the Knn model
    from sklearn.tree import DecisionTreeClassifier
    dtree = DecisionTreeClassifier()
    # dtree.fit(data_train, labels_train)
    
    scores = cross_val_score(dtree, X, Y[:,1].astype(int), scoring='accuracy', cv=cv, n_jobs=-1)
    print(np.mean(scores),scores)

    # #predict for the testing data
    # y = dtree.predict(data_test)
    # #pring the results
    # print ("accuracy:")
    # print (accuracy_score(labels_test, y))
    # i = 0
    # for couple in couples_test:
    #     if int(y[i]) - int(labels_test[i]) == 0:
    #         print(couple + ", " + str(labels_test[i]) + ", " + str(y[i]))
    #     else:
    #         print(couple + ", " + str(labels_test[i]) + ", " + str(y[i]))
        # i += 1


if __name__ == '__main__':
    main()

