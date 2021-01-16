from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    # nbClusters = 5 #number of clusters
    #load the data
    X = np.loadtxt(".\OutputDir\window_matrix.csv", delimiter=",")
    #data shuffling
    # np.random.shuffle(X)
    #apply k-means
    db = DBSCAN(eps=7, min_samples=2)
    dbscan = db.fit_predict(X)
    print(dbscan)
    plt.figure()
    X = PCA(n_components=2).fit_transform(X)
    plt.scatter(X[:, 0], X[:, 1], c=dbscan)
    plt.show()
    clusters = dict()
    nbClusters = max(db.labels_)+2
    for i in range(0, nbClusters):
        clusters[i] = []
    i = 0
    #get the clusters
    with open(".\OutputDir\window_matrix_terms.txt", "r") as f:
        for line in f:
            term = line.strip()
            clusterNb = dbscan[i]+1
            i += 1
            clusters[clusterNb].append(term)
    for cluster_id, cluster  in clusters.items():
        print("cluster: " + str(cluster_id))
        print(cluster)

if __name__ == '__main__':
    main()