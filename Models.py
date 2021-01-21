from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier


def Decision_Tree(X, Y, kfold=5):
    cv = KFold(n_splits=kfold, random_state=1, shuffle=True)
    params = {
        'criterion'     : ["gini", "entropy"],
        'splitter'      : ["best", "random"],
        'max_depth'     : [2,3,5,8,10,15,20],
        'random_state'  : [1]
    }
    gs_dtree = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=params, cv=cv)
    gs_dtree.fit(X, Y[:,1].astype(int))
    return gs_dtree