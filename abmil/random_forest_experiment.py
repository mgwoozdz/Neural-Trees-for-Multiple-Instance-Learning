from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from models.random_forest.forest import RandomForest
from models.random_forest.tree import DecisionTree

breast = datasets.load_breast_cancer()
X, y = breast.data, breast.target
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

clf = RandomForestClassifier(criterion="entropy", oob_score=True)
clf.fit(X_train, y_train)
print("Sklearn RandomForestClassifier score: ", clf.score(X_test, y_test))


tree = DecisionTree()
tree.fit(X_train, y_train)
y = tree.predict(X_test)

print("Random tree oob score: ", tree.oob_score)
print("Random tree score: ", sum(y == y_test) / len(y_test))

rf = RandomForest(estimators_num=5)
rf.fit(X_train, y_train)
print("Random forest oob score: ", rf.oob_score)
y = rf.predict(X_test)
print("Random forest score: ", sum(y == y_test) / len(y_test))
