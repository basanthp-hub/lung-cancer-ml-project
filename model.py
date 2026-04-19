import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def train_model():
    data = pd.read_csv("iris.csv")

    X = data[["sepal_length","sepal_width","petal_length","petal_width"]]
    y = data["target"]

    model = DecisionTreeClassifier()
    model.fit(X, y)

    return model