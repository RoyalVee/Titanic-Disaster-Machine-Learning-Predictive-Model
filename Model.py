"""
this model uses Decision Tree Classifier
to predict if a person survived from the Titanic disaster
based on the person's details

"""
from titanic_ml_competition.feature_selection import generate_model_arg
from titanic_ml_competition.getdata import getdata
from sklearn import tree
import csv

train_features, test_features, train_label = generate_model_arg("train.csv", "test.csv")

clf = tree.DecisionTreeClassifier()
clf.fit(train_features, train_label)
pred = clf.predict(test_features)



###create a csv file of the prediction
test_dict = getdata("test.csv")
id = ["PassengerId"]
for person in test_dict.keys():
    id.append(test_dict[person]["PassengerId"])

sur = ["Survived"]
for i in list(pred):
    sur.append(i)

with open('Predictions.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(id, sur))

