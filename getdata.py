"""
The code will be used to extract the data
from the csv file and covert it into a dictionary.
where the keys will be the persons name and the value will their details
"""
import pandas as pd

def getdata(filename):
    """this function will be used to fetch the raw data
    covert the raw data to a dictionary
    :argument: raw file name
    :returns returns a dictionary of keys name and values persons details('PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked')
    """
    data = pd.read_csv(filename)
    data = data.fillna(0)
    person_details = {}
    for index in range(len(data.index)):
        subdict = {}
        for detail_type in list(data.columns):
            if detail_type == "Name":
                continue
            else:
                subdict[detail_type] = data[detail_type][index]
        person_details[data["Name"][index]] = subdict

    return person_details
