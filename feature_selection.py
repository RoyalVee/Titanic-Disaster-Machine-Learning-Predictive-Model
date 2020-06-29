from titanic_ml_competition.getdata import getdata
"""
this code will be used to select the features
that will be used for the model training and testing
"""

def select_feature(data,features):
    """selected the need features from the data"""
    newdata = {}
    for person in data.keys():
        newsubdict = {}
        for feature in features:
            newsubdict[feature] = data[person][feature]
        newdata[person] = newsubdict

    return newdata


def label(data):
    """get the label which will be a list if the 'Survived' index"""
    Survived = []
    for person in data.keys():
        Survived.append(data[person]['Survived'])

    return Survived

def transform_features(data):
    features = []
    for person in data.keys():
        person_feature = []
        ##person_feature.append(int(data[person]['PassengerId']))
        person_feature.append(int(data[person]['Pclass']))

        ##transform sex (male = 1, female = 2)
        if str(data[person]['Sex']).lower == "male":
            person_feature.append(1)
        elif str(data[person]['Sex']).lower == "female":
            person_feature.append(2)
        else:
            person_feature.append(0)

        person_feature.append(float(data[person]['Age']))
        person_feature.append(int(data[person]['SibSp']))
        person_feature.append(int(data[person]['Parch']))
        person_feature.append(int(data[person]['Fare']))

        ##transform cabin value(example C67 = 367, A57 = 157, B78 = 278)
        alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
        cab = data[person]['Cabin']
        newcab = 0
        for alpha in range(len(alphabet)):
            if str(data[person]['Cabin'])[0].lower() == alphabet[alpha]:
                if " " in str(cab):
                    cab = str(cab).split(" ")[0]
                newcab = str(alpha + 1) + str(cab)[1:]
                break
        person_feature.append(int(newcab))

        """"
        ##transform 'Embarked' (C =1, Q =1, S =3 others = 0)
        if data[person]['Embarked'] == "C":
            person_feature.append(1)
        elif data[person]['Embarked'] == "Q":
            person_feature.append(2)
        elif data[person]['Embarked'] == "S":
            person_feature.append(3)
        else:
            person_feature.append(0)"""

        features.append(person_feature)

    return features


def generate_model_arg(raw_train_file, raw_test_file):
    """
    this function is to generate the argument needed for training the model algorithm

    :param raw_train_file:
    :param raw_test_file:
    :return: train_features, test_features, train_label
    """
    selected_features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
    test_data = select_feature(getdata("test.csv"), selected_features)
    train_data = select_feature(getdata("train.csv"), selected_features)
    train_label = label(getdata("train.csv"))
    train_features = transform_features(train_data)
    test_features = transform_features(test_data)

    return train_features, test_features, train_label

