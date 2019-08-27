import numpy as np
from sklearn import svm
from sklearn.externals import joblib
import pprint
from tqdm import tqdm
from random import shuffle

import src.util as util
import src.eci as eci


def get_instances(path):
    f = open(path)
    instances = []
    for line in f:
        splited = line.strip().split(',')
        instances.append([int(e) for e in splited])
    return instances

def train_and_save_model(dataset):
    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(dataset.train, dataset.labels_train)
    joblib.dump(clf, 'model/adult_model.pkl')

def tabular_driver(dataset, gamma):
    clf = joblib.load('model/adult_model.pkl') 

    # # model learning performance
    # pprint.pprint(clf.score(dataset.validation, dataset.labels_validation))

    interpreter = eci.ECI(
        training_data=np.array(dataset.train), 
        feature_names=dataset.feature_names,
        categorical_features=dataset.categorical_features,
        categorical_names=dataset.categorical_names)

    instance = dataset.test[0]
    factual, contrastive = interpreter.interpret(
        instance=instance, class_label=clf.predict([instance])[0], 
        predict_fn=clf.predict_proba, c=1, gamma=gamma, kappa=1)

    subject_vec = list(instance)
    subject_pred = clf.predict([instance])[0]
    subject_vec.append(subject_pred)

    factual_vec = list([int(e) for e in factual])
    factual_pred = clf.predict([factual])[0]
    factual_vec.append(factual_pred)

    contrastive_vec = list([int(e) for e in contrastive])
    contrastive_pred = clf.predict([contrastive])[0]
    contrastive_vec.append(contrastive_pred)

    if ((subject_pred != factual_pred) 
        or (factual_pred == contrastive_pred)
        or (subject_pred == contrastive_pred)):
        print("ExplantionError: explanation not exist")

    print(subject_vec, factual_vec, contrastive_vec)


def main():
    dataset = util.load_dataset('adult')
    # train_and_save_model(dataset)

    tabular_driver(dataset, 0.34)
    

if __name__ == "__main__":
    main()