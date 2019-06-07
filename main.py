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

    instances = get_instances('data/samples.txt')
    shuffle(instances)
    for i in tqdm(range(1000)):
        instance = np.array(instances[i][:-1])
        fact, contrast = interpreter.interpret(
            instance=instance, class_label=clf.predict([instance])[0], 
            predict_fn=clf.predict_proba, c=1, gamma=gamma, kappa=1)
        instance_vec = list(instance)
        instance_pred = clf.predict([instance])[0]
        instance_vec.append(instance_pred)
        fact_vec = list([int(e) for e in fact])
        fact_pred = clf.predict([fact])[0]
        fact_vec.append(fact_pred)
        contrast_vec = list([int(e) for e in contrast])
        contrast_pred = clf.predict([contrast])[0]
        contrast_vec.append(contrast_pred)

        if ((instance_pred != fact_pred) 
            or (fact_pred == contrast_pred)
            or (instance_pred == contrast_pred)):
            continue

        contents = [
            '{',
            '    \"instance\": {',
            '        \"subject\": ' + str(instance_vec) + ',', 
            '        \"fact\": ' + str(fact_vec) + ',', 
            '        \"contrast\": ' + str(contrast_vec),
            '    }',
            '},'
        ]
        util.Debug.log(contents=contents)

    # for i in range(1):
    #     instance = dataset.validation[i]
    #     fact, contrast = interpreter.interpret(
    #         instance=instance, class_label=clf.predict([instance])[0], 
    #         predict_fn=clf.predict_proba, perturbation_size=10000, 
    #         c=1, gamma=gamma, kappa=1)
    #     for k, v in {'subject': instance, 'fact': fact, 'contrast': contrast}.items():
    #         class_label = clf.predict([v])[0]
    #         predict_proba = clf.predict_proba([v])[0]
    #         values = []
    #         for e in v: values.append(int(e))
    #         contents = [f"{k} feature_values {str(values)} class_label {class_label} predict_proba {predict_proba}"]
    #         contents.append(f"prediction: {dataset.class_names[clf.predict([v])[0]]}")
    #         for j in range(len(v)):
    #             feature_name = dataset.feature_names[j]
    #             value = int(v[j])
    #             categorical_name = dataset.categorical_names[j][value]
    #             contents.append(f"\t{feature_name}: {categorical_name}")
    #         util.Debug.log(contents=contents)
    #     util.Debug.log(contents=['\n'])

def main():
    print("Hello, World!")
    
    util.Debug.init('temp/log.txt')

    dataset = util.load_dataset('adult')

    # contents = [
    #     dataset.class_names, 
    #     dataset.feature_names, 
    #     dataset.categorical_names
    # ]
    # util.Debug.log(
    #     contents=contents,
    #     pformat=True
    # )
    # util.Debug.log(contents=['\n'])

    # train_and_save_model(dataset)

    util.Debug.log(contents=['['])
    tabular_driver(dataset, 0.34)
    util.Debug.log(contents=[']'])
    

if __name__ == "__main__":
    main()