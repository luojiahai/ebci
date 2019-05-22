import numpy as np
from sklearn import svm
import pprint

import utilities
import eci


def tabular_driver(dataset, gamma):
    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(dataset.train, dataset.labels_train)
    # pprint.pprint(clf.score(dataset.validation, dataset.labels_validation))

    interpreter = eci.ECI(training_data=np.array(dataset.train), 
                          feature_names=dataset.feature_names,
                          categorical_features=dataset.categorical_features,
                          categorical_names=dataset.categorical_names)

    for i in range(10):
        instance = dataset.validation[i]
        fact, contrast = interpreter.interpret(instance=instance, 
                                            class_label=clf.predict([instance])[0], 
                                            predict_fn=clf.predict_proba,
                                            c=1, gamma=gamma, kappa=1)

        for k, v in {'subject': instance, 'fact': fact, 'contrast': contrast}.items():
            class_label = clf.predict([v])[0]
            predict_proba = clf.predict_proba([v])[0]
            values = []
            for e in v: values.append(int(e))
            contents = [f"{k} feature_values {str(values)} class_label {class_label} predict_proba {predict_proba}"]
            contents.append(f"prediction: {dataset.class_names[clf.predict([v])[0]]}")
            for j in range(len(v)):
                feature_name = dataset.feature_names[j]
                value = int(v[j])
                categorical_name = dataset.categorical_names[j][value]
                contents.append(f"\t{feature_name}: {categorical_name}")
            utilities.Debug.log(contents=contents)
            
        utilities.Debug.log(contents=['\n'])

def main():
    print("Hello, World!")

    # reset log
    open('log.txt', 'w').close()

    dataset = utilities.load_dataset('adult')
    utilities.Debug.log(contents=[dataset.class_names, 
                                  dataset.feature_names, 
                                  dataset.categorical_names],
                        pformat=True)
    utilities.Debug.log(contents=['\n'])
    tabular_driver(dataset, 0.5)
    

if __name__ == "__main__":
    main()