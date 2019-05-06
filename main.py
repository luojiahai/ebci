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
    instance = dataset.validation[0]
    class_label = clf.predict([instance])[0]
    fact, contrast = interpreter.interpret(instance=instance, 
                                           class_label=class_label, 
                                           predict_fn=clf.predict_proba,
                                           c=1, gamma=gamma, kappa=1)
    results = [
        f"instance {instance} class {class_label} predict proba {clf.predict_proba([instance])[0]}",
        f"fact {fact} class {clf.predict([fact])[0]} predict proba {clf.predict_proba([fact])[0]}",
        f"contrast {contrast} class {clf.predict([contrast])[0]} predict proba {clf.predict_proba([contrast])[0]}",
        '\n'
    ]
    utilities.Debug.log(contents=results)

def main():
    print("Hello, World!")

    # reset log
    open('log.txt', 'w').close()

    dataset = utilities.load_dataset('adult')
    utilities.Debug.log(contents=[dataset.class_names, 
                                  dataset.feature_names, 
                                  dataset.categorical_names,
                                  '\n'],
                        pformat=True)
    tabular_driver(dataset, 0.7)
    

if __name__ == "__main__":
    main()