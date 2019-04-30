import numpy as np
from sklearn import svm
import pprint

import utilities
import ebci


def tabular_driver_loan():
    dataset = utilities.load_dataset('loan')
    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(dataset.train, dataset.labels_train)
    # pprint.pprint(clf.score(dataset.validation, dataset.labels_validation))
    pprint.pprint(dataset.class_names)
    pprint.pprint(dataset.feature_names)
    pprint.pprint(dataset.categorical_names)

    interpreter = ebci.EBCI(training_data=np.array(dataset.train), 
                            feature_names=dataset.feature_names,
                            categorical_features=dataset.categorical_features,
                            categorical_names=dataset.categorical_names)
    instance = dataset.validation[0]
    class_label = clf.predict([instance])[0]
    fact, contrast = interpreter.interpret(instance=instance, 
                                           class_label=class_label, 
                                           predict_fn=clf.predict_proba,
                                           c=1, gamma=0.1, kappa=1)
    print("instance", instance, class_label, clf.predict_proba([instance])[0])
    print("fact", fact, clf.predict([fact])[0], clf.predict_proba([fact])[0])
    print("contrast", contrast, clf.predict([contrast])[0], clf.predict_proba([contrast])[0])

def main():
    print("Hello, World!")

    tabular_driver_loan()
    

if __name__ == "__main__":
    main()