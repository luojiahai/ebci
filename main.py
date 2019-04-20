import numpy as np
from sklearn import svm

import utilities
import ebci


def tabular_driver_loan():
    dataset = utilities.load_dataset('loan')
    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(dataset.train, dataset.labels_train)
    # pprint.pprint(clf.score(dataset.validation, dataset.labels_validation))

    interpreter = ebci.EBCI(training_data=np.array(dataset.train), 
                            feature_names=dataset.feature_names,
                            categorical_features=dataset.categorical_features,
                            categorical_names=dataset.categorical_names)
    instance = dataset.validation[0]
    class_label = clf.predict([instance])[0]
    pp, pn = interpreter.interpret(instance=instance, 
                                   class_label=class_label, 
                                   predict_fn=clf.predict_proba,
                                   c=1, gamma=0.1, kappa=1)
    print("instance", instance, class_label, clf.predict_proba([instance])[0])
    print("pertinent positive instance", pp, clf.predict([pp])[0], clf.predict_proba([pp])[0])
    print("pertinent negative instance", pn, clf.predict([pn])[0], clf.predict_proba([pn])[0])

def main():
    print("Hello, World!")

    tabular_driver_loan()
    

if __name__ == "__main__":
    main()