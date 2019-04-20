import numpy as np
from sklearn import svm
import pprint

import utilities


def main():
    print("hello, World!")

    dataset = utilities.load_dataset('loan')
    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(dataset.train, dataset.labels_train)
    pprint.pprint(clf.score(dataset.validation, dataset.labels_validation))


if __name__ == "__main__":
    main()