import os
import copy
import numpy as np
import sklearn
import sklearn.preprocessing
import sklearn.tree
from sklearn.utils import check_random_state
from abc import ABCMeta, abstractmethod
import pprint


class BaseDiscretizer():
    """
    Abstract class - Build a class that inherits from this class to implement
    a custom discretizer.
    Method bins() is to be redefined in the child class, as it is the actual
    custom part of the discretizer.
    """

    __metaclass__ = ABCMeta  # abstract class

    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        """Initializer
        Args:
            data: numpy 2d array
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. These features will not be discretized.
                Everything else will be considered continuous, and will be
                discretized.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
        """
        self.to_discretize = ([x for x in range(data.shape[1])
                              if x not in categorical_features])
        self.names = {}
        self.lambdas = {}
        self.means = {}
        self.stds = {}
        self.mins = {}
        self.maxs = {}
        self.random_state = check_random_state(random_state)

        # To override when implementing a custom binning
        bins = self.bins(data, labels)
        bins = [np.unique(x) for x in bins]

        for feature, qts in zip(self.to_discretize, bins):
            n_bins = qts.shape[0]  # Actually number of borders (= #bins-1)
            boundaries = np.min(data[:, feature]), np.max(data[:, feature])
            name = feature_names[feature]

            self.names[feature] = ['%s <= %.2f' % (name, qts[0])]
            for i in range(n_bins - 1):
                self.names[feature].append('%.2f < %s <= %.2f' %
                                           (qts[i], name, qts[i + 1]))
            self.names[feature].append('%s > %.2f' % (name, qts[n_bins - 1]))

            self.lambdas[feature] = lambda x, qts=qts: np.searchsorted(qts, x)
            discretized = self.lambdas[feature](data[:, feature])

            self.means[feature] = []
            self.stds[feature] = []
            for x in range(n_bins + 1):
                selection = data[discretized == x, feature]
                mean = 0 if len(selection) == 0 else np.mean(selection)
                self.means[feature].append(mean)
                std = 0 if len(selection) == 0 else np.std(selection)
                std += 0.00000000001
                self.stds[feature].append(std)
            self.mins[feature] = [boundaries[0]] + qts.tolist()
            self.maxs[feature] = qts.tolist() + [boundaries[1]]

    @abstractmethod
    def bins(self, data, labels):
        """
        To be overridden
        Returns for each feature to discretize the boundaries
        that form each bin of the discretizer
        """
        raise NotImplementedError("Must override bins() method")

    def discretize(self, data):
        """Discretizes the data.
        Args:
            data: numpy 2d or 1d array
        Returns:
            numpy array of same dimension, discretized.
        """
        ret = data.copy()
        for feature in self.lambdas:
            if len(data.shape) == 1:
                ret[feature] = int(self.lambdas[feature](ret[feature]))
            else:
                ret[:, feature] = self.lambdas[feature](
                    ret[:, feature]).astype(int)
        return ret

    def undiscretize(self, data):
        ret = data.copy()
        for feature in self.means:
            mins = self.mins[feature]
            maxs = self.maxs[feature]
            means = self.means[feature]
            stds = self.stds[feature]

            def get_inverse(q):
                return max(mins[q],
                           min(self.random_state.normal(means[q], stds[q]), maxs[q]))
            if len(data.shape) == 1:
                q = int(ret[feature])
                ret[feature] = get_inverse(q)
            else:
                ret[:, feature] = (
                    [get_inverse(int(x)) for x in ret[:, feature]])
        return ret

class QuartileDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):

        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)

    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            qts = np.array(np.percentile(data[:, feature], [25, 50, 75]))
            bins.append(qts)
        return bins

class DecileDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)

    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            qts = np.array(np.percentile(data[:, feature],
                                         [10, 20, 30, 40, 50, 60, 70, 80, 90]))
            bins.append(qts)
        return bins

class EntropyDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        if(labels is None):
            raise ValueError('Labels must be not None when using \
                             EntropyDiscretizer')
        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)

    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            # Entropy splitting / at most 8 bins so max_depth=3
            dt = sklearn.tree.DecisionTreeClassifier(criterion='entropy',
                                                     max_depth=3,
                                                     random_state=self.random_state)
            x = np.reshape(data[:, feature], (-1, 1))
            dt.fit(x, labels)
            qts = dt.tree_.threshold[np.where(dt.tree_.children_left > -1)]

            if qts.shape[0] == 0:
                qts = np.array([np.median(data[:, feature])])
            else:
                qts = np.sort(qts)

            bins.append(qts)

        return bins

class Debug(object):
    def __init__(self):
        pass

    @staticmethod
    def log(contents, pformat=False):
        f = open('log.txt', 'a')
        for content in contents:
            if (pformat):
                f.write(pprint.pformat(content))
            else:
                f.write(str(content))
            f.write('\n')
        f.close()

class Dataset(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def map_array_values(array, value_map):
    # value map must be { src : target }
    ret = array.copy()
    for src, target in value_map.items():
        ret[ret == src] = target
    return ret

def load_csv_dataset(data, target_idx, delimiter=',',
                     feature_names=None, categorical_features=None,
                     features_to_use=None, feature_transformations=None,
                     discretize=False, balance=False, fill_na='-1', filter_fn=None, skip_first=False):
    """if not feature names, takes 1st line as feature names
    if not features_to_use, use all except for target
    if not categorical_features, consider everything < 20 as categorical"""
    if feature_transformations is None:
        feature_transformations = {}
    try:
        data = np.genfromtxt(data, delimiter=delimiter, dtype='|S128')
    except:
        import pandas
        data = pandas.read_csv(data,
                               header=None,
                               delimiter=delimiter,
                               na_filter=True,
                               dtype=str).fillna(fill_na).values
    if target_idx < 0:
        target_idx = data.shape[1] + target_idx
    ret = Dataset({})
    if feature_names is None:
        feature_names = list(data[0])
        data = data[1:]
    else:
        feature_names = copy.deepcopy(feature_names)
    if skip_first:
        data = data[1:]
    if filter_fn is not None:
        data = filter_fn(data)
    for feature, fun in feature_transformations.items():
        data[:, feature] = fun(data[:, feature])
    labels = data[:, target_idx]
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(labels)
    ret.labels = le.transform(labels)
    labels = ret.labels
    ret.class_names = list(le.classes_)
    ret.class_target = feature_names[target_idx]
    if features_to_use is not None:
        data = data[:, features_to_use]
        feature_names = ([x for i, x in enumerate(feature_names)
                          if i in features_to_use])
        if categorical_features is not None:
            categorical_features = ([features_to_use.index(x)
                                     for x in categorical_features])
    else:
        data = np.delete(data, target_idx, 1)
        feature_names.pop(target_idx)
        if categorical_features:
            categorical_features = ([x if x < target_idx else x - 1
                                     for x in categorical_features])

    if categorical_features is None:
        categorical_features = []
        for f in range(data.shape[1]):
            if len(np.unique(data[:, f])) < 20:
                categorical_features.append(f)
    categorical_names = {}
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[:, feature])
        data[:, feature] = le.transform(data[:, feature])
        categorical_names[feature] = le.classes_
    data = data.astype(float)
    ordinal_features = []
    if discretize:
        disc = QuartileDiscretizer(data, categorical_features, feature_names)
        data = disc.discretize(data)
        ordinal_features = [x for x in range(data.shape[1])
                            if x not in categorical_features]
        categorical_features = range(data.shape[1])
        categorical_names.update(disc.names)
    ret.ordinal_features = ordinal_features
    ret.categorical_features = categorical_features
    ret.categorical_names = categorical_names
    ret.feature_names = feature_names
    np.random.seed(1)
    if balance:
        idxs = np.array([], dtype='int')
        min_labels = np.min(np.bincount(labels))
        for label in np.unique(labels):
            idx = np.random.choice(np.where(labels == label)[0], min_labels)
            idxs = np.hstack((idxs, idx))
        data = data[idxs]
        labels = labels[idxs]
        ret.data = data
        ret.labels = labels

    splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                  test_size=.2,
                                                  random_state=1)
    train_idx, test_idx = [x for x in splits.split(data)][0]
    ret.train = data[train_idx]
    ret.labels_train = ret.labels[train_idx]
    cv_splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                     test_size=.5,
                                                     random_state=1)
    cv_idx, ntest_idx = [x for x in cv_splits.split(test_idx)][0]
    cv_idx = test_idx[cv_idx]
    test_idx = test_idx[ntest_idx]

    ret.validation = data[cv_idx]
    ret.labels_validation = ret.labels[cv_idx]
    ret.test = data[test_idx]
    ret.labels_test = ret.labels[test_idx]
    ret.test_idx = test_idx
    ret.validation_idx = cv_idx
    ret.train_idx = train_idx
    ret.data = data
    return ret

def load_dataset(dataset_name, 
                 balance=False, 
                 discretize=True, 
                 dataset_folder='./data'):
    if dataset_name == 'adult':
        feature_names = ["Age", "Workclass", "fnlwgt", "Education",
                         "Education-Num", "Marital Status", "Occupation",
                         "Relationship", "Race", "Sex", "Capital Gain",
                         "Capital Loss", "Hours per week", "Country", 'Income']
        features_to_use = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        categorical_features = [1, 3, 5, 6, 7, 8, 9, 10, 11, 13]
        education_map = {
            '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
            'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
            'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
            'Some-college': 'High School grad', 'Masters': 'Masters',
            'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
            'Assoc-voc': 'Associates',
        }
        occupation_map = {
            "Adm-clerical": "Admin", "Armed-Forces": "Military",
            "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
            "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
            "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
            "Service", "Priv-house-serv": "Service", "Prof-specialty":
            "Professional", "Protective-serv": "Other", "Sales":
            "Sales", "Tech-support": "Other", "Transport-moving":
            "Blue-Collar",
        }
        country_map = {
            'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
            'China', 'Columbia': 'South-America', 'Cuba': 'Other',
            'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
            'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
            'France': 'Euro_1', 'Germany': 'Euro_1', 'Greece': 'Euro_2',
            'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
            'Holand-Netherlands': 'Euro_1', 'Honduras': 'Latin-America',
            'Hong': 'China', 'Hungary': 'Euro_2', 'India':
            'British-Commonwealth', 'Iran': 'Other', 'Ireland':
            'British-Commonwealth', 'Italy': 'Euro_1', 'Jamaica':
            'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
            'Latin-America', 'Nicaragua': 'Latin-America',
            'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
            'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_2',
            'Portugal': 'Euro_2', 'Puerto-Rico': 'Latin-America', 'Scotland':
            'British-Commonwealth', 'South': 'Euro_2', 'Taiwan': 'China',
            'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
            'United-States': 'United-States', 'Vietnam': 'SE-Asia'
        }
        married_map = {
            'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
            'Married-civ-spouse': 'Married', 'Married-spouse-absent':
            'Separated', 'Separated': 'Separated', 'Divorced':
            'Separated', 'Widowed': 'Widowed'
        }
        label_map = {'<=50K': 'Less than $50,000', '>50K': 'More than $50,000'}

        def cap_gains_fn(x):
            x = x.astype(float)
            d = np.digitize(x, [0, np.median(x[x > 0]), float('inf')],
                            right=True).astype('|S128')
            return map_array_values(d, {'0': 'None', '1': 'Low', '2': 'High'})

        transformations = {
            3: lambda x: map_array_values(x, education_map),
            5: lambda x: map_array_values(x, married_map),
            6: lambda x: map_array_values(x, occupation_map),
            10: cap_gains_fn,
            11: cap_gains_fn,
            13: lambda x: map_array_values(x, country_map),
            14: lambda x: map_array_values(x, label_map),
        }
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'adult/adult.data'), -1, ', ',
            feature_names=feature_names, features_to_use=features_to_use,
            categorical_features=categorical_features, discretize=discretize,
            balance=balance, feature_transformations=transformations)
    if (dataset_name == 'loan'):
        def filter_fn(data):
                to_remove = ['Does not meet the credit policy. Status:Charged Off',
                'Does not meet the credit policy. Status:Fully Paid',
                'In Grace Period', '-999', 'Current']
                for x in to_remove:
                    data = data[data[:, 16] != x]
                return data
        bad_statuses = set(["Late (16-30 days)", "Late (31-120 days)", "Default", "Charged Off"])
        transformations = {
            16:  lambda x: np.array([y in bad_statuses for y in x]).astype(int),
            19:  lambda x: np.array([len(y) for y in x]).astype(int),
            6:  lambda x: np.array([y.strip('%') if y else -1 for y in x]).astype(float),
            35:  lambda x: np.array([y.strip('%') if y else -1 for y in x]).astype(float),
        }
        features_to_use = [2, 12, 13, 19, 29, 35, 51, 52, 109]
        categorical_features = [12, 109]
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'LoanStats3a_securev1.csv'),
            16, ',',  features_to_use=features_to_use,
            feature_transformations=transformations, fill_na='-999',
            categorical_features=categorical_features, discretize=discretize,
            filter_fn=filter_fn, balance=True)
        dataset.class_names = ['Good Loan', 'Bad Loan']
    if (dataset_name == 'breast'):
        features_to_use = [i for i in range(2, 12)]
        feature_names = ['id', 'diagnosis',
                         'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                         'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                         'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                         'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
                         'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
                         'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
        dataset = load_csv_dataset(
            os.path.join(dataset_folder + '/breast-cancer', 'wdbc.data'),
            1, ',', feature_names=feature_names, features_to_use=features_to_use, 
            discretize=discretize, balance=True)
        dataset.class_names = ['Benign', 'Malignant']
    return dataset
