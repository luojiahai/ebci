import numpy as np
import sklearn.metrics
import sklearn.preprocessing
import sklearn.utils
import collections


class EBCI(object):
    def __init__(self,
                 training_data,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 random_state=None):
        self.random_state = sklearn.utils.check_random_state(random_state)

        if categorical_features is None:
            categorical_features = []
        if feature_names is None:
            feature_names = [str(i) for i in range(training_data.shape[1])]
        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)

        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)
        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in self.categorical_features:
            column = training_data[:, feature]
            feature_count = collections.Counter(column)
            values, frequencies = map(list, zip(*(feature_count.items())))
            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 float(sum(frequencies)))
            self.scaler.mean_[feature] = 0
            self.scaler.scale_[feature] = 1

    def interpret(self, 
                  instance, 
                  class_label,
                  predict_fn, 
                  distance_metric='euclidean',
                  perturbation_size=5000, 
                  c=1, gamma=1, kappa=1):
        samples = self.perturbation(instance, perturbation_size)
        distance_fn = self.pairwise_distance(distance_metric)
        ppos_scores = list()
        pneg_scores = list()

        for sample in samples:
            ppos_loss = self.ppos_loss_fn(sample, class_label, predict_fn, kappa)
            ppos_distance = distance_fn(instance, sample)[0]
            ppos_score = c * ppos_loss + gamma * ppos_distance
            ppos_scores.append(ppos_score)

            pneg_loss = self.pneg_loss_fn(sample, class_label, predict_fn, kappa)
            pneg_distance = distance_fn(instance, sample)[0]
            pneg_score = c * pneg_loss + gamma * pneg_distance
            pneg_scores.append(pneg_score)

        ppos_instance_idx = np.argmin(ppos_scores[1:], axis=0) + 1
        pneg_instance_idx = np.argmin(pneg_scores[1:], axis=0) + 1
        ppos_instance = samples[ppos_instance_idx]
        pneg_instance = samples[pneg_instance_idx]

        return ppos_instance, pneg_instance

    def perturbation(self, 
                     instance, 
                     perturbation_size):
        data = np.zeros((perturbation_size, instance.shape[0]))
        data = self.random_state.normal(0, 1, 
                                        perturbation_size * instance.shape[0]
                                        ).reshape(perturbation_size, instance.shape[0])
        data = data * self.scaler.scale_ + instance

        categorical_features = self.categorical_features
        first_row = instance
        data[0] = instance.copy()
        inverse = data.copy()

        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values, 
                                                      size=perturbation_size, 
                                                      replace=True, 
                                                      p=freqs)
            binary_column = np.array([1 if x == first_row[column] else 0 
                                      for x in inverse_column])
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column

        inverse[0] = instance

        return inverse

    def pairwise_distance(self, 
                          distance_metric):
        lambda_fn = lambda x, y: sklearn.metrics.pairwise_distances(y.reshape(1,-1), 
                                                                    x.reshape(1,-1),
                                                                    metric=distance_metric).ravel()
        return lambda_fn

    def ppos_loss_fn(self, 
                     sample,
                     class_label, 
                     predict_fn, 
                     kappa):
        prediction = predict_fn([sample])[0]
        max_prediction = -1
        for i in range(len(prediction)):
            if (i != class_label and prediction[i] >= max_prediction):
                max_prediction = prediction[i]
        return max(max_prediction - prediction[class_label], -kappa)

    def pneg_loss_fn(self, 
                     sample,
                     class_label, 
                     predict_fn, 
                     kappa):
        prediction = predict_fn([sample])[0]
        max_prediction = -1
        for i in range(len(prediction)):
            if (i != class_label and prediction[i] >= max_prediction):
                max_prediction = prediction[i]
        return max(prediction[class_label] - max_prediction, -kappa)
        