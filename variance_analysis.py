import numpy as np
import scipy.linalg
import copy


def get_class_features(features, labels, config):
    class_features = [[] for i in range(config['class_num'])]
    for i in range(len(labels)):
        class_features[int(labels[i])].append(features[i])
    return class_features


def get_variation(features, labels, config):
    features = copy.deepcopy(features)
    labels = copy.deepcopy(labels)
    avg_feature = np.mean(features, axis=0)
    features -= avg_feature
    class_features = get_class_features(features, labels, config)
    feature_dim = len(class_features[0][0])
    between_class_covariance = np.zeros((feature_dim, feature_dim))
    within_class_covariance = np.zeros((feature_dim, feature_dim))
    for i in range(config['class_num']):
        cur_class_features = np.array(class_features[i])
        cur_class_avg_feature = np.mean(cur_class_features, axis=0)
        between_class_covariance += np.matmul(cur_class_avg_feature.reshape(-1, 1),
                                              cur_class_avg_feature.reshape(1, -1))
        cur_class_centralized_features = cur_class_features - cur_class_avg_feature
        cur_class_covariance = np.matmul(np.transpose(cur_class_centralized_features), cur_class_centralized_features)
        cur_class_covariance /= len(cur_class_features)
        within_class_covariance += cur_class_covariance
    between_class_covariance /= config['class_num']
    within_class_covariance /= config['class_num']
    between_class_inverse_covariance = scipy.linalg.pinv(between_class_covariance)
    within_variation = np.trace(np.matmul(within_class_covariance, between_class_inverse_covariance))
    between_variation = np.trace(between_class_covariance / np.linalg.norm(between_class_covariance))
    return within_variation, between_variation


def get_label_distribution(labels):
    label_distribution = {}
    for label in labels:
        label = int(label)
        if label in label_distribution:
            label_distribution[label] += 1
        else:
            label_distribution[label] = 1
    label_distribution = dict(sorted(label_distribution.items(), key=lambda item: item[0]))
    return label_distribution


def get_variation_imbalance(features, labels, config):
    features = copy.deepcopy(features)
    labels = copy.deepcopy(labels)
    avg_feature = np.mean(features, axis=0)
    features -= avg_feature
    class_features = get_class_features(features, labels, config)
    label_distribution = get_label_distribution(labels)
    feature_dim = len(class_features[0][0])
    between_class_covariance = np.zeros((feature_dim, feature_dim))
    within_class_covariance = np.zeros((feature_dim, feature_dim))
    for i in range(config['class_num']):
        cur_class_features = np.array(class_features[i])
        cur_class_avg_feature = np.mean(cur_class_features, axis=0)
        between_class_covariance += np.matmul(cur_class_avg_feature.reshape(-1, 1),
                                              cur_class_avg_feature.reshape(1, -1)) * label_distribution[i]
        cur_class_centralized_features = cur_class_features - cur_class_avg_feature
        cur_class_covariance = np.matmul(np.transpose(cur_class_centralized_features), cur_class_centralized_features)
        within_class_covariance += cur_class_covariance
    between_class_covariance /= len(labels)
    within_class_covariance /= len(labels)
    between_class_inverse_covariance = scipy.linalg.pinv(between_class_covariance)
    within_variation = np.trace(np.matmul(within_class_covariance, between_class_inverse_covariance))
    between_variation = np.trace(between_class_covariance / np.linalg.norm(between_class_covariance))
    return within_variation, between_variation
