import torch
import numpy as np
import math
import random
import copy
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy import stats

plt.switch_backend('agg')


def set_random_seed(seed):
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_minibatches_idx(n, minibatch_size, shuffle=True):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(math.ceil(n / minibatch_size)):
        minibatches.append(idx_list[minibatch_start: minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    return minibatches


def find_list_index(num_list, x):
    index = len(num_list) - 1
    for i in range(len(num_list)):
        if num_list[len(num_list) - 1 - i] <= x:
            index = len(num_list) - 1 - i
            break
    return index


def linear_regression(X, Y, option, delta=False):
    X = copy.deepcopy(X).reshape(-1, 1)
    Y = copy.deepcopy(Y)
    if delta:
        Y = [Y[i] - Y[i + 1] for i in range(len(Y))[:-1]]
        X = X[:-1]
    if option == 'poly':
        X = np.log(X)
        Y = np.log(Y)
    elif option == 'exp':
        Y = np.log(Y)
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    y_pred = regr.predict(X)
    rsquare = r2_score(Y, y_pred)
    print('Coefficients: \n', regr.coef_, regr.intercept_)
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f' % rsquare)
    return X, Y, y_pred, rsquare


def analyze_terminal_scaling_laws(rate_reduction_list, terminal_layer_path):
    rate_reduction_list = np.array(rate_reduction_list)
    X = np.arange(len(rate_reduction_list)) + 1
    Y = rate_reduction_list
    X, Y, y_pred, rsquare = linear_regression(X, Y, option='exp', delta=False)
    print('pearson correlation', stats.pearsonr(X.reshape(-1), Y))
    plt.rcParams.update({'font.size': 20})
    plt.plot(X, y_pred, color='#1f77b4', linewidth=4)
    plt.scatter(X, Y, color='k', s=80)
    plt.ylim([-1, 4])
    plt.savefig(terminal_layer_path)
    plt.close()


def analyze_terminal_scaling_laws_mixresidual(rate_reduction_list, terminal_layer_path):
    rate_reduction_list = np.array(rate_reduction_list)
    # mid_len = len(rate_reduction_list) // 2
    mid_len = 6 # for resnetmixv2 with 4 blocks
    X = np.arange(len(rate_reduction_list)) + 1
    Y = rate_reduction_list
    X = X[: mid_len + 1]
    Y = Y[: mid_len + 1]
    X, Y, y_pred, rsquare = linear_regression(X, Y, option='exp', delta=False)
    print('pearson correlation', stats.pearsonr(X.reshape(-1), Y))
    plt.rcParams.update({'font.size': 20})
    plt.plot(X, y_pred, color='#1f77b4', linewidth=4)
    X = np.arange(len(rate_reduction_list)) + 1
    Y = rate_reduction_list
    X = X[mid_len:]
    Y = Y[mid_len:]
    X, Y, y_pred, rsquare = linear_regression(X, Y, option='exp', delta=False)
    print('pearson correlation', stats.pearsonr(X.reshape(-1), Y))
    plt.rcParams.update({'font.size': 20})
    plt.plot(X, y_pred, color='#1f77b4', linewidth=4)
    X = np.arange(len(rate_reduction_list)) + 1
    Y = np.log(rate_reduction_list)
    print(X, Y)
    plt.scatter(X, Y, color='k', s=80)
    plt.ylim([2, 3.5])
    # plt.xlabel('Layer')
    plt.savefig(terminal_layer_path)
    plt.close()

