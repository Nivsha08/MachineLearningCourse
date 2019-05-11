from numpy import count_nonzero, logical_and, logical_or, concatenate, mean, array_split, poly1d, polyfit, array
from numpy.random import permutation
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt


SVM_DEFAULT_DEGREE = 3
SVM_DEFAULT_GAMMA = 'auto'
SVM_DEFAULT_C = 1.0
ALPHA = 1.5


def prepare_data(data, labels, max_count=None, train_ratio=0.8):
    """
    :param data: a numpy array with the features dataset
    :param labels:  a numpy array with the labels
    :param max_count: max amount of samples to work on. can be used for testing
    :param train_ratio: ratio of samples used for train
    :return: train_data: a numpy array with the features dataset - for train
             train_labels: a numpy array with the labels - for train
             test_data: a numpy array with the features dataset - for test
             test_labels: a numpy array with the features dataset - for test
    """
    if max_count:
        data = data[:max_count]
        labels = labels[:max_count]

    # shuffle the dataset
    shuffled_dataset = permutation(concatenate((data, labels[:, None]), axis=1))

    train_samples = round(train_ratio * max_count)

    train_data = shuffled_dataset[:train_samples, :-1]
    train_labels = shuffled_dataset[:train_samples, -1]
    test_data = shuffled_dataset[train_samples:, :-1]
    test_labels = shuffled_dataset[train_samples:, -1]

    return train_data, train_labels, test_data, test_labels


def get_stats(prediction, labels):
    """
    :param prediction: a numpy array with the prediction of the model
    :param labels: a numpy array with the target values (labels)
    :return: tpr: true positive rate
             fpr: false positive rate
             accuracy: accuracy of the model given the predictions
    """

    num_of_labels = labels.shape[0]
    positives = count_nonzero(labels, axis=0)
    negatives = num_of_labels - positives

    tp_counter = 0
    fp_counter = 0
    success_counter = 0

    for i in range(num_of_labels):
        if prediction[i] == labels[i]:
            success_counter += 1
        if prediction[i] == 1 and labels[i] == 1:
            tp_counter += 1
        elif prediction[i] == 1 and labels[i] == 0:
            fp_counter += 1

    tpr = tp_counter / positives
    fpr = fp_counter / negatives
    accuracy = success_counter / num_of_labels

    return tpr, fpr, accuracy


def get_k_fold_stats(folds_array, labels_array, clf):
    """
    :param folds_array: a k-folds arrays based on a dataset with M features and N samples
    :param labels_array: a k-folds labels array based on the same dataset
    :param clf: the configured SVC learner
    :return: mean(tpr), mean(fpr), mean(accuracy) - means across all folds
    """
    tpr = []
    fpr = []
    accuracy = []

    k = len(folds_array)

    # choose a different testset from the k arrays, and treat the rest as trainset
    # combine the arrays with the their labels columns
    for i in range(k):
        test_labels_array = labels_array[i]
        testset_array = folds_array[i]

        train_labels_array = concatenate([sub_array for j, sub_array in enumerate(labels_array) if j != i], axis=0)
        trainset_array = concatenate([sub_array for j, sub_array in enumerate(folds_array) if j != i], axis=0)

        clf.fit(trainset_array, train_labels_array)

        # get prediction from the SVC learner
        predicted_labels = clf.predict(testset_array)

        # get model stats and append them to the lists
        tpr_stats, fpr_stats, accuracy_stats = get_stats(predicted_labels, test_labels_array)
        tpr.append(tpr_stats)
        fpr.append(fpr_stats)
        accuracy.append(accuracy_stats)

    return mean(tpr), mean(fpr), mean(accuracy)


def compare_svms(data_array,
                 labels_array,
                 folds_count,
                 kernels_list=('poly', 'poly', 'poly', 'rbf', 'rbf', 'rbf',),
                 kernel_params=({'degree': 2}, {'degree': 3}, {'degree': 4}, {'gamma': 0.005}, {'gamma': 0.05}, {'gamma': 0.5},)):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :param kernels_list: a list of strings defining the SVM kernels
    :param kernel_params: a dictionary with kernel parameters - degree, gamma, c
    :return: svm_df: a dataframe containing the results as described below
    """
    svm_df = pd.DataFrame()
    svm_df['kernel'] = kernels_list
    svm_df['kernel_params'] = kernel_params
    svm_df['tpr'] = None
    svm_df['fpr'] = None
    svm_df['accuracy'] = None

    tpr_list = []
    fpr_list = []
    accuracy_list = []

    for i in range(len(kernels_list)):
        if kernels_list[i] == 'poly':
            clf = SVC(kernel=kernels_list[i], degree=kernel_params[i]['degree'], gamma='auto')
        else:
            clf = SVC(kernel=kernels_list[i], gamma=kernel_params[i]['gamma'])

        folds_array = array_split(data_array, folds_count)
        labels_folds_array = array_split(labels_array, folds_count)

        tpr, fpr, accuracy = get_k_fold_stats(folds_array, labels_folds_array, clf)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        accuracy_list.append(accuracy)

    svm_df['tpr'] = tpr_list
    svm_df['fpr'] = fpr_list
    svm_df['accuracy'] = accuracy_list

    print(svm_df)

    return svm_df


def get_most_accurate_kernel():
    """
    :return: integer representing the row number of the most accurate kernel
    """
    best_kernel = 0
    return best_kernel


def get_kernel_with_highest_score():
    """
    :return: integer representing the row number of the kernel with the highest score
    """
    best_kernel = 0
    return best_kernel


def plot_roc_curve_with_score(df, alpha_slope=1.5):
    """
    :param df: a dataframe containing the results of compare_svms
    :param alpha_slope: alpha parameter for plotting the linear score line
    :return:
    """
    x = df.fpr.tolist()
    y = df.tpr.tolist()
    b, m = polyfit(x, y, 1)

    line = []
    for i in x:
        line.append(b + alpha_slope * i)

    print(line)
    plt.title("ROC Scutter Plot")
    plt.scatter(x, y)
    plt.plot(x, line, '-r')
    plt.show()


def evaluate_c_param(data_array, labels_array, folds_count):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :return: res: a dataframe containing the results for the different c values. columns similar to `compare_svms`
    """

    res = pd.DataFrame()
    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return res


def get_test_set_performance(train_data, train_labels, test_data, test_labels):
    """
    :param train_data: a numpy array with the features dataset - train
    :param train_labels: a numpy array with the labels - train

    :param test_data: a numpy array with the features dataset - test
    :param test_labels: a numpy array with the labels - test
    :return: kernel_type: the chosen kernel type (either 'poly' or 'rbf')
             kernel_params: a dictionary with the chosen kernel's parameters - c value, gamma or degree
             clf: the SVM leaner that was built from the parameters
             tpr: tpr on the test dataset
             fpr: fpr on the test dataset
             accuracy: accuracy of the model on the test dataset
    """

    kernel_type = ''
    kernel_params = None
    clf = SVC(class_weight='balanced')  # TODO: set the right kernel
    tpr = 0.0
    fpr = 0.0
    accuracy = 0.0

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return kernel_type, kernel_params, clf, tpr, fpr, accuracy
