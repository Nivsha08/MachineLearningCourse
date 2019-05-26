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
        if prediction[i] == 1 and labels[i] == 0:
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
                 kernel_params=(
                         {'degree': 2}, {'degree': 3}, {'degree': 4}, {'gamma': 0.005}, {'gamma': 0.05},
                         {'gamma': 0.5},)):
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
        clf = SVC(
            kernel=kernels_list[i],
            degree=(kernel_params[i]['degree'] if 'degree' in kernel_params[i] else SVM_DEFAULT_DEGREE),
            gamma=(kernel_params[i]['gamma'] if 'gamma' in kernel_params[i] else SVM_DEFAULT_GAMMA),
            C=(kernel_params[i]['C'] if 'C' in kernel_params[i] else SVM_DEFAULT_C)
        )

        folds_array = array_split(data_array, folds_count)
        labels_folds_array = array_split(labels_array, folds_count)

        tpr, fpr, accuracy = get_k_fold_stats(folds_array, labels_folds_array, clf)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        accuracy_list.append(accuracy)

    svm_df['tpr'] = tpr_list
    svm_df['fpr'] = fpr_list
    svm_df['accuracy'] = accuracy_list

    return svm_df


def get_most_accurate_kernel(accuracies_results):
    """
    :return: integer representing the row number of the most accurate kernel
    """
    accuracies_col = list(accuracies_results)
    best_accuracy = max(accuracies_col)
    return accuracies_col.index(best_accuracy)


def get_kernel_with_highest_score(scores_results):
    """
    :return: integer representing the row number of the kernel with the highest score
    """
    scores_col = list(scores_results)
    best_score = max(scores_col)
    return scores_col.index(best_score)


def plot_roc_curve_with_score(df, alpha_slope=ALPHA):
    """
    :param df: a dataframe containing the results of compare_svms
    :param alpha_slope: alpha parameter for plotting the linear score line
    :return:
    """
    # define the plot axes and properties
    plt.figure(figsize=[10, 7])
    plt.title("ROC Scatter Plot")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # plot the scatter ROC
    x = df.fpr.tolist()
    y = df.tpr.tolist()
    plt.scatter(x, y)

    # find the best kernel point
    best_kernel_index = get_kernel_with_highest_score(df['score'])
    best_kernel_x = df.fpr.tolist()[best_kernel_index]
    best_kernel_y = df.tpr.tolist()[best_kernel_index]

    # find the line equation that pass through the point with the given alpha_slope
    b = best_kernel_y - alpha_slope * best_kernel_x

    # plot the line
    line_x = array([0, 1])
    line_y = alpha_slope * line_x + b
    plt.plot(line_x, line_y, 'k--', color='r')

    plt.show()


def create_best_kernel_params(result_dataframe, i_options, j_options):
    """
    Dynamically build the kernel list and kernel params of the different possible C values.
    :param result_dataframe: the dataframe contains all kernel's stats, in order to get the best kernel
    :param i_options: possible i's
    :param j_options: possible j's
    :return: list of kernels and params
    """
    # get the best kernel parameters
    best_kernel_index = get_kernel_with_highest_score(result_dataframe['score'])
    best_kernel_type = result_dataframe.kernel.tolist()[best_kernel_index]
    best_kernel_params = result_dataframe.kernel_params.tolist()[best_kernel_index]

    kernels_list = []
    kernel_params_list = []
    for i in i_options:
        for j in j_options:
            c = (10 ** i) * (j / 3)
            current_params = best_kernel_params.copy()
            current_params['C'] = c
            kernels_list.append(best_kernel_type)
            kernel_params_list.append(current_params)
    return kernels_list, kernel_params_list


def evaluate_c_param(result_dataframe, data_array, labels_array, folds_count):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :return: res: a dataframe containing the results for the different c values. columns similar to `compare_svms`
    """

    i_options = [1, 0, -1, -2, -3, -4]
    j_options = [3, 2, 1]

    kernel_list, kernel_params = create_best_kernel_params(result_dataframe, i_options, j_options)

    res = compare_svms(data_array, labels_array, folds_count,
                       kernels_list=kernel_list,
                       kernel_params=kernel_params)
    return res


def get_best_kernel_and_c_params(result_dataframe):
    """
    Dynamically choose the best C value and kernel params based on the dataframe
    :param result_dataframe: the dataframe contains all kernel's stats with the different C values
    :return: kernel_params: the best params of the best kernel with the best C value
    """
    score_col = list(result_dataframe['score'])
    best_c_index = score_col.index(max(score_col))
    kernel_type = list(result_dataframe['kernel'])[best_c_index]
    kernel_params = list(result_dataframe['kernel_params'])[best_c_index]
    return kernel_type, kernel_params


def get_test_set_performance(result_dataframe, train_data, train_labels, test_data, test_labels):
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
    kernel_type, kernel_params = get_best_kernel_and_c_params(result_dataframe)
    clf = SVC(kernel=kernel_type,
              degree=(kernel_params['degree'] if 'degree' in kernel_params else SVM_DEFAULT_DEGREE),
              gamma=(kernel_params['gamma'] if 'gamma' in kernel_params else SVM_DEFAULT_GAMMA),
              C=kernel_params['C'],
              class_weight='balanced')

    clf.fit(train_data, train_labels)
    test_prediction = clf.predict(test_data)

    tpr, fpr, accuracy = get_stats(test_prediction, test_labels)

    return kernel_type, kernel_params, clf, tpr, fpr, accuracy
