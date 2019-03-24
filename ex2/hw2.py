import numpy as np

np.random.seed(42)

chi_table = {0.01: 6.635,
             0.005: 7.879,
             0.001: 10.828,
             0.0005: 12.116,
             0.0001: 15.140,
             0.00001: 19.511}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.    
    """
    gini = 0.0

    if data.shape[1] > 1:
        label_col = data[:, data.shape[1] - 1]
        labels, count_of_labels = np.unique(label_col, return_counts=True)
        num_of_labels = len(labels)
        sum_of_classes = np.sum(count_of_labels)

        for i in range(num_of_labels):
            probability = (count_of_labels[i] / sum_of_classes)
            gini -= np.power(probability, 2)
        gini = 1 + gini

    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """
    entropy = 0.0

    if data.shape[1] > 1:
        label_col = data[:, data.shape[1] - 1]
        labels, count_of_labels = np.unique(label_col, return_counts=True)
        num_of_labels = len(labels)
        if num_of_labels == 1:
            return entropy
        sum_of_classes = np.sum(count_of_labels)
        for i in range(num_of_labels):
            probability = np.divide(count_of_labels[i], sum_of_classes)
            entropy -= np.multiply(probability, np.log2(probability))

    return entropy


class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.

    def __init__(self, feature, value, group_a_size, group_b_size):
        self.feature = feature  # column index of criteria being tested
        self.value = value  # value necessary to get a true result
        self.children = []
        self.group_a_instances = group_a_size
        self.group_b_instances = group_b_size

    def add_child(self, node):
        self.children.append(node)

    def to_string(self):
        if self.group_a_instances > 0 and self.group_b_instances > 0:
            print("[A%d <= %f]" % (self.feature, self.value))
            print(" ", end=" ")
        elif self.group_a_instances > 0:
            print("leaf: [{1.0: %d}]" % self.group_a_instances)
            print(" ", end=" ")
        else:
            print("leaf: [{0.0: %d}]" % self.group_b_instances)
            print(" ", end=" ")


def build_thresholds_for_attribute_values(data, attribute_index):
    attribute_col = data[:, attribute_index]
    sorted_values = np.sort(np.unique(attribute_col))
    thresholds = []
    for value_index in range(len(sorted_values) - 1):
        pair_average = np.average((sorted_values[value_index], sorted_values[value_index + 1]), axis=0)
        thresholds.append(pair_average)
    return thresholds


def calc_weighted_average_by_attribute(data, attribute_index, threshold, impurity):
    group_a_instances, group_b_instances, sv_a, sv_b = split_data(data, attribute_index, threshold)
    S = sv_a + sv_b
    weighted_average = (np.divide(sv_a, S) * impurity(group_a_instances)) + \
                       (np.divide(sv_b, S) * impurity(group_b_instances))
    return weighted_average


def split_data(data, attribute_index, threshold):
    group_a_rows_indices = []
    group_b_rows_indices = []
    for i in range(data.shape[0]):
        if data[i][attribute_index] <= threshold:
            group_a_rows_indices.append(i)
        else:
            group_b_rows_indices.append(i)
    return data[group_a_rows_indices, :], data[group_b_rows_indices, :], \
           len(group_a_rows_indices), len(group_b_rows_indices)


def find_best_information_gain_params(data, impurity):
    current_impurity = impurity(data)
    print(current_impurity)
    best_information_gain = 0
    best_attribute_index = 0
    best_threshold = 0
    for attribute_index in range(data.shape[1]):
        thresholds = build_thresholds_for_attribute_values(data, attribute_index)
        for threshold in thresholds:
            weighted_average = calc_weighted_average_by_attribute(data, attribute_index, threshold, impurity)
            information_gain = current_impurity - weighted_average
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_attribute_index = attribute_index
                best_threshold = threshold
    return best_information_gain, best_attribute_index, best_threshold


def remove_attribute_column(data, attribute_index):
    return np.delete(data, attribute_index, axis=1)


def build_tree(data, impurity):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    node_impurity = impurity(data)
    if node_impurity == 0:
        root = DecisionNode(None, None, 0, 0)
    elif data.shape[1] > 1:
        information_gain, attribute_index, threshold = find_best_information_gain_params(data, impurity)
        group_a_instances, group_b_instances, group_a_size, group_b_size = split_data(data, attribute_index, threshold)
        root = DecisionNode(attribute_index, threshold, group_a_size, group_b_size)
        # print("(%d,%d)" % (root.group_a_instances, root.group_b_instances))
        group_a_instances = remove_attribute_column(group_a_instances, attribute_index)
        group_b_instances = remove_attribute_column(group_b_instances, attribute_index)
        root.add_child(build_tree(group_a_instances, impurity))
        root.add_child(build_tree(group_b_instances, impurity))
    else:
        root = DecisionNode(None, None, 0, 0)
    return root


def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred


def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy


def print_tree(node):
    """
    prints the tree according to the example in the notebook

    Input:
    - node: a node in the decision tree

    This function has no return value
    """
    node.to_string();
    for child_node in node.children:
        child_node.to_string();
