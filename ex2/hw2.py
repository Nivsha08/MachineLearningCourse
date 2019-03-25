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

    p_value = 1

    def __init__(self, feature, value, group_a_size, group_b_size):
        self.feature = feature  # column index of criteria being tested
        self.value = value  # value necessary to get a true result
        self.labels_split = {}
        self.children = []
        self.group_a_instances = group_a_size
        self.group_b_instances = group_b_size

    def add_child(self, node):
        self.children.append(node)

    def to_string(self):
        if len(self.children) == 0:
            for label in self.labels_split:
                print("leaf: [{%d: %d}]" % (label, self.labels_split[label]))
        else:
            print("[A%d <= %f]" % (self.feature, self.value))


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
    weighted_average = ((sv_a / S) * impurity(group_a_instances)) + \
                       ((sv_b / S) * impurity(group_b_instances))
    return weighted_average


def split_data(data, attribute_index, threshold):
    group_a_data = data[data[:, attribute_index] <= threshold]
    group_b_data = data[data[:, attribute_index] > threshold]
    return group_a_data, group_b_data, group_a_data.shape[0], group_b_data.shape[0]


def find_best_information_gain_params(data, impurity):
    current_impurity = impurity(data)
    best_information_gain = 0
    best_attribute_index = 0
    best_threshold = 0
    for attribute_index in range(data.shape[1] - 1):
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


def set_node_labels_split_information(node, data):
    labels, count_of_labels = np.unique(data[:, -1], return_counts=True)
    for i in range(len(count_of_labels)):
        node.labels_split[labels[i]] = count_of_labels[i]


def compute_chi_square_value_group_a(attribute_index, threshold, data, D):
    count_of_labels = np.unique(data[:, -1], return_counts=True)[1]
    group_a_probability = count_of_labels[0] / D
    group_b_probability = 1 - group_a_probability
    df = data[data[:, attribute_index] <= threshold].shape[0]
    pf = data[data[:, attribute_index] <= threshold]
    pf = pf[pf[:, -1] == 0].shape[0]
    nf = data[data[:, attribute_index] <= threshold]
    nf = nf[nf[:, -1] == 1].shape[0]
    E0 = df * group_a_probability
    E1 = df * group_b_probability
    if E0 != 0 and E1 != 0:
        chi_value = (np.power((pf - E0), 2) / E0) + (np.power((nf - E1), 2) / E1)
        return chi_value
    else:
        return 0


def compute_chi_square_value_group_b(attribute_index, threshold, data, D):
    count_of_labels = np.unique(data[:, -1], return_counts=True)[1]
    group_b_probability = count_of_labels[0] / D
    group_a_probability = 1 - group_b_probability
    df = data[data[:, attribute_index] > threshold].shape[0]
    pf = data[data[:, attribute_index] > threshold]
    pf = pf[pf[:, -1] == 0].shape[0]
    nf = data[data[:, attribute_index] > threshold]
    nf = nf[nf[:, -1] == 1].shape[0]
    E0 = df * group_a_probability
    E1 = df * group_b_probability
    if E0 != 0 and E1 != 0:
        chi_value = (np.power((pf - E0), 2) / E0) + (np.power((nf - E1), 2) / E1)
        return chi_value
    else:
        return 0


def compute_node_chi_value(attribute_index, threshold, group_a_instances, group_b_instances, group_a_size,
                           group_b_size):
    D = group_b_size + group_a_size
    return compute_chi_square_value_group_a(attribute_index, threshold, group_a_instances, D) + \
           compute_chi_square_value_group_b(attribute_index, threshold, group_b_instances, D)


def chi_square_test(node_chi_value, p_value):
    return p_value == 1 or node_chi_value > chi_table[p_value]


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
    if data is not None and data.shape[1] > 1:
        information_gain, attribute_index, threshold = find_best_information_gain_params(data, impurity)
        group_a_instances, group_b_instances, group_a_size, group_b_size = split_data(data, attribute_index, threshold)
        root = DecisionNode(attribute_index, threshold, group_a_size, group_b_size)
        group_a_instances = remove_attribute_column(group_a_instances, attribute_index)
        group_b_instances = remove_attribute_column(group_b_instances, attribute_index)
        if group_a_size == 0 or group_b_size == 0:
            set_node_labels_split_information(root, data)
            return root
        else:
            root_chi_value = compute_node_chi_value(attribute_index, threshold, group_a_instances, group_b_instances,
                                                    group_a_size, group_b_size)
            root.chi_value = root_chi_value
            if not chi_square_test(root.chi_value, root.p_value):
                set_node_labels_split_information(root, data)
                return root
            else:
                root.add_child(build_tree(group_a_instances, impurity))
                root.add_child(build_tree(group_b_instances, impurity))
                return root


def calc_tree_accuracy_by_p_value(training_data, test_data, impurity, current_p_value):
    DecisionNode.p_value = current_p_value
    tree_root = build_tree(training_data, impurity)
    current_accuracy = calc_accuracy(tree_root, test_data)
    return current_accuracy

def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    predict_label = -1
    if len(node.labels_split) == 1:
        predict_label = list(node.labels_split.keys())[0]
    elif len(node.children) != 0:
        split_attribute = node.feature
        split_threshold = node.value
        instance_value_of_attribute = instance[split_attribute]
        if instance_value_of_attribute <= split_threshold:
            predict_label = predict(node.children[0], instance)
        else:
            predict_label = predict(node.children[1], instance)
    return predict_label


def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    success_predictions_count = 0
    total_instances = dataset.shape[0]
    for instance_index in range(total_instances):
        actual_label = dataset[instance_index][-1]
        model_prediction = predict(node, dataset[instance_index, :])
        if model_prediction == actual_label:
            success_predictions_count += 1
    accuracy = (success_predictions_count / total_instances) * 100
    return accuracy


def print_tree_acc(node, acc):
    node.to_string();
    if len(node.children) > 0:
        for child_node in node.children:
            print("%s" % ("   " * acc), end="")
            print_tree_acc(child_node, (acc + 1))


def print_tree(node):
    """
    prints the tree according to the example in the notebook

    Input:
    - node: a node in the decision tree

    This function has no return value
    """
    print_tree_acc(node, 1)
