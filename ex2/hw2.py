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


def compute_chi_statistics(data, attribute_index, threshold):
    probability_of_label_zero = (data[data[:, data.shape[1] - 1] == 0].shape[0]) / data.shape[0]
    probability_of_label_one = 1 - probability_of_label_zero
    less_than_threshold_instances = data[data[:, attribute_index] <= threshold]
    df_less_than_threshold = less_than_threshold_instances.shape[0]
    less_than_threshold_and_label_zero = less_than_threshold_instances[
        less_than_threshold_instances[:, less_than_threshold_instances.shape[1] - 1] == 0].shape[0]
    less_than_threshold_and_label_one = less_than_threshold_instances.shape[0] - less_than_threshold_and_label_zero
    E0_less_than_threshold = df_less_than_threshold * probability_of_label_zero
    E1_less_than_threshold = df_less_than_threshold * probability_of_label_one
    chi_less_than_threshold = (np.power((less_than_threshold_and_label_zero - E0_less_than_threshold), 2) \
                              / E0_less_than_threshold) + (np.power(
        (less_than_threshold_and_label_one - E1_less_than_threshold), 2) / E1_less_than_threshold)

    more_than_threshold_instances = data[data[:, attribute_index] > threshold]
    df_more_than_threshold = more_than_threshold_instances.shape[0]
    more_than_threshold_and_label_zero = more_than_threshold_instances[
        more_than_threshold_instances[:, more_than_threshold_instances.shape[1] - 1] == 0].shape[0]
    more_than_threshold_and_label_one = more_than_threshold_instances.shape[0] - more_than_threshold_and_label_zero
    E0_more_than_threshold = df_more_than_threshold * probability_of_label_zero
    E1_more_than_threshold = df_more_than_threshold * probability_of_label_one
    chi_more_than_threshold = (np.power((more_than_threshold_and_label_zero - E0_more_than_threshold), 2) \
                               / E0_more_than_threshold) + (np.power(
        (more_than_threshold_and_label_one - E1_more_than_threshold), 2) / E1_more_than_threshold)

    return chi_less_than_threshold + chi_more_than_threshold


def build_tree(data, impurity, p_value):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    information_gain, attribute_index, threshold = find_best_information_gain_params(data, impurity)
    group_a_instances, group_b_instances, group_a_size, group_b_size = split_data(data, attribute_index, threshold)
    root = DecisionNode(attribute_index, threshold, group_a_size, group_b_size)
    set_node_labels_split_information(root, data)
    if group_a_size == 0 or group_b_size == 0:
        return root
    else:
        if p_value == 1:
            root.add_child(build_tree(group_a_instances, impurity, p_value))
            root.add_child(build_tree(group_b_instances, impurity, p_value))
            return root
        else:
            root_chi_value = compute_chi_statistics(data, attribute_index, threshold)
            if root_chi_value > chi_table[p_value]:
                root.add_child(build_tree(group_a_instances, impurity, p_value))
                root.add_child(build_tree(group_b_instances, impurity, p_value))
                return root
            else:
                return root


def majority_of_labels_in_node(node):
    return max(node.labels_split, key=node.labels_split.get)


def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    if len(node.children) == 0:
        if node.group_a_instances == 0 or node.group_b_instances == 0:
            predict_label = list(node.labels_split.keys())[0]
            return predict_label
        else:
            return majority_of_labels_in_node(node)
    else:
        split_attribute = node.feature
        split_threshold = node.value
        instance_value_of_attribute = instance[split_attribute]
        if instance_value_of_attribute <= split_threshold:
            predict_label = predict(node.children[0], instance)
            return predict_label
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


def count_tree_nodes(root):
    if len(root.children) == 0:
        return 1
    else:
        return 1 + count_tree_nodes(root.children[0]) + count_tree_nodes(root.children[1])


def get_tree_potential_parents_accumulator(node, acc_list):
    if len(node.children) == 0:
        return
    else:
        for child in node.children:
            if len(child.children) == 0:
                acc_list.append(node)
            else:
                get_tree_potential_parents_accumulator(child, acc_list)


def get_tree_potential_parents(tree):
    acc_list = []
    get_tree_potential_parents_accumulator(tree, acc_list)
    return acc_list


def prune_parent(root, parent):
    if len(root.children) == 0:
        return
    elif root == parent:
        root.children = []
        return
    else:
        for child in root.children:
            prune_parent(child, parent)


def predict_post_pruning(node, node_to_treat_as_a_leaf, instance):
    if len(node.children) == 0:
        if node.group_a_instances == 0 or node.group_b_instances == 0:
            predict_label = list(node.labels_split.keys())[0]
            return predict_label
        else:
            return majority_of_labels_in_node(node)
    if node == node_to_treat_as_a_leaf:
        return majority_of_labels_in_node(node)
    else:
        split_attribute = node.feature
        split_threshold = node.value
        instance_value_of_attribute = instance[split_attribute]
        if instance_value_of_attribute <= split_threshold:
            predict_label = predict_post_pruning(node.children[0], node_to_treat_as_a_leaf, instance)
            return predict_label
        else:
            predict_label = predict_post_pruning(node.children[1], node_to_treat_as_a_leaf, instance)
            return predict_label


def calc_post_pruning_accuracy(node, node_to_treat_as_a_leaf, dataset):
    success_predictions_count = 0
    total_instances = dataset.shape[0]
    for instance_index in range(total_instances):
        actual_label = dataset[instance_index][-1]
        model_prediction = predict_post_pruning(node, node_to_treat_as_a_leaf, dataset[instance_index, :])
        if model_prediction == actual_label:
            success_predictions_count += 1
    accuracy = (success_predictions_count / total_instances) * 100
    return accuracy


def find_best_node_to_prune(tree, potential_parents, dataset):
    best_accuracy = 0
    best_parent = None
    for parent in potential_parents:
        accuracy = calc_post_pruning_accuracy(node=tree, node_to_treat_as_a_leaf=parent, dataset=dataset)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_parent = parent
    return best_accuracy, best_parent


def print_tree_acc(node, acc):
    node.to_string()
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
