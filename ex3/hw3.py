import numpy as np

np.random.seed(42)


####################################################################################################
#                                            Part A
####################################################################################################

class NaiveNormalClassDistribution:
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, std) for a class conditional normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.all_dataset = dataset
        self.class_instances = dataset[dataset[:, -1] == class_value]

    def calc_feature_mean(self, instances_feature_column):
        return np.mean(instances_feature_column, axis=0)

    def calc_feature_std(self, instances_feature_column):
        return np.std(instances_feature_column, axis=0)

    def calc_feature_conditional_distribution(self, instance, feature_index):
        feature_value = instance[feature_index]
        mean = self.calc_feature_mean(self.class_instances[:, feature_index])
        std = self.calc_feature_std(self.class_instances[:, feature_index])
        return normal_pdf(feature_value, mean, std)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.class_instances.shape[0] / self.all_dataset.shape[0]

    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        instance_likelihood = 1
        for i in range(x.shape[0] - 1):
            instance_likelihood *= self.calc_feature_conditional_distribution(x, i)
        return instance_likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()


class MultiNormalClassDistribution:
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.all_dataset = dataset
        self.class_instances = dataset[dataset[:, -1] == class_value]

    def calc_features_mean_vector(self, class_instances):
        mean_vector = []
        for i in range(class_instances.shape[1] - 1):
            mean_vector.append(np.mean(class_instances[:, i], axis=0))
        return np.array(mean_vector)

    def calc_features_cov_matrix(self, class_instances):
        number_of_features = class_instances.shape[1] - 1
        return np.cov(np.transpose(class_instances[:, 0:number_of_features]))

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.class_instances.shape[0] / self.all_dataset.shape[0]

    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        mean_vector = self.calc_features_mean_vector(self.class_instances)
        cov_matrix = self.calc_features_cov_matrix(self.class_instances)
        return multi_normal_pdf(x, mean_vector, cov_matrix)

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    coefficient = 1 / np.sqrt(2 * np.pi * np.power(std, 2))
    return coefficient * np.exp(-0.5 * np.power(((x - mean) / std), 2))


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variante normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - cov:  The covariance matrix.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    d = mean.shape[0]
    cov_matrix_det = np.linalg.det(cov)
    cov_inverse = np.linalg.inv(cov)
    distance_from_mean = x[0:d] - mean
    exponent = -0.5 * np.matmul(np.matmul(np.transpose(distance_from_mean), cov_inverse), distance_from_mean)
    return (np.power((2 * np.pi), -d / 2)) * (np.power(cov_matrix_det, -0.5)) * np.exp(exponent)


####################################################################################################
#                                            Part B
####################################################################################################
EPSILLON = 1e-6  # == 0.000001 It could happen that a certain value will only occur in the test set.


# In case such a thing occur the probability for that value will EPSILLON.

class DiscreteNBClassDistribution:
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with la place smoothing.
        
        Input
        - dataset: The dataset from which to compute the probabilites (Numpy Array).
        - class_value : Compute the relevant parameters only for instances from the given class.
        """
        self.all_dataset = dataset
        # Ni
        self.class_instances = dataset[dataset[:, -1] == class_value]
        # |Vi|

    def laplace_smoothing(self, feature_index, feature_value):
        ni = self.class_instances.shape[0]
        feature_column = self.all_dataset[:, feature_index]
        vj = len(np.unique(feature_column))
        nij = self.class_instances[self.class_instances[:, feature_index] == feature_value].shape[0]
        return (nij + 1) / (ni + vj)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.class_instances.shape[0] / self.all_dataset.shape[0]

    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        instance_likelihood = 1
        for i in range(x.shape[0] - 1):
            instance_likelihood *= self.laplace_smoothing(i, x[i])
        return instance_likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior


####################################################################################################
#                                            General
####################################################################################################            
class MAPClassifier:
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a postreiori classifier. 
        This class will hold 2 class distribution, one for class 0 and one for class 1, and will predicit and instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
        
        Input
            - An instance to predict.
            
        Output
            - 0 if the posterior probability of class 0 is higher 1 otherwise.
        """
        return 0 if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x) else 1


def compute_accuracy(testset, map_classifier):
    """
    Compute the accuracy of a given a testset and using a map classifier object.
    
    Input
        - testset: The test for which to compute the accuracy (Numpy array).
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / #testset size
    """
    successful_predictions = 0
    testset_size = testset.shape[0]
    for i in range(testset_size):
        prediction = map_classifier.predict(testset[i, :])
        if prediction == testset[i, -1]:
            successful_predictions += 1
    print("test size:", testset_size)
    print("successful predictions", successful_predictions)
    return (successful_predictions / testset_size) * 100
