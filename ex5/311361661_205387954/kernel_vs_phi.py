# -----------------------
#   PART 1 - QUESTION 4
# -----------------------
import time
import sklearn.metrics.pairwise as sk_kernel
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# draw 20,000 random vectors with 20 dimensions
num_of_vectors = 20000
n = 20
vectors = np.random.rand(num_of_vectors, n)

# calculating the gram matrix (M[i][j] = K(Xi, Xj))
start_time = time.time()
gram_matrix = np.square(np.matmul(vectors, vectors.T) + 1)
end_time = time.time()
print("Gram matrix with kernel function - Total time: %s seconds" % (end_time - start_time))

# mapping the vectors from the lower dimension (20) to the higher dimension (231)
phi = PolynomialFeatures(degree=2)
mapped_vectors = phi.fit_transform(vectors)

coef_list = []
i = 0
while i <= n:
    j = i
    while j <= n:
        if i == j:
            coef_list.append(1)
        else:
            coef_list.append(np.sqrt(2))
        j += 1
    i += 1
coef_vector = np.array(coef_list)

mapped_vectors = np.multiply(mapped_vectors, coef_vector)

# calculating the mapping matrix (M[i][j] = phi(x)phi(y))
start_time = time.time()
phi_matrix = np.matmul(mapped_vectors, mapped_vectors.T)
end_time = time.time()
print("Gram matrix with Phi function - Total time: %s seconds" % (end_time - start_time))

# comparing the matrices
np.allclose(gram_matrix, phi_matrix)

