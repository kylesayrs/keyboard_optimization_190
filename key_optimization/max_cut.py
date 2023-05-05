import scipy
import numpy
import cvxpy as cp
import matplotlib.pyplot as plt


def solve_sdp(W: numpy.ndarray, k: int):
    n = W.shape[0]
    Y = cp.Variable((n, n), symmetric=True)
    ones_matrix = numpy.ones((n, n))

    objective = cp.Maximize(
        cp.sum(cp.sum(cp.multiply(W, (ones_matrix - Y))))
    )
    constraints = [
        Y >> 0,
        cp.diag(Y) == 1,
        Y >= (-1 / (k - 1)),
        cp.sum(cp.sum(Y)) <= 0,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    X = decompose_gram_matrix(Y.value)[:(k - 1)]

    return X


def decompose_gram_matrix(X: numpy.ndarray):
    eigen_values, eigen_vectors = numpy.linalg.eig(X)
    assert all(eigen_values >= -1e-04)  # tolerate small negative eigenvalues
    eigen_values = numpy.abs(eigen_values)

    D = numpy.zeros(X.shape)
    numpy.fill_diagonal(D, eigen_values)
    D_sqrt = numpy.sqrt(D)

    U = eigen_vectors

    P = D_sqrt @ U.T
    return P


def random_projection_method(X: numpy.ndarray, k: int):
    random_vectors = numpy.random.randn(X.shape[0], 2)
    similaries = X.T @ random_vectors

    return numpy.argmax(similaries, axis=1)


def clustering_method():
    raise NotImplementedError


def vertex_projection_method():
    raise NotImplementedError


def greedy_swap():
    raise NotImplementedError


def visualize_points(points: numpy.ndarray):
    # expand to 3 dims if necessary
    if len(points) < 3:
        extra_dims = [
            [numpy.zeros(points.shape[1])]
            for _ in range(3 - len(points))
        ]
        points = numpy.concatenate((points, *extra_dims))

    # TSNE if necessary
    if len(points) > 3:
        raise NotImplementedError("TSNE is not implemented yet")

    # create figure
    figure = plt.figure(figsize = (5, 5))
    axis = figure.add_subplot(111, projection="3d")
    
    # create plot
    axis.scatter3D(*(points), s=40, color="orange", marker="o")
    for point_i, point in enumerate(points.T):
        axis.text(*point, chr(97 + point_i))

    plt.show()


if __name__ == "__main__":
    k = 2
    W = numpy.loadtxt("demo_bigram_matrix.txt")
    #W = numpy.loadtxt("bigram_matrix.txt")

    X = solve_sdp(W, k)
    #visualize_points(X)

    class_assigments = random_projection_method(X, k)

    greedy_swap()
