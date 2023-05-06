#!python3
import argparse

import numpy
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


parser = argparse.ArgumentParser()
parser.add_argument("weight_file_path")           # positional argument
parser.add_argument("-k", type=int, default=2)
parser.add_argument("-clustering_method", default="random", choices=["random", "clustering", "vertices"])
parser.add_argument("-num_iterations", default=100)


def solve_sdp(W: numpy.ndarray, k: int, equal_partition: bool = True):
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
        (cp.sum(cp.sum(Y)) <= 0) if equal_partition else True
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    X = _decompose_gram_matrix(Y.value)[:(k - 1)]

    return X


def _decompose_gram_matrix(Y: numpy.ndarray):
    _, eigen_values, U_t = numpy.linalg.svd(Y, hermitian=True)

    assert all(eigen_values >= -1e-04)  # tolerate small negative eigenvalues
    eigen_values = numpy.abs(eigen_values)

    D = numpy.zeros(Y.shape)
    numpy.fill_diagonal(D, eigen_values)
    D_sqrt = numpy.sqrt(D)

    return D_sqrt @ U_t


def _nd_argsort(x: numpy.ndarray):
    return numpy.dstack(numpy.unravel_index(numpy.argsort(x.ravel()), x.shape))[0]


def random_projection_method(X: numpy.ndarray, k: int):
    random_vectors = numpy.random.randn(X.shape[0], k)
    similaries = X.T @ random_vectors

    return numpy.argmax(similaries, axis=1)


def clustering_method(X: numpy.ndarray, k: int):
    raise NotImplementedError


def vertex_projection_method(X: numpy.ndarray, k: int):
    raise NotImplementedError

    # NOTE: This decomposition is not necessary aligned with the decomposition
    # of the points since there is ambiguity in choice of basis vectors when
    # computing the eigenvectors
    vertex_gram_matrix = numpy.full((k, k), (-1 / (k - 1)))
    numpy.fill_diagonal(vertex_gram_matrix, 1)
    vertex_gram_matrix = vertex_gram_matrix + 0.001 * numpy.random.randn(k, k)
    vertices = _decompose_gram_matrix(vertex_gram_matrix)[:(k - 1)]

    random_vectors = numpy.random.randn(X.shape[0], k)
    similaries = X.T @ random_vectors

    return numpy.argmax(similaries, axis=1)


def _get_swap_rewards(partition: numpy.ndarray, W: numpy.ndarray, num_partitions: int):
    num_elements = W.shape[0]

    swap_rewards = [[None for _ in range(num_partitions)] for _ in range(num_elements)]
    for element_index in range(num_elements):
        for partition_index in range(num_partitions):
            indicies_not_in_partition = partition != partition_index
            swap_rewards[element_index][partition_index] = (
                sum(W[element_index][indicies_not_in_partition])
            )

    return numpy.array(swap_rewards)


def greedy_swap(partition: numpy.ndarray, W: numpy.ndarray, num_partitions: int):
    num_elements = W.shape[0]
    min_partition_size = num_elements // num_partitions

    while True:
        swap_rewards = _get_swap_rewards(partition, W, num_partitions)
        for element_index, partition_index in reversed(_nd_argsort(swap_rewards)):
            element_partition_index = partition[element_index]
            if (
                partition[element_index] != partition_index and
                numpy.count_nonzero(partition == element_partition_index) > (min_partition_size + 1) and
                numpy.count_nonzero(partition == partition_index) <= min_partition_size
            ):
                partition[element_index] = partition_index
                break
        
        else:
            break

    return partition


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
        perplexity = min(8, points.shape[1] - 1)
        points = TSNE(n_components=3, perplexity=perplexity).fit_transform(points.T).T

    # create figure
    figure = plt.figure(figsize = (5, 5))
    axis = figure.add_subplot(111, projection="3d")
    
    # create plot
    axis.scatter3D(*points, s=40, color="orange", marker="o")
    for point_i, point in enumerate(points.T):
        axis.text(*point, chr(97 + point_i))

    plt.show()


def _freq_dict(partition: numpy.ndarray, k: int):
    return {
        partition_i: numpy.count_nonzero(partition == partition_i)
        for partition_i in range(k)
    }


def _letter_dict(partition: numpy.ndarray, k: int):
    dictionary = {
        partition_i: []
        for partition_i in range(k)
    }

    for letter_i, partition_index in enumerate(partition):
        dictionary[partition_index].append(chr(97 + letter_i))

    return dictionary


def get_reward(partition: numpy.ndarray, W: numpy.ndarray, num_partitions: int):
    swap_rewards = _get_swap_rewards(partition, W, num_partitions)
    return sum([
        swap_rewards[element_index, element_partition]
        for element_index, element_partition in enumerate(partition)
    ])


def get_reward_upper_bound(W: numpy.ndarray, num_partitions: int):
    num_elements = W.shape[0]
    min_partition_size = num_elements // num_partitions

    upper_bound = 0
    for weight in W:
        sorted_indices = numpy.argsort(-1 * weight)
        upper_bound += sum(weight[sorted_indices][:(num_elements - min_partition_size)])

    return upper_bound


def max_cut(args):
    # load data
    W = numpy.loadtxt(args.weight_file_path)

    # step 1
    X = solve_sdp(W, args.k, equal_partition=True)
    visualize_points(X)

    # apply projections and take save the best
    best_partition = None
    best_partition_reward = numpy.NINF
    for iteration_i in range(args.num_iterations):
        # step 2
        if args.clustering_method == "random":
            partition = random_projection_method(X, args.k)
        elif args.clustering_method == "clustering":
            partition = clustering_method(X, args.k)
        elif args.clustering_method == "vertices":
            partition = vertex_projection_method(X, args.k)
        else:
            raise ValueError(f"Unknown clustering method {args.clustering_method}")

        # step 3
        greedy_swap(partition, W, args.k)

        # save best
        partition_reward = get_reward(partition, W, args.k)
        if partition_reward > best_partition_reward:
            best_partition = partition.copy()
            best_partition_reward = partition_reward

    upper_bound = get_reward_upper_bound(W, args.k)
    return best_partition, best_partition_reward, upper_bound


if __name__ == "__main__":
    args = parser.parse_args()
    partition, partition_reward, upper_bound = max_cut(args)

    print(f"Best partition: {partition}")
    print(f"reward/upper_bound: {partition_reward} / {upper_bound}")
    print(_letter_dict(partition, args.k))
