import cvxpy as cp
import numpy


if __name__ == "__main__":
    num_letters = 3

    # variables
    P = cp.Variable((num_letters, num_letters))
    X = cp.
    D = cp.Constant(numpy.array([[0, 1, 2],
                                 [1, 0, 3],
                                 [2, 3, 0]]))
    W = cp.Constant(numpy.array([[0, 10, 1],
                                 [10, 0, 3],
                                 [1, 3, 0]]))

    # construct problem
    objective = cp.Minimize(cp.norm((P.T @ D @ P) * W, p="fro"))
    constraints = [
        cp.sum(P, axis=0) == 1,
        cp.sum(P, axis=1) == 1,
        P >= 0
    ]
    problem = cp.Problem(objective, constraints)

    # solve and print
    result = problem.solve()
    print(result)
    p_value_cleaned = numpy.round(P.value).astype(numpy.int32)
    print(p_value_cleaned)
