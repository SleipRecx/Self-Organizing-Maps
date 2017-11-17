import numpy as np

tensor = np.array


def normalize(vec1: tensor) -> tensor:
    return vec1 / np.linalg.norm(vec1, axis=0), np.linalg.norm(vec1, axis=0)


def euclidean_distance(vec1: tensor, vec2: tensor) -> float:
    return np.linalg.norm(vec2 - vec1)


def tsp_distance(index1: int, index2: int, output_size: int) -> float:
    return min(abs(index2 - index1), output_size - abs(index2 - index1))


def best_matching_unit(vec: tensor, weights: tensor) -> int:
    distances = np.apply_along_axis(euclidean_distance, 1, weights, vec)
    return distances.argmin()


def best_matching_unit2d(vec: tensor, weights: tensor) -> tensor:
    best_distance = float("inf")
    best_index = tensor([0, 0])
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            vec2 = weights[i][j]
            result = euclidean_distance(vec, vec2)
            best_distance = min(result, best_distance)
            if best_distance == result:
                best_index = tensor([i, j])
    return best_index


def decay_function(name: str):
    def exp_decay(number: float, time: int, time_constant: int) -> float:
        return number * np.exp((-time / time_constant))

    if name == "exp":
        return exp_decay
    assert False
