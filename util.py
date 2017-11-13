import numpy as np
from typing import Callable

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


def decay_function(name: str) -> Callable:
    def exp_decay(number: float, time: int, time_constant: int) -> float:
        return number * np.exp((-time / time_constant))

    if name == "exp":
        return exp_decay
    assert False
