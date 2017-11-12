import numpy as np
from visual import print_progress, visualize_tsp
from util import best_matching_unit, euclidean_distance, decay_function
from tsp import create_tsp_solution
import math

tensor = np.array
np.random.seed(123)


class SOM:
    def __init__(self, cases: tensor, epochs: int = 100, weight_scale: int = 7, decay: str = "exp"):
        self.cases = cases
        self.weights = np.random.normal(self.cases.min(), self.cases.max(), size=self.cases.shape) * weight_scale
        self.epochs = epochs
        self.init_learning_rate = 0.65
        self.init_radius = len(self.cases) / 2
        self.radius_constant = self.epochs / np.log(self.init_radius)
        self.decay_function = decay_function(decay)

    def get_neighbourhood(self, index: int, radius: float) -> tensor:
        radius = int(math.ceil(radius))
        result = []
        for i in range(1, radius):
            result.append((index + i) % len(self.weights))
            result.append((index - i) % len(self.weights))
        return result

    def train(self) -> tensor:
        examples = self.cases
        for epoch in range(self.epochs):
            learning_rate = self.decay_function(self.init_learning_rate, epoch, self.epochs)
            radius = int(self.decay_function(self.init_radius, epoch, self.radius_constant))
            print_progress(epoch, self.epochs)
            solution = create_tsp_solution(self.cases, self.weights)
            visualize_tsp(solution, self.weights)
            for example in examples:
                bmu = best_matching_unit(vec=example, weights=self.weights)
                self.weights[bmu] = self.weights[bmu] + learning_rate * (example - self.weights[bmu])
                neighbours = self.get_neighbourhood(bmu, radius)
                for i in neighbours:
                    distance = euclidean_distance(tensor([bmu]), tensor([i]))
                    influence = np.exp(-distance ** 2 / (2 * radius ** 2))
                    self.weights[i] = self.weights[i] + influence * learning_rate * (example - self.weights[i])
