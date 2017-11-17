import numpy as np
from itertools import product
from visual import print_progress, visualize_tsp
from util import best_matching_unit, tsp_distance, decay_function, euclidean_distance, best_matching_unit2d
from tsp import create_tsp_solution
from typing import List, Tuple
import math

tensor = np.array
np.random.seed(123)


class SOM:
    def __init__(self, cases: tensor, epochs: int = 300, weight_scale: int = 5):
        self.cases = cases
        self.weight_scale = weight_scale
        self.weights = np.random.uniform(self.cases.min(), self.cases.max(), size=self.cases.shape) * weight_scale
        self.epochs = epochs
        self.init_learning_rate = 0.1
        self.init_radius = len(self.cases) / 2
        self.radius_constant = self.epochs / np.log(self.init_radius)
        self.decay_function = decay_function("exp")

    def get_neighbourhood_with_overflow(self, index: int, radius: float) -> tensor:
        radius = int(math.ceil(radius))
        result = []
        for i in range(1, radius):
            result.append((index + i) % len(self.weights))
            result.append((index - i) % len(self.weights))
        return result

    def train(self):
        examples = self.cases
        for epoch in range(self.epochs):
            learning_rate = self.decay_function(self.init_learning_rate, epoch, self.epochs)
            radius = int(self.decay_function(self.init_radius, epoch, self.radius_constant))
            print_progress(epoch, self.epochs)
            if epoch % 10 == 0:
                solution = create_tsp_solution(self.cases, self.weights)
                visualize_tsp(solution, self.weights)
            for example in examples:
                bmu = best_matching_unit(vec=example, weights=self.weights)
                self.weights[bmu] = self.weights[bmu] + learning_rate * (example - self.weights[bmu])
                neighbours = self.get_neighbourhood_with_overflow(bmu, radius)
                for i in neighbours:
                    distance = tsp_distance(bmu, i, len(self.weights))
                    influence = np.exp(-distance ** 2 / (2 * radius ** 2))
                    self.weights[i] = self.weights[i] + influence * learning_rate * (example - self.weights[i])


class MSOM:
    def __init__(self, cases: tensor, labels: tensor, epochs: int = 2, output_shape: Tuple = (20, 20)):
        self.cases = cases
        self.labels = tensor(list(map(lambda x: x.argmax(), labels)))
        self.epochs = epochs
        self.classes = {}
        self.output_rows, self.output_cols = output_shape
        self.weights = np.random.uniform(0, 1, size=(self.output_rows, self.output_cols, self.cases.shape[1]))
        self.init_learning_rate = 0.7
        self.init_radius = 5
        self.radius_constant = self.epochs / np.log(self.init_radius)
        self.decay_function = decay_function("exp")

    def get_neighbourhood(self, index: tensor, radius: float) -> List[Tuple]:
        result = []
        for i in range(self.output_rows):
            for j in range(self.output_cols):
                indices = (i, j)
                distance = euclidean_distance(index, tensor(list(indices)))
                if distance <= radius:
                    if not np.array_equal(index, tensor(list(indices))):
                        result.append((indices, distance))
        return result

    def reset_winning_node(self):
        for key in product(range(self.output_rows), range(self.output_cols)):
            self.classes[key] = [0 for _ in range(len(np.unique(self.labels)))]

    def train(self):
        for epoch in range(self.epochs):
            self.reset_winning_node()
            learning_rate = self.decay_function(self.init_learning_rate, epoch, self.epochs)
            radius = self.decay_function(self.init_radius, epoch, self.radius_constant)
            print_progress(epoch, self.epochs)
            for i in range(len(self.cases)):
                example = self.cases[i]
                bmu = best_matching_unit2d(example, self.weights)
                i, j = bmu[0], bmu[1]
                self.classes[tuple(bmu)][self.labels[i]] += 1
                self.weights[i][j] = self.weights[i][j] + learning_rate * (example - self.weights[i][j])
                neighbours = self.get_neighbourhood(bmu, radius)
                for neighbour in neighbours:
                    i, j = neighbour[0]
                    distance = neighbour[1]
                    influence = np.exp(-distance ** 2 / (2 * radius ** 2))
                    self.weights[i][j] = self.weights[i][j] + influence * learning_rate * (example - self.weights[i][j])

    def predict(self, cases: tensor):
        result = []
        for case in cases:
            bmu = best_matching_unit2d(case, self.weights)
            prediction = self.classes[tuple(bmu)]
            index = prediction.index(max(prediction))
            result.append(index)
        return result


