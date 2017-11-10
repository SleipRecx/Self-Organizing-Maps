import numpy as np
from visual import visualize_tsp
import matplotlib.pyplot as plt

tensor = np.array
np.random.seed(123)


def parse_tsp_problem(filename: str) -> tensor:
    file = open(filename, "r")
    file.readline()
    file.readline()
    number_of_cities = int(file.readline().split()[-1])
    file.readline()
    cities = file.read().split("\n")[1:number_of_cities + 1]
    cities = list(map(lambda x: x.split(), cities))
    return tensor(list(map(lambda x: list(map(float, x)), cities)))


def normalize(vec1: tensor) -> tensor:
    return vec1 / np.linalg.norm(vec1, axis=0), np.linalg.norm(vec1, axis=0)


def euclidean_distance(vec1: tensor, vec2: tensor) -> float:
    return np.linalg.norm(vec2 - vec1)


def best_matching_unit(vec: tensor, weights: tensor) -> int:
    distances = np.apply_along_axis(euclidean_distance, 1, weights, vec)
    return distances.argmax()


def exp_decay_function(learning_rate: float, time: int, time_constant: int) -> float:
    return learning_rate * np.exp((-time / time_constant))


def get_neighbourhood(weights: tensor, index: int, radius: int) -> tensor:
    result = set()
    for i in range(1, radius):
        result.add((index + i) % len(weights))
        result.add((index - i) % len(weights))
    return result


def train(examples: tensor, weights: tensor, epochs: int = 500) -> tensor:
    learning_rate = 0.6
    time_constant = int(len(weights) / 2)
    radius = int(len(examples) / 2)
    for epoch in range(epochs):
        print("Epoch:", epoch)
        learning_rate = exp_decay_function(learning_rate, epoch, time_constant)
        for example in examples:
            bmu = best_matching_unit(vec=example, weights=weights)
            weights[bmu] = weights[bmu] + learning_rate * (example - weights[bmu])
            neighbours = get_neighbourhood(weights, bmu, radius)
            for i in neighbours:
                influence = np.exp(-(abs(i - bmu)) / 2 * len(neighbours))
                weights[i] = weights[i] + influence * learning_rate * (example - weights[i])
    return weights


def tsp_distance(cities) -> int:
    distance = 0
    for i in range(len(cities)):
        if i + 1 < len(cities) - 1:
            distance += euclidean_distance(cities[i], cities[i + 1])
        distance += euclidean_distance(cities[i], cities[0])
    return distance


tsp = parse_tsp_problem("data/3.txt")
labels = tsp[:, 0:1]
cities, scale = normalize(tsp[:, 1:])
weights = np.random.normal(cities.min(), cities.max(), size=cities.shape) * 7
weights = train(examples=cities, weights=weights)
solution_map = {}

for city in cities:
    winner = best_matching_unit(city, weights)
    if winner in solution_map:
        solution_map[winner].append(city.tolist())
    else:
        solution_map[winner] = [city.tolist()]
result = []
for key, value in solution_map.items():
    result.extend(value)

cities = cities * scale
result = np.array(result) * scale
print(tsp_distance(cities))
print(tsp_distance(result))
