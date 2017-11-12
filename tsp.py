import numpy as np
from util import euclidean_distance, best_matching_unit

tensor = np.array


def parse_tsp_problem(filename: str) -> tensor:
    file = open(filename, "r")
    file.readline()
    file.readline()
    number_of_cities = int(file.readline().split()[-1])
    file.readline()
    cities = file.read().split("\n")[1:number_of_cities + 1]
    cities = list(map(lambda x: x.split(), cities))
    return tensor(list(map(lambda x: list(map(float, x)), cities)))


def tsp_distance(cities) -> int:
    distance = 0
    for i in range(0, len(cities) - 1):
        distance += euclidean_distance(cities[i], cities[i + 1])
    distance += euclidean_distance(cities[0], cities[-1])
    return distance


def create_tsp_solution(cities: tensor, weights: tensor) -> tensor:
    solution_map = {}
    for city in cities:
        winner = best_matching_unit(city, weights)
        if winner in solution_map:
            solution_map[winner].append(city.tolist())
        else:
            solution_map[winner] = [city.tolist()]
    result = []
    for i in range(len(weights)):
        if i in solution_map.keys():
            result.extend(solution_map[i])
    return tensor(result)
