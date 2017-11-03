import numpy as np
tensor = np.array


def parse_tsp_problem(filename: str) -> tensor:
    file = open(filename, "r")
    number_of_cities = int(file.readline().split()[-1])
    cities = file.read().split("\n")[2:number_of_cities]
    cities = list(map(lambda x: x.split(), cities))
    return tensor(list(map(lambda x: list(map(float, x)), cities)))


print(parse_tsp_problem("data/6.txt"))