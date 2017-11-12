import matplotlib.pyplot as plt
from tsp import tsp_distance
from termcolor import colored
import os
import imageio

import numpy as np

tensor = np.array


def visualize_tsp(cities: tensor, weights: tensor):
    plt.clf()
    distance = tsp_distance(cities)
    plt.title("Traveling Salesman Cities, " + "Distance: " + str(int(distance)))
    cities = cities.tolist()
    cities.append(cities[0])
    cities = tensor(cities)
    x = [c[0] for c in cities]
    y = [c[1] for c in cities]
    plt.plot(x, y, "ko")
    plt.plot(x, y, color="blue")

    # weights = weights.tolist()
    # weights.append(weights[0])
    # weights = tensor(weights)
    # x = [c[0] for c in weights]
    # y = [c[1] for c in weights]
    # plt.plot(x, y, color="black", linestyle="dashed")
    files = os.listdir("images")
    files = list(filter(lambda x: "png" in x, files))
    current = 0
    if len(files) != 0:
        numbers = sorted(list(map(lambda x: int(x.split("/")[-1].split(".")[0]), files)))
        current = numbers[-1]
    name = "images/" + str(current + 1) + ".png"
    plt.savefig(name)


def print_progress(current_value, max_value):
    percentage = int((current_value / (max_value - 1)) * 100)
    color = "red"
    if percentage > 33:
        color = "yellow"
    if percentage > 66:
        color = "cyan"

    bar = "=" * int(percentage / 2) + "=>"
    complete_bar = "[" + bar + "] ".rjust(54 - len(bar), ' ')

    percent_string = str(percentage) + "%, "
    message = percent_string.ljust(4, ' ') + "Epoch: " + str(current_value + 1).ljust(10, ' ')

    output = colored(complete_bar + message, color)
    end = "\n" if percentage == 100 else ''
    print("\r" + output, end=end, flush=True)


def create_gif():
    filenames_original = os.listdir("images")
    filenames = list(filter(lambda x: "png" in x, filenames_original))
    filenames = list(map(lambda x: "images/" + x, filenames))
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('tsp.gif', images, duration=0.2)
    for file in filenames_original:
        os.remove("images/" + file)
