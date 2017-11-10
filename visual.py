import matplotlib.pyplot as plt
import numpy as np

tensor = np.array


def visualize_tsp(cities: tensor):
    plt.title("Traveling Salesman Cities")
    fig1 = plt.figure()
    fig1.add_subplot()
    x = [c[0] for c in cities]
    y = [c[1] for c in cities]
    plt.plot(x, y, "ko")
    plt.plot(x, y)
    plt.draw()
    plt.show(block=False)
