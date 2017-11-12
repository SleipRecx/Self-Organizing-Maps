from visual import create_gif
from tsp import parse_tsp_problem, create_tsp_solution
from som import SOM
import imageio

def solve_tsp(filename: str):
    # Parse problem from txt file
    problem = parse_tsp_problem(filename)
    cases = problem[:, 1:]

    # Initialize self organizing map
    som = SOM(cases=cases)
    # Train
    som.train()

    # Use Som weights to generate a solution for tsp problem
    solution = create_tsp_solution(som.cases, som.weights)
    create_gif()


if __name__ == "__main__":
    solve_tsp(filename="data/2.txt")
