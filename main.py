from tsp import parse_tsp_problem, create_tsp_solution
from visual import visualize_tsp
from som import SOM


def log_params_result(epochs: int, init_learning_rate: float, weight_scale: int, problem: str, result_distance: int):
    def get_optimal_value(problem):
        print(problem.split("/")[1][0])
        f = open("data/best_values.txt")
        return int(f.readlines()[int(problem.split("/")[1][0]) - 1])

    with open("data/results.txt", "a") as file:
        optimal_value = get_optimal_value(problem)
        file.write("{0}\t\tepochs:{1}\t\tlearning_rate:{2}\t\tweight_scale:{3}\t\tresult:{4}\t\tmin_result:{5} \n"
                   .format(problem, epochs, init_learning_rate, weight_scale,
                           int(result_distance), (optimal_value * 0.1 + optimal_value)))


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
    visualize_tsp(solution, som.weights)


    #log_params_result(som.epochs, som.init_learning_rate, som.weight_scale, filename, tsp_distance(solution))


if __name__ == "__main__":
    solve_tsp(filename="data/8.txt")
