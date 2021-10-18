import mlrose_hiive as mlrose
from mlrose_hiive import RHCRunner, SARunner, GARunner, MIMICRunner, NNGSRunner, ExpDecay, GeomDecay, FourPeaks, FlipFlop, SixPeaks
from mlrose_hiive import gradient_descent as gd, simulated_annealing as sa, random_hill_climb as rhc, genetic_alg as ga, mimic
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from utils import load_grades_data

random_seed = 23

config = {
    "gd": {
        "iteration_list": [1000],
        "max_attempts": 50
    },
    "rhc": {
        "iteration_list": [1000],
        "max_attempts": 50,
        "restart_list": [3]
    },
    "sa": {
        "iteration_list": [1000],
        "max_attempts": 50,
        "temperature_list": [0.01]
    },
    "ga": {
        "iteration_list": [1000],
        "max_attempts": 50,
        "population_sizes": [25],
        "mutation_rates": [0.001]
    }
}


def run(config, OUTPUT_FOLDER="output"):
    X, y = load_grades_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=random_seed)
    shared_params = {
        'learning_rate_init': [0.05],
        'activation': [mlrose.relu],
        'hidden_layer_sizes': [[60]]
    }
    runner_params = {
        "x_train": X_train,
        "y_train": y_train,
        "x_test": X_test,
        "y_test": y_test,
        "experiment_name": "nn",
        "clip_max": 1,
        "max_attempts": 50,
        "n_jobs": 5,
        "seed": random_seed,
        "cv": 2
    }

    fitness_results = {}
    time_results = {}
    acc_results = {}
    curves_results = {}

    # gradient dececent
    # if not config['gd']['skip']:
    print("Running gd nn...")
    it_list = config['gd']['iteration_list']
    shared_params['max_iters'] = [max(config['gd']['iteration_list'])]
    gd_params = shared_params | {"learning_rate_init": [0.002]}
    gd_nnr = NNGSRunner(algorithm=gd, grid_search_parameters=gd_params,
                        iteration_list=it_list, **runner_params)

    start_time = time.time()
    run_stats, curves, cv_results, grid_search_cv = gd_nnr.run()
    running_time = time.time() - start_time

    run_stats = run_stats[run_stats['Iteration'] != 0]
    run_stats = run_stats.query("Fitness == Fitness.min()")
    run_stats.reset_index(inplace=True, drop=True)

    curves_results['Gradient Descent'] = curves['Fitness']
    curves.plot(title="GD Fitness vs Iterations",
                xlabel="Iterations", ylabel="Fitness", x="Iteration", y="Fitness")
    plt.savefig(f"{OUTPUT_FOLDER}/gd-nn-fitness.png")

    best_fitness = run_stats.iloc[0].Fitness
    fitness_results['Gradient Descent'] = best_fitness
    time_results['Gradient Descent'] = running_time

    acc_score = cv_results['mean_train_score'][0]
    acc_results["Gradient Descent"] = acc_score

    print(
        f"Fitness: {best_fitness}\tTime: {running_time} seconds\taccuracy: {acc_score}\n")

    # # randomized hill climbing
    # if not config['rhc']['skip']:
    print("Running rhc nn...")
    rhc_params = shared_params | {"restarts": config['rhc']['restart_list']}
    rhc_params = rhc_params | {"learning_rate_init": [0.7]}

    rhc_params['max_iters'] = [max(config['rhc']['iteration_list'])]
    it_list = config['rhc']['iteration_list']
    rhc_nnr = NNGSRunner(algorithm=rhc, grid_search_parameters=rhc_params,
                         iteration_list=it_list, **runner_params)

    start_time = time.time()
    run_stats, curves, cv_results, grid_search_cv = rhc_nnr.run()
    running_time = time.time() - start_time

    run_stats = run_stats[run_stats['Iteration'] != 0]
    run_stats = run_stats.query("Fitness == Fitness.min()")
    run_stats.reset_index(inplace=True, drop=True)
    curves = curves.query(
        f"current_restart == {run_stats['current_restart'][0]}")
    curves.reset_index(inplace=True, drop=True)

    curves_results['RHC'] = curves['Fitness']
    curves.plot(title="RHC Fitness vs Iterations",
                xlabel="Iterations", ylabel="Fitness", x="Iteration", y="Fitness")
    plt.savefig(f"{OUTPUT_FOLDER}/rhc-nn-fitness.png")

    best_fitness = run_stats.iloc[0].Fitness
    fitness_results['RHC'] = best_fitness
    time_results['RHC'] = running_time
    restarts = run_stats.iloc[0].restarts

    acc_score = cv_results['mean_train_score'][0]
    acc_results["RHC"] = acc_score

    print(
        f"Fitness: {best_fitness}\tTime: {running_time} seconds\taccuracy: {acc_score}\trestarts: {restarts}\n")

    print("Running sa nn...")
    temp_list = [GeomDecay(init_temp=t, min_temp=0.00000000000000000001, decay=0.01)
                 for t in config['sa']['temperature_list']]
    sa_params = shared_params | {"schedule": temp_list,
                                 "max_iters": [max(config['sa']['iteration_list'])]} | {"learning_rate_init": [0.7]}
    it_list = config['sa']['iteration_list']
    sa_nnr = NNGSRunner(algorithm=sa, grid_search_parameters=sa_params,
                        iteration_list=it_list, **runner_params)

    start_time = time.time()
    run_stats, curves, cv_results, grid_search_cv = sa_nnr.run()
    running_time = time.time() - start_time

    run_stats = run_stats[run_stats['Iteration'] != 0]
    run_stats = run_stats.query("Fitness == Fitness.min()")
    run_stats.reset_index(inplace=True, drop=True)

    curves_results['SA'] = curves['Fitness']
    curves.plot(title="SA Fitness vs Iterations",
                xlabel="Iterations", ylabel="Fitness", x="Iteration", y="Fitness")
    plt.savefig(f"{OUTPUT_FOLDER}/sa-nn-fitness.png")

    best_fitness = run_stats.iloc[0].Fitness
    fitness_results['SA'] = best_fitness
    time_results['SA'] = running_time

    acc_score = cv_results['mean_train_score'][0]
    acc_results["SA"] = acc_score

    temp = run_stats.iloc[0].schedule_init_temp
    print(
        f"Fitness: {best_fitness}\tTime: {running_time} seconds\taccuracy: {acc_score}\ttemperature: {temp}\n")

    # genetic algorithms
    print("Running ga nn...")
    ga_params = {"mutation_prob": config['ga']['mutation_rates'],
                 "pop_size": config['ga']['population_sizes']}
    ga_params = shared_params | ga_params
    ga_params['max_iters'] = [max(config['rhc']['iteration_list'])]
    it_list = config['ga']['iteration_list']
    ga_nnr = NNGSRunner(algorithm=ga, grid_search_parameters=ga_params,
                        iteration_list=it_list, **runner_params)

    start_time = time.time()
    run_stats, curves, cv_results, grid_search_cv = ga_nnr.run()
    running_time = time.time() - start_time

    run_stats = run_stats[run_stats['Iteration'] != 0]
    run_stats = run_stats.query("Fitness == Fitness.min()")
    run_stats.reset_index(inplace=True, drop=True)

    curves_results['GA'] = curves['Fitness']
    curves.plot(title="GA Fitness vs Iterations",
                xlabel="Iterations", ylabel="Fitness", x="Iteration", y="Fitness")
    plt.savefig(f"{OUTPUT_FOLDER}/ga-nn-fitness.png")

    best_fitness = run_stats.iloc[0].Fitness
    pop_size = run_stats.iloc[0].pop_size
    mut_rate = run_stats.iloc[0].mutation_prob
    fitness_results['GA'] = best_fitness
    time_results['GA'] = running_time

    acc_score = cv_results['mean_train_score'][0]
    acc_results["GA"] = acc_score

    print(f"Fitness: {best_fitness}\tTime: {running_time} seconds\taccuracy: {acc_score}\tpop_size: {pop_size}\tmut_rate: {mut_rate}\n")

    overall_results = pd.DataFrame(
        [fitness_results, time_results, acc_results])
    overall_results.index = ["Fitness", "Running Time (s)", "accuracy"]

    html = overall_results.to_html(index=True)
    with open(f"{OUTPUT_FOLDER}/nn-results.html", 'w') as fp:
        fp.write(html)
    print(overall_results)
    curves_results = pd.DataFrame(curves_results)
    curves_results.plot(title="NN Convergence: Fitness over Iterations",
                        xlabel="Iterations", ylabel="Fitness")
    plt.savefig(f"{OUTPUT_FOLDER}/nn-fitness.png")
    print()


run(config)
