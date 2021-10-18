import mlrose_hiive as mlrose
from mlrose_hiive import RHCRunner, SARunner, GARunner, MIMICRunner, NNGSRunner, ExpDecay, GeomDecay, FourPeaks, FlipFlop, SixPeaks
from mlrose_hiive import simulated_annealing as sa, random_hill_climb as rhc, genetic_alg as ga, mimic
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
random_seed = 199101440


def queens_problem(n=8):
    def queens_max(state): return sum(np.arange(len(state))) - \
        mlrose.Queens().evaluate(state)
    fitness_queens = mlrose.CustomFitness(queens_max)
    prob = mlrose.DiscreteOpt(
        length=n, fitness_fn=fitness_queens, maximize=True, max_val=n)
    prob.set_mimic_fast_mode(True)
    return prob


nq_config = {
    'size': 32,
    'sizes': [8, 32, 64],
    'rhc': {
        'iteration_list': [10000],
        'max_attempts': 100,
        'restart_list': [10, 15, 20],
    },
    'sa': {
        'iteration_list': [10000],
        'max_attempts': 100,
        'temperature_list': [1.0, 1.0, 10.0, 100.0]
    },
    'ga': {
        'iteration_list': [10000],
        'max_attempts': 100,
        'population_sizes': [5, 25, 50, 100],
        'mutation_rates': [0.2, 0.1, 0.05, 0.01],
    },
    'mimic': {
        'iteration_list': [10000],
        'max_attempts': 100,
        'population_sizes': [5, 25, 50, 100],
        'keep_percent_list': [0.25, 0.1, 0.05, 0.01],
    },
}


def plot_fitness_curve(rhc, sa, ga, mimic):
    plt.figure()
    plt.plot(rhc['Iteration'], rhc['Fitness'], label='RHC')
    plt.plot(sa['Iteration'], sa['Fitness'], label='SA')
    plt.plot(ga['Iteration'], ga['Fitness'], label='GA')
    plt.plot(mimic['Iteration'], mimic['Fitness'], label='MIMIC')
    plt.title('N-Queens Fitness vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel("Fitness")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("nqueens-fitness.png")


def plot_time_curve(rhc, sa, ga, mimic):
    plt.figure()
    plt.plot(rhc['Iteration'], rhc['Time'], label='RHC')
    plt.plot(sa['Iteration'], sa['Time'], label='SA')
    plt.plot(ga['Iteration'], ga['Time'], label='GA')
    plt.plot(mimic['Iteration'], mimic['Time'], label='MIMIC')
    plt.title('N-Queens Time vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel("Time")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("nqueens-time.png")


def plot_fevals_curve(rhc, sa, ga, mimic):
    plt.figure()
    plt.plot(rhc['Iteration'], rhc['FEvals'], label='RHC')
    plt.plot(sa['Iteration'], sa['FEvals'], label='SA')
    plt.plot(ga['Iteration'], ga['FEvals'], label='GA')
    plt.plot(mimic['Iteration'], mimic['FEvals'], label='MIMIC')
    plt.title('N-Queens FEvals vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel("FEvals")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("nqueens-fevals.png")


def run():
    config = nq_config['rhc']
    problem = queens_problem(nq_config['size'])

    problem_size_results = {}

    # Run Random Hill
    rhc_runner = RHCRunner(problem, experiment_name="nqueens-rhc", seed=random_seed,
                           iteration_list=config['iteration_list'], max_attempts=config['max_attempts'], restart_list=config['restart_list'])
    print("generating flip-flop results")
    rhc_run_stats, rhc_run_curves = rhc_runner.run()
    rrs = rhc_run_stats[rhc_run_stats['Iteration'] != 0]
    best_restarts = [rrs[rrs['Restarts'].eq(
        i)]['Fitness'].idxmax() for i in config['restart_list']]
    rrs = rrs[rrs.index.isin(best_restarts)]
    rrs.reset_index(inplace=True, drop=True)
    rhc_run_stats = rrs
    best_run = rrs.query('Fitness == Fitness.max()').query(
        'Restarts == Restarts.max()').iloc[0]
    best_restarts = int(best_run['Restarts'])
    rhc_run_curves = rhc_run_curves.query(
        f"(current_restart == {best_run['current_restart']}) & (Restarts == {best_run['Restarts']})")
    rhc_run_curves.reset_index(inplace=True, drop=True)
    print("done RHC")
    print(f"RHC SCORE: {rhc_run_curves.iloc[-1]['Fitness']}")
    print(
        f"best restarts: {best_restarts}\tTime: {best_run['Time']}\tIterations:{rhc_run_curves['Iteration'].iloc[-1]}\n")

    # PROBLEM_SIZES
    max_iters = max(config['iteration_list'])
    max_attempts = config['max_attempts']
    rhc_problem_results = np.zeros(len(nq_config['sizes']))
    for i in range(len(nq_config['sizes'])):
        n = nq_config['sizes'][i]
        start_time = time.time()
        prob = queens_problem(n)
        rhc_problem_results[i] = rhc(prob, max_attempts=max_attempts,
                                     max_iters=max_iters, random_state=random_seed, restarts=best_restarts)[1]
        running_time = time.time() - start_time
        print(
            f"RHC: Completed for size {n}\ttime: {running_time}\tFitness: {rhc_problem_results[i]}")
    problem_size_results["RHC"] = rhc_problem_results

    # Run Simulated Annealing
    config = nq_config['sa']
    sa_runner = SARunner(problem, experiment_name=f"nqueens-sa", seed=random_seed, decay_list=[
                         GeomDecay], iteration_list=config['iteration_list'], max_attempts=config['max_attempts'], temperature_list=config['temperature_list'])
    sa_run_stats, sa_run_curves = sa_runner.run()
    sars = sa_run_stats[sa_run_stats['Iteration'] != 0]
    sars.reset_index(inplace=True, drop=True)
    best_run = sars.query('Fitness == Fitness.max()').iloc[0]
    best_temp = best_run['Temperature']
    sa_run_curves = sa_run_curves[sa_run_curves['Temperature']
                                  == best_run['Temperature']]
    sa_run_curves.reset_index(inplace=True, drop=True)
    print("done SA")
    print(
        f"best Temperature: {best_temp}\tTime: {best_run['Time']}\tIterations:{sa_run_curves['Iteration'].iloc[-1]}\n")
    print(f"SA SCORE: {sa_run_curves.iloc[-1]['Fitness']}")

    max_iters = max(config['iteration_list'])
    max_attempts = config['max_attempts']
    sa_problem_results = np.zeros(len(nq_config['sizes']))
    for i in range(len(nq_config['sizes'])):
        n = nq_config['sizes'][i]
        start_time = time.time()
        prob = queens_problem(n)
        sa_problem_results[i] = sa(prob, max_attempts=max_attempts, max_iters=max_iters,
                                   random_state=random_seed, schedule=GeomDecay(init_temp=best_temp.init_temp))[1]
        running_time = time.time() - start_time
        print(
            f"Completed for size {n}\ttime: {running_time}\tFitness: {sa_problem_results[i]}")
    problem_size_results["SA"] = sa_problem_results

    # Run GA
    config = nq_config['ga']
    ga_runner = GARunner(problem, experiment_name=f'nqueens-ga', seed=random_seed,
                         iteration_list=config['iteration_list'], max_attempts=config['max_attempts'], population_sizes=config['population_sizes'], mutation_rates=config['mutation_rates'])
    ga_run_stats, ga_run_curves = ga_runner.run()
    ga_run_stats = ga_run_stats[ga_run_stats['Iteration'] != 0]
    pop_size_results = ga_run_stats.groupby('Population Size')['Fitness'].max()
    mut_rate_results = ga_run_stats.groupby('Mutation Rate')['Fitness'].max()
    ga_groups = ga_run_stats.groupby(
        ['Mutation Rate', 'Population Size']).max()['Fitness']
    best_mut_rate, best_pop_size = ga_groups.idxmax()
    ga_run_curves = ga_run_curves.query(
        f"(`Mutation Rate` == {best_mut_rate}) & (`Population Size` == {best_pop_size})")
    ga_run_stats.reset_index(inplace=True, drop=True)
    ga_run_curves.reset_index(inplace=True, drop=True)
    best_time = ga_run_curves.iloc[-1]['Time']
    print(
        f'best Mutation Rate: {best_mut_rate}\tbest Population Size: {best_pop_size}\tTime: {best_time}\tIterations:{ga_run_curves["Iteration"].iloc[-1]}\n')
    print("done GA")
    print(f"GA SCORE: {ga_run_curves.iloc[-1]['Fitness']}")

    max_iters = max(config['iteration_list'])
    max_attempts = config['max_attempts']
    ga_problem_results = np.zeros(len(nq_config['sizes']))
    for i in range(len(nq_config['sizes'])):
        n = nq_config['sizes'][i]
        start_time = time.time()
        prob = queens_problem(n)
        ga_problem_results[i] = ga(prob, max_attempts=max_attempts, max_iters=max_iters,
                                   random_state=random_seed, pop_size=int(best_pop_size), mutation_prob=best_mut_rate)[1]
        running_time = time.time() - start_time
        print(
            f"Completed for size {n}\ttime: {running_time}\tFitness: {ga_problem_results[i]}")
    problem_size_results["GA"] = ga_problem_results

    # Run Mimic
    config = nq_config['mimic']
    mimic_runner = MIMICRunner(problem=problem, experiment_name=f"nqueens-mimic", seed=random_seed, use_fast_mimic=True,
                               max_attempts=100, iteration_list=config['iteration_list'], population_sizes=config['population_sizes'], keep_percent_list=config['keep_percent_list'])
    mimic_run_stats, mimic_run_curves = mimic_runner.run()
    mimic_run_stats = mimic_run_stats[mimic_run_stats['Iteration'] != 0]
    pop_size_results = mimic_run_stats.groupby('Population Size')[
        'Fitness'].max()
    keep_percent_results = mimic_run_stats.groupby('Keep Percent')[
        'Fitness'].max()
    mimic_groups = mimic_run_stats.groupby(
        ['Keep Percent', 'Population Size']).max()['Fitness']
    best_keep_percent, best_pop_size = mimic_groups.idxmax()
    mimic_run_curves = mimic_run_curves.query(
        f"(`Keep Percent` == {best_keep_percent}) & (`Population Size` == {best_pop_size})")
    mimic_run_stats.reset_index(inplace=True, drop=True)
    mimic_run_curves.reset_index(inplace=True, drop=True)
    best_time = mimic_run_curves.iloc[-1]['Time']
    print("done MIMIC")
    print(f"MIMIC SCORE: {mimic_run_curves.iloc[-1]['Fitness']}")
    print(
        f'best Keep Percentage: {best_keep_percent}\tbest Population Size: {best_pop_size}\tTime: {best_time}\tIterations:{mimic_run_curves["Iteration"].iloc[-1]}\n')

    plot_fitness_curve(rhc_run_curves, sa_run_curves,
                       ga_run_curves, mimic_run_curves)
    plot_time_curve(rhc_run_curves, sa_run_curves,
                    ga_run_curves, mimic_run_curves)
    plot_fevals_curve(rhc_run_curves, sa_run_curves,
                      ga_run_curves, mimic_run_curves)

    max_iters = max(config['iteration_list'])
    max_attempts = config['max_attempts']
    mimic_problem_results = np.zeros(len(nq_config['sizes']))
    for i in range(len(nq_config['sizes'])):
        n = nq_config['sizes'][i]
        prob = queens_problem(n)
        start_time = time.time()
        mimic_problem_results[i] = mimic(prob, max_attempts=max_attempts, max_iters=max_iters,
                                         random_state=random_seed, pop_size=int(best_pop_size), keep_pct=best_keep_percent)[1]
        running_time = time.time() - start_time
        print(
            f"Completed for size {n}\ttime: {running_time}\tFitness: {mimic_problem_results[i]}")
    problem_size_results["MIMIC"] = mimic_problem_results
    print()

    problem_size_results = pd.DataFrame(
        problem_size_results, index=nq_config['sizes'])
    problem_size_results.index.rename('Problem Size', inplace=True)

    html = problem_size_results.to_html(index=True)
    with open(f"nqueens-problem_sizes.html", 'w') as fp:
        fp.write(html)


run()
