import pickle
import os
import numpy as np

import sys
path0 = '/content/drive/MyDrive/Colab Notebooks/ARMUL/experiments in the paper'
sys.path.append(path0)
from experiment import experiment_synthetic, experiment_real


# run synthetic data experiments
def run_synthetic(idx_model, idx_seed, T = 500, L = 10): 
    # idx_model: 1 to 3
    # idx_seed: from 1 to 10

    tmp = ['vanilla', 'clustered', 'lowrank']
    setting = tmp[idx_model - 1]


    if setting == 'vanilla':
        K = 1
    if setting == 'clustered':
        K = 3
    if setting == 'lowrank':
        K = 3

    S = 10
    seed_list = np.array(range(1, S + 1)) * 10000 # random seeds
    for i in range(S):
        seed_list[i] = seed_list[i] + (idx_seed - 1) * 1000


    n, m, d = 200, 30, 50
    epsilon_list = [0, 0.2]
    delta_list = [(i / 10) for i in range(11)]

    signal_norm = 2
    sigma = 1
    norm_outliers = 2

    eta = 0.02 # step-size
    lbd_factors = [(2 * i / L) for i in range(1, L + 1)]
    ratios = np.sqrt(d / n) * np.ones(m)
    lbd_list = [factor * ratios for factor in lbd_factors]
    n_fold = 5 # for CV


    ###############################################

    path = path0 + '/results/synthetic_{}'.format(setting)
    if not os.path.exists(path):
        os.makedirs(path)

    for seed in seed_list:
        # detect if this seed has been done
        if os.path.exists( path + '/{}.txt'.format(seed) ):
            print('seed {} is done before'.format(seed))
            continue

        # run the experiment
        print('starting seed {}'.format(seed))
        test = experiment_synthetic(n, m, d)

        results = dict()
        for (u, epsilon) in enumerate(epsilon_list):
            for (v, delta) in enumerate(delta_list):
                test.getsamples(setting = setting, K = K, signal_norm = signal_norm, sigma = sigma, delta = delta, epsilon = epsilon, norm_outliers = norm_outliers, seed = seed)
                test.run(lbd_list, n_fold = n_fold, eta = eta, T = T)
                results[(epsilon, delta, 'all')] = test.err
                results[(epsilon, delta, 'S')] = test.err_S

        print('seed {} finished'.format(seed))

        with open(path + '/{}.txt'.format(seed), 'wb') as fp:
            pickle.dump(results, fp)


# run real data experiments
def run_real(idx_model, idx_seed, T = 200, L = 10):
    # idx_model: 0 to 3
    # idx_seed: from 1 to 10

    tmp = ['ITL', 'vanilla', 'clustered', 'lowrank']
    model = tmp[idx_model]

    # list of random seeds
    S = 10
    seed_list = np.array(range(1, S + 1)) * 10000 # random seeds
    for i in range(S):
        seed_list[i] = seed_list[i] + (idx_seed - 1) * 1000

    dataset = 'HAR_binary'
    PCs = 100
    eta = 0.1
    lbd_factors = [(0.5 * i / L) for i in range(1, L + 1)]

    path = path0 + '/results/real_{}'.format(model)

    if not os.path.exists(path):
        os.makedirs(path)



    if idx_model > 0:
        if model == 'vanilla':
            model_base = 'DP'
            K_list = [1]
        if model == 'clustered':
            model_base = 'clustered'
            K_list = [2, 3, 4, 5]
        if model == 'lowrank':
            model_base = 'lowrank'
            K_list = [1, 2, 3, 4, 5]

        path_base = path0 + '/results/real_{}_base'.format(model_base)
        if not os.path.exists(path_base):
            os.makedirs(path_base)


    for seed in seed_list:
        # check if this seed has been done
        check_model_seed = os.path.exists( path + '/{}.txt'.format(seed) )
        exists_baseline = (idx_model > 0)
        if exists_baseline:
            check_baseline_seed = os.path.exists( path_base + '/{}.txt'.format(seed) )
        
        if check_model_seed:
            if not exists_baseline:
                print('seed {} is done before'.format(seed))
                continue
            else: # baseline exists
                if check_baseline_seed:
                    print('seed {} is done before'.format(seed))
                    continue

        # if ITL, then the experiment is not done yet
        # if not ITL, then either ARMUL or baseline experiment is not done yet
        
        print('starting seed {}'.format(seed))
        test = experiment_real(dataset, path = '/content/drive/MyDrive/Colab Notebooks/ARMUL/', PCs = PCs, prop = None, seed = seed)

        if idx_model > 0: # vanilla, clustered or lowrank
            if check_model_seed:
                print('seed {} model {} is done before'.format(seed, model))
            else:
                # new algorithm
                results = []
                for K in K_list:
                    test.run(lbd_factors = lbd_factors, model = model, n_fold = 5, K = K, eta_global = eta, eta_local = eta, eta_B = eta, eta_Z = eta, T_global = T)
                    tmp = [test.results, test.errors_cv]
                    results.append(tmp)
                    print('seed {} K {} finished'.format(seed, K))

                with open(path + '/{}.txt'.format(seed), 'wb') as fp:
                    pickle.dump(results, fp)

            if check_baseline_seed:
                print('seed {} baseline {} is done before'.format(seed, model_base))
            else:
                # baseline
                test.run_baseline(model = model_base, K_list = K_list, eta = eta, eta_B = eta, eta_Z = eta, T = T)
                results_base = [test.results_baseline, test.errors_cv_baseline]
                
                with open(path_base + '/{}.txt'.format(seed), 'wb') as fp:
                    pickle.dump(results_base, fp)

        else: # ITL
            K_list = [1]
            test.run_baseline(model = model, K_list = K_list, eta = eta, eta_B = eta, eta_Z = eta, T = T)
            results_base = [test.results_baseline, test.errors_cv_baseline]
            
            with open(path + '/{}.txt'.format(seed), 'wb') as fp:
                pickle.dump(results_base, fp)



