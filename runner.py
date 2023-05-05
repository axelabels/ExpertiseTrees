import csv
import os
import shutil
import sys
from itertools import product
from time import process_time, time

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from agent import ExpertiseTree, MetaCMAB, NearestNeighbor, TreeHeuristic
from bandit import DatasetBandit
from expert import OracleExpert
from policy import GreedyPolicy
from tools import df_to_sarray, get_bounds, greedy_choice

seed = int(sys.argv[1])
arm_counts = (None,)
expert_counts = (4,32)[:]

verbose = True
work_path =f"results/"
os.makedirs(work_path, exist_ok=True)
bandit_types = (
    "mushroom",
    "adult",
    "letter",
    "nursery",
    "pendigits",
    "BNG(page-blocks,nominal,295245)",
    "BNG(glass,nominal,137781)",
    "BNG(tic-tac-toe)",
    "BNG(vote)",
    "electricity",
    "covertype",
    "kropt",
    "BNG(breast-w)",
    "BNG(page-blocks)",
    "BNG(glass)",
    "mammography",
    "eye_movements",
    "mozilla4",
    "KDDCup99",
    "MagicTelescope",
    "Click_prediction_small",
    "artificial-characters",
    "bank-marketing",
    "eeg-eye-state",
    "kr-vs-k",
    "ldpa",
    "skin-segmentation",
    "spoken-arabic-digit",
    "walking-activity",
    "volcanoes-b1",
    "creditcard",
    "Amazon_employee_access",
    "CreditCardSubset",
    "PhishingWebsites",
    "Diabetes130US",
    "numerai28.6",
    "fars",
    "shuttle",
    "Run_or_walk_information",
    "tamilnadu-electricity",
    "jungle_chess_2pcs_raw_endgame_complete",
    "MiniBooNE",
    "jannis",
    "helena",
    "microaggregation2",
)[:]
shapes = ("grid", )
smooths = ('square', )
all_sensitive_dimensions = (2,)
all_expert_context_dimensions = (2, 4, 16, 64)[:]
all_n_regions = (1, 4, 16, 64)[:]

EXPERT_CONF = 'homogeneous'


def collect_results( experiment, bandit_type, n_arms, n_experts,  n_regions, shape, sensitive_dimensions, expert_context_dimensions,smooth,h5f):
    np.random.seed(experiment)
    

    experiment_name = "_".join(map(str, (experiment, n_arms, n_experts, 0.5,
                                               bandit_type, n_regions, shape, smooth, sensitive_dimensions, expert_context_dimensions)))+".csv"
    
    if experiment_name in h5f:
        return
    n_trials = 1000

    def generate_expert(bandit, i):

        experts = [OracleExpert(bandit, GreedyPolicy())]
        return experts[i % len(experts)]

    def generate_experts(n, bandit):
        return [generate_expert(bandit, i) for i in range(n)]


    def initialize_experiment(bandit, learners, experts, seed,  expert_spread,  reset=True):
        np.random.seed(seed)
        if reset:
            bandit.reset()
            [e.reset() for e in experts]
        for learner in learners:
            learner.reset()

        if reset:
            learners[0].prior_play(experts, 0, expert_spread,
                                 bandit)

            # cache future contexts and predictions for efficiency
            bandit.cache_contexts(n_trials, seed)
            for i, e in (enumerate(experts)):
                e.cache_predictions(bandit, n_trials)

    margin=.25
    
    def initialize_expertise(
        n_experts,
        sensitive_dimensions,
        n_regions,
        smooth="square",
        shape="grid",
        precision=16,
    ):
    
        
        expertise_maps = []
        for i in range(n_experts):
          
            a = np.arange(n_regions, dtype="uint8") % 2
            a = a.reshape((int(n_regions ** 0.5), int(n_regions ** 0.5)))

            for j in range(0, int(n_regions ** 0.5), 2):
                a[j] = a[j][::-1]
            img = np.zeros((precision, precision))

            if n_regions > 1:
                images = []
                perfs = []
                for z in (2, 4, 16, 64,256):
                    if shape == "overlap":
                        if z > n_regions:
                            break
                    else:
                        if z != n_regions:
                            continue
                    a = (np.arange(z, dtype="uint8")+i) % 2
                    np.random.shuffle(a)
                    if z == 2:
                        if np.random.uniform() > 0.5:
                            a = np.repeat(a[None], 2, axis=0)
                        else:
                            a = np.repeat(a[None], 2, axis=1)
                        a = a.reshape((2, 2))
                    else:
                        a = a.reshape((int(z ** 0.5), int(z ** 0.5)))
                    images.append(
                        cv2.resize(
                            a, (precision, precision), interpolation=cv2.INTER_AREA
                        )
                    )
                assert np.mean(images)==0.5
                img[:
                ] = np.mean(images, axis=0)


            if n_regions==1:
                img[:]=np.random.choice([0,1])
            nu_cs = img.flatten() * (1 - 2 * margin) + margin
            img = img * (1 - 2 * margin) + margin

            class ExpertiseMap():
                def __init__(self,img):
                    self.img = img
                def predict(self,X):
                    r =[]
                    for x in X:
                        m,l = x*precision
                        side = precision
                        m, l = (x[:2] * (side)).astype(int)
                        honesty = self.img[m,l]
                        r.append(honesty)
                    return np.array(r)


            regr = ExpertiseMap(img)

            expertise_maps.append(regr)
        
        return expertise_maps
    np.random.seed(experiment)
    bandit = DatasetBandit(family=bandit_type)
    n_arms = bandit.k

    experts = generate_experts(n_experts, bandit)

    bounds = get_bounds(bandit.X[:,:expert_context_dimensions])

    # set up learners
    learners = []

    learners += [MetaCMAB(bandit, GreedyPolicy(), n_experts)]
    learners += [TreeHeuristic(bandit, GreedyPolicy(), n_experts)]
    
    for incremental in (False,True):
        learners += [ExpertiseTree(bandit, GreedyPolicy(), n_experts,
                                    incremental=incremental,pre_prune=True, bounds=bounds)]
   
    for size in (.1, .01,):
        learners += [NearestNeighbor(bandit, GreedyPolicy(), n_experts,
                           bounds=bounds, cluster_size=size, mode='knn')]

    class Oracle(dict):
        def reset(self):
            [a.reset() for a in self.values()]

        def short_name(self):
            return 'Oracle'

        def get_model(self,sensitive_context):
            side = int(n_regions**.5)
            m, l = (sensitive_context[:2]*(side)).astype(int)
            region_id = np.arange(n_regions).reshape((side, side))[m, l]
            return self[region_id]
    oracle_learner = Oracle()
    oracle_learner.update({region_id: MetaCMAB(bandit, GreedyPolicy(), n_experts) for region_id in range(n_regions)})

    learners += [oracle_learner]

    # set up experiment (initializes bandits and experts)
    initialize_experiment(bandit, learners, experts, experiment, EXPERT_CONF,  reset=True)
    expertise = initialize_expertise(
        n_experts=n_experts, sensitive_dimensions=sensitive_dimensions, n_regions=n_regions, shape=shape, smooth=smooth)

    # run experiment
    results = np.zeros((n_experts+len(learners)+2, n_trials))
    timings = np.zeros(len(learners))
    for t in trange(n_trials,disable=not verbose):
        # Get current context and expert advice
        context = bandit.observe_contexts(cache_index=t)
        sensitive_context = context[:expert_context_dimensions]
        
        sampled_rewards = bandit.sample(cache_index=t)
        advice = np.array([e.value_estimates(cache_index=t) for e in experts])
        optimal_advice = bandit.action_values

        dishonest_advice = 1-advice
        if type(bandit) == DatasetBandit:
            dishonest_advice[:] = greedy_choice((1-optimal_advice)*np.random.uniform(size=n_arms))[None]
            assert dishonest_advice.shape == advice.shape
        for i, e in enumerate(experts):

            honesty = expertise[i].predict(
                [sensitive_context[:sensitive_dimensions]])

            advice[i] = advice[i] if np.random.uniform() < honesty else dishonest_advice[i]
        # Choose action, log reward, and update each learner
        meta_context = {'advice': advice,
                        'base': context, 'context': sensitive_context}
        for n, learner in enumerate(learners):
  

            if type(learner) == Oracle:
                learner = learner.get_model(sensitive_context[:sensitive_dimensions])
            start = process_time()
            action = learner.choose(meta_context)

            reward = sampled_rewards[action]
            results[n_experts+n, t] = reward
            learner.observe(reward, action)
            timings[n] += process_time() - start

        # Log expert performance
        for e, expert in enumerate(experts):
            choice = np.argmax(advice[e])
            results[e, t] = sampled_rewards[choice]

        results[-1, t] = np.max(bandit.action_values)  # Best expected reward
        results[-2, t] = np.mean(bandit.action_values)  # Random policy



    data=[]
    WINDOW = n_trials
    for s in np.arange(0, n_trials, WINDOW):

        for n, learner in enumerate(learners):
            learner_score = np.mean(results[n_experts+n, s:s+WINDOW])
            data.append([s, learner.short_name(), experiment, learner_score, "value", n_arms, n_experts, EXPERT_CONF,
                        smooth, shape, n_regions, getattr(learner, 'weights_decay', 0), bandit_type, sensitive_dimensions, expert_context_dimensions,
                    timings[n],])
            if verbose: print(learner,learner_score)
        for sort_n, n in enumerate(sorted(range(n_experts), key=lambda n: np.mean(results[n]))):
            data.append([s, f"expert {sort_n}", experiment, np.mean(results[n, s:s+WINDOW]), "value", n_arms, n_experts, EXPERT_CONF,
                         smooth, shape, n_regions, 0, bandit_type, sensitive_dimensions, expert_context_dimensions,0])

        data.append([s, f"random", experiment, np.mean(results[-2, s:s+WINDOW]), "value", n_arms, n_experts, EXPERT_CONF,
                    smooth, shape, n_regions, 0, bandit_type, sensitive_dimensions, expert_context_dimensions,0])
        data.append([s, f"optimal", experiment, np.mean(results[-1, s:s+WINDOW]), "value", n_arms, n_experts, EXPERT_CONF,
                     smooth, shape, n_regions, 0, bandit_type, sensitive_dimensions, expert_context_dimensions,0])
    header = ['t', 'algorithm', 'experiment', 'performance', 'type', 'n_arms', 'n_experts', 'configuration', 
              "smooth", "shape", "n_regions", "decay", 'bandit_type', 'sensitive_dimensions', 'expert_context_dimensions', "time"]

    
    df = pd.DataFrame(data, columns=header)
    sa, saType = df_to_sarray(df)
    try:
        h5f.create_dataset(experiment_name, data=sa, dtype=saType)
    except:
        print("Failed to save",experiment_name)
bandit_order = bandit_types[(seed%len(bandit_types)):]+bandit_types[:(seed%len(bandit_types))]
configurations = np.array(list(product( (seed,), bandit_order, arm_counts, expert_counts,
                          all_n_regions, shapes, all_sensitive_dimensions, all_expert_context_dimensions,smooths)), dtype=object)
first_conf = configurations[0]
bar = tqdm(configurations,disable=not verbose)
for conf in bar:
    bar.set_description(str(conf))
    h5f_filename_tmp = work_path + f"{seed}_tmp.hdf5"
    h5f_filename = work_path + f"{seed}.hdf5"
    h5f = h5py.File(h5f_filename_tmp, "a")
    collect_results(*(list(conf)+[h5f]))
    h5f.close()  
    shutil.copy(h5f_filename_tmp, h5f_filename)
    h5f = h5py.File(h5f_filename_tmp, "a")
