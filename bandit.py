from functools import lru_cache
from math import ceil
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import copy
from tools import *
from tools import normalize, get_coords,  mse, mae
from scipy.stats import wrapcauchy

import pandas as pd

MAX_ATTEMPTS_BANDIT_DISTANCE = 100
BANDIT_DISTANCE_EPSILON = .05

DATASET_SUBSET_SIZE = 100000
DISTANCE_SUBSET_SIZE = 1000

INF = float("inf")


MUSHROOM_DF = None
ADULT_DF = None

DATA_ROOT = "data/"
dic_int_columns = {}
dic_og_columns = {}
dic_mapped_columns = {}
dic_excl_columns = {}


import openml
import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
dataroot = os.path.join(DATA_ROOT,'openmldatasets')
def get_open_ml_data(name):
    d = fetch_openml(name,data_home=dataroot,as_frame=True)
    int_columns=False 
    z = d['data']
    og_columns = z.columns 
    excl_columns = d['target_names'][0]
    X = pd.get_dummies(z,drop_first=True)
    y = pd.get_dummies(d['target'],prefix=d['target_names'][0])
    df = pd.concat((X,y),axis=1)
    mapped_columns = df.columns
    k=y.shape[1]
    bernoulli=True
    expected_reward = np.mean(y.values)
    
    return int_columns, og_columns, excl_columns, mapped_columns, k,df, bernoulli, expected_reward 

@lru_cache(None)
def get_df_data(family):
    if family == 'mushroom':

        int_columns = False
        MUSHROOMS_CSV = "datasets/mushroom/data/mushroom_csv.csv"
        MUSHROOM_DF = pd.read_csv(MUSHROOMS_CSV)
        og_columns = MUSHROOM_DF.columns
        excl_columns = 'class'
        MUSHROOM_DF = pd.get_dummies(MUSHROOM_DF).rename(
            columns={'class_e': 'reward0'})
        mapped_columns = MUSHROOM_DF.columns
        
        k = 2
        df = MUSHROOM_DF
        bernoulli = True
        expected_reward=1/k
  
    elif family == "adult":

        int_columns = False

        ADULT_CSV = "datasets/adult/adult.data"
        ADULT_DF = (pd.read_csv(ADULT_CSV, sep=',', header=None, names=('age', 'workclass', 'fnlwgt', 'education',
                                                                        'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                                                                        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income')))
        ADULT_DF = ADULT_DF[ADULT_DF['occupation'] != ' ?']
        og_columns = ADULT_DF.columns
        excl_columns = ('occupation', 'fnlwgt', 'education-num',)
        order = ADULT_DF.columns
        order = [
            i for i in order if i not in excl_columns]+['occupation']
        ADULT_DF = pd.get_dummies(ADULT_DF[order])
        mapped_columns = ADULT_DF.columns

        k = 14
        df = ADULT_DF
        bernoulli = True
        expected_reward=1/k
 

    else:
        return get_open_ml_data(family)

    return int_columns, None if int_columns else og_columns, None if int_columns else excl_columns, None if int_columns else mapped_columns, k, df, bernoulli,expected_reward


class DatasetBandit():
    metric = mse

    def __init__(self, precision=200, reset=True, invert=False,  reduce_to=0, family='mushroom',  verbose=False, bernoulli=False):

        self.bernoulli = bernoulli

        
        self.int_columns, self.og_columns, self.excl_columns, self.mapped_columns, self.k, self.df, self.bernoulli,self.expected_reward = get_df_data(
            family)
        X_step = ceil(len(self.df)/DATASET_SUBSET_SIZE)
        self.X = self.df.values[::X_step, :-self.k]
        self.X = self.X[:,np.random.choice(self.X.shape[1],size=self.X.shape[1],replace=False)]
        self.X = np.hstack((np.random.uniform(size=(len(self.X),2)),self.X,))
        if self.X.shape[1]<64:
            self.X = np.hstack((self.X,np.random.uniform(size=(len(self.X),64-self.X.shape[1])),))

        self.Y = self.df.values[::X_step, -self.k:]
        self.step_size = ceil(len(self.X)/DISTANCE_SUBSET_SIZE)
        self.oracle = KNeighborsRegressor(
            n_neighbors=1,  leaf_size=500).fit(self.X, self.Y)
        self.reduced_oracle = KNeighborsRegressor(n_neighbors=1,  leaf_size=500).fit(
            self.X[::self.step_size], self.Y[::self.step_size])
        self.dims = self.X.shape[-1]
        self.cache_id = -1
        self.precision = precision
        self.invert = invert
        self.reduce_to = reduce_to
        self.cached_all_indices = {}
        self.family = family

        self.cached_contexts = None
        self.cached_values = None
        self.cached_rewards = None
        self._value_landscapes = None
        self.permutation_list = None
        self.permutations = np.arange(self.dims)
        if reset:
            self.reset()

    @property
    def value_landscapes(self):
        if self._value_landscapes is None:
            self._value_landscapes = self.get(
                self.X[::self.step_size], reduced=True)
        return self._value_landscapes

    def max_distance(self):
        return 2/self.k

    def distance(self, other):
        if self == other:
            return 0
        return DatasetBandit.metric(self.value_landscapes, other.value_landscapes)

    @property
    def grid_data(self):
        return self.value_landscapes


    def from_bandit(self, desired_distance, enforce_distance=True,  verbose=False, precomputed_permutations=True):
  
        return self
        

    def reset(self):
        self.cache_id = -1
        self._value_landscapes = None
        self.cached_contexts = None
        self.cached_all_indices = {}

    def get(self, contexts,  reduced=False):

        assert np.shape(contexts)[1:] == (
            self.dims,), f"context should be of shape {(-1,self.dims)}, but has shape {np.shape(contexts)}"

        if reduced:
            values = np.copy(self.reduced_oracle.predict(
                contexts[:,  self.permutations]))
        else:
            values = np.copy(self.oracle.predict(
                contexts[:,  self.permutations]))

        if self.invert:
            values = 1 - values
        if self.bernoulli:
            values = values/np.sum(values, axis=1)[:, np.newaxis]

        return values

    def observe_contexts(self, center=.5,  k=None, cache_index=None, steps=None, step=None):
        if cache_index is not None:
            self.contexts = self.cached_contexts[cache_index]
            self.action_values = self.cached_values[cache_index]
            self.optimal_value = np.max(self.action_values)
            return self.contexts

        if k is None:
            k = self.k

        self.contexts = np.zeros((self.dims,))
        if steps is not None:

            if steps > len(self.X):  # random experiences
                all_indices = np.random.choice(len(self.X), size=steps)
            else:  # closest experiences
                if tuple(center) not in self.cached_all_indices:

                    self.cached_all_indices[tuple(center)] = self.oracle.kneighbors(
                        [center], n_neighbors=steps, return_distance=False)[0]

                all_indices = self.cached_all_indices[tuple(center)]

            indices = all_indices[step]

        else:
            indices = np.random.choice(
                len(self.X), size=1, replace=k < len(self.X))
        self.contexts[ :] = self.X[indices]

        self.action_values = self.get(self.contexts[None,:])[0]
        self.optimal_value = np.max(self.action_values)

        return self.contexts

    def cache_contexts(self, t, cache_id):
        if self.cached_contexts is None or len(self.cached_contexts) != t:
            self.cached_contexts = np.zeros((t,  self.dims))
            indices = np.random.choice(
                len(self.X), size=t, replace=t < len(self.X))
            self.cached_contexts[:] = self.X[indices][:, :]

            self.cached_values = self.get(self.cached_contexts[:])
            assert np.shape(self.cached_values) == (
                t, self.k), (np.shape(self.cached_values), "vs", (t, self.k))
            self.cached_rewards = self.sample(self.cached_values)

            assert np.shape(self.cached_rewards) == (t, self.k)
            self.cache_id = cache_id

        return self.cached_contexts

    def pull(self, action, cache_index=None):
        if cache_index is not None:
            return self.cached_rewards[cache_index, action], action == np.argmax(self.cached_values[cache_index])
        if self.bernoulli:
            return np.random.uniform() < self.action_values[action], action == np.argmax(self.action_values)
        else:
            return self.action_values[action], action == np.argmax(self.action_values)

    def sample(self, values=None, cache_index=None):
        if cache_index is not None:
            return self.cached_rewards[cache_index]

        if values is None:
            values = self.action_values
        if self.bernoulli:
            return np.random.uniform(size=np.shape(values)) < values
        else:
            return values