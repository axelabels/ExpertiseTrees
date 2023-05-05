from __future__ import print_function

from math import ceil

from scipy.stats import gmean

from sklearn.gaussian_process.kernels import RationalQuadratic, PairwiseKernel


from policy import *
from tools import *


EXPECTED_STD_REWARD = np.float64(.5)


MAX_SLICE_SIZE = 100
RANDOM_CONFIDENCE = .5


class Agent(object):

    def __init__(self, bandit, policy, name=None):
        self.bandit = bandit
        self.policy = policy

        self.value_mode = False

        self.prior_bandit = None

        self.t = 0
        self.reward_history = []
        self.context_history = []
        self.action_history = []
        self.cache_id = None

        self.oracle_confidence = False
        self.name = name
        self.confidence = np.zeros(bandit.k)
        self.mu = np.zeros(bandit.k)

    def predict_normalized(self, contexts, arm=None, batch=None):
        mu = np.random.uniform(size=len(contexts))
        sigma = np.zeros(len(contexts)) + .5

        return mu, sigma

    def value_estimates(self, contexts=None, cache_index=None, return_std=False, arm=None, batch=True):

        if cache_index is not None:
            assert self.is_prepared(
            ), "When an int is given as context the expert should be prepared in advance"
            self.mu, self.sigma = np.copy(self.cached_predictions[cache_index])
        else:
            self.mu, self.sigma = self.predict_normalized(
                contexts, arm=arm, batch=batch)
        if return_std:
            return self.mu, self.sigma
        return self.mu

    def is_prepared(self, cache_id=None):
        if cache_id is None:
            return self.cache_id == self.bandit.cache_id
        return self.cache_id == cache_id

    def set_name(self, name):
        self.name = name

    def __str__(self):
        if self.name is not None:
            return self.name
        return '{}'.format(str(self.policy))

    def probabilities(self, contexts):

        if isinstance(contexts, int):
            assert self.is_prepared(
            ), "When an int is given as context the expert should be prepared in advance"
            self._probabilities = np.copy(self.cached_probabilities[contexts])
        else:
            self._probabilities = self.policy.probabilities(self, contexts)

        return self._probabilities

    def prior_play(self, steps=0, bandit=None, spread=None, noise=None, exploratory=True):
        pass

    def reset(self):
        self.t = 0
        self.reward_history = []
        self.context_history = []
        self.action_history = []
        self.cache_id = None

    def choose(self, contexts, greedy=False):
        return self.policy.choose(self, contexts, greedy=greedy)

    def observe(self, reward, arm, contexts):
        self.t += 1
        self.context_history.append(contexts)
        self.reward_history.append(float(reward))

    def update_confidence(self, contexts=None, value_mode=False, confidence_rho=None, oracle_confidence=True):

        if isinstance(contexts, int):
            assert self.is_prepared()
            self.confidence = self.cached_confidences[contexts]
            return self.confidence

        pred_val, uncertainty = self.value_estimates(
            contexts, return_std=True, batch=False)
        assert len(np.shape(pred_val)) == len(np.shape(contexts)
                                              ), (np.shape(pred_val), np.shape(contexts))
        if oracle_confidence:
            true_mu = np.clip(self.bandit.action_values, 0, 1)
            self.confidence = (np.ones(self.bandit.k) -
                               np.abs(true_mu - pred_val))
        else:
            self.confidence = RANDOM_CONFIDENCE + \
                np.clip(1 - uncertainty, 0, 1) / 2
        if not value_mode:
            self.confidence[:] = gmean(self.confidence, axis=1)

        return self.confidence

    def cache_predictions(self, bandit, trials):

        if not self.is_prepared(bandit.cache_id) or len(self.cached_predictions) < trials:
            self.cached_predictions = np.array(
                self.predict_normalized(bandit.cached_contexts))

            assert np.shape(self.cached_predictions) == (
                2, trials, bandit.k), np.shape(self.cached_predictions)
            self.cached_predictions = np.moveaxis(
                self.cached_predictions, 1, 0)
            self.cached_predictions[:, 0] = scale(
                self.cached_predictions[:, 0])
            self.cache_id = bandit.cache_id
            self.cached_probabilities = greedy_choice(
                self.cached_predictions[:, 0], axis=1)
            recomputed = True
        elif len(self.cached_predictions) > trials:
            self.cached_predictions = self.cached_predictions[:trials]
            self.cached_probabilities = self.cached_probabilities[:trials]
            recomputed = False
        return recomputed

    def cache_confidences(self, bandit,  value_mode, confidence_rho, oracle_confidence=True):

        uncertainties = self.cached_predictions[:, 1]

        self.cached_votes = np.zeros_like(self.cached_probabilities)
        self.cached_votes[np.arange(len(self.cached_votes)), np.argmax(
            self.cached_probabilities, axis=1)] = 1

        self.total_expected_reward = np.sum(
            np.sum(self.cached_votes * bandit.cached_rewards, axis=1))
        if oracle_confidence:

            self.cached_confidences = np.sum(
                self.cached_votes * bandit.cached_rewards, axis=1)[:, np.newaxis]
            uniform_performance = np.mean((bandit.cached_rewards))
            best_performance = np.mean(np.max(bandit.cached_rewards, axis=1))
            worst_performance = np.mean(np.min(bandit.cached_rewards, axis=1))

            if not value_mode:
                self.cached_confidences[:, :] = gmean(np.clip(self.cached_confidences, 1e-6, 1), axis=1)[:,
                                                                                                         np.newaxis]

            self.cached_confidences[:] = normalize(self.cached_confidences.mean(), offset=worst_performance,
                                                   scale=best_performance - worst_performance) ** (
                np.log(RANDOM_CONFIDENCE) / np.log(
                    normalize(uniform_performance, offset=worst_performance,
                              scale=best_performance - worst_performance)))

            if confidence_rho not in (None, 0):
                self.cached_confidences[:] = np.random.beta(
                    1 + np.mean(self.cached_confidences) / confidence_rho,
                    1 + (1 - np.mean(self.cached_confidences)) / confidence_rho)

        else:
            self.cached_confidences = 1 - np.tanh(uncertainties)
            assert (0 <= self.cached_confidences).all() and (
                self.cached_confidences <= 1).all()

            self.cached_confidences = self.cached_confidences * \
                (1 - RANDOM_CONFIDENCE) + RANDOM_CONFIDENCE
            if not value_mode:
                self.cached_confidences[:, :] = gmean(
                    np.clip(self.cached_confidences, 1e-6, 1), axis=1)[:, np.newaxis]


class KernelUCB(Agent):
    MAX_SPREAD = 0.5
    EXPLORATORY = False
    KERNELS = [RationalQuadratic(alpha=.1), PairwiseKernel(metric='laplacian')]

    def __init__(self, bandit, policy, gamma=.1, beta=1, kernel=None):

        super().__init__(bandit, policy)

        self.gamma, self.beta = gamma, beta

        self.kernel = kernel or RationalQuadratic(
            length_scale=(self.bandit.dims ** .5), alpha=0.1)

        self.reset()

    @property
    def advice_type(self):
        return "value"

    @property
    def choice_only(self):
        return False

    def choose(self, contexts, greedy=False):
        return self.policy.choose(self, contexts, greedy=greedy)

    def reset(self):

        self.learning = 1
        self.reward_history = ([[] for _ in range(self.bandit.k)])
        self.context_history = ([[] for _ in range(self.bandit.k)])
        self.k_inv_history = ([None for _ in range(self.bandit.k)])
        self.cache_id = None  # define a dictionary to store kernel matrix inverse in each tround

    def observe(self, reward, arm, contexts):

        self.t += 1
        self.context_history[arm].append(contexts)
        self.reward_history[arm].append(float(reward))
        self.update_kinv(arm)

    def prior_play(self, steps=0, bandit=None, spread=None):

        assert 0 < spread < 1 + \
            2, "Hypercube sides should be strictly positive but are {}".format(
                spread)
        self.prior_bandit = bandit
        self.center = np.random.uniform(0, 1, size=bandit.dims)

        if self.EXPLORATORY:
            for _ in range(steps*self.bandit.k):
                contexts = bandit.observe_contexts(self.center, spread)
                action = self.choose(contexts, greedy=True)
                reward, _ = bandit.pull(action)
                self.observe(reward, action, contexts)
        else:  # batch training
            for _ in range(steps):
                contexts = bandit.observe_contexts(center=self.center, k=1)
                samp = bandit.sample().astype(float)
                for arm in range(self.bandit.k):
                    self.context_history[arm].append(contexts)

                    self.reward_history[arm].append(samp[arm])

            for arm in range(self.bandit.k):
                self.update_kinv(arm)

        self.learning = 0

    def update_kinv(self, arm):
        contexts = normalize(self.context_history[arm], axis=0)
        self.k_inv_history[arm] = np.linalg.inv(self.kernel(
            contexts) + np.identity(len(contexts)) * self.gamma)

    def predict_normalized(self, contexts, arm=None, slice_size=None, batch=True):

        if not batch:
            assert arm is None
            if np.shape(contexts) in ((1, self.bandit.dims), (self.bandit.dims,)):
                values = np.array([self.predict_normalized(contexts.reshape(
                    (1, -1)), arm=a, batch=True)[0] for a in range(self.bandit.k)])
            elif np.shape(contexts) == (self.bandit.k, self.bandit.dims):
                values = np.array([self.predict_normalized(
                    contexts[a], arm=a, batch=True) for a in range(self.bandit.k)])

            return values[:, 0], values[:, 1]

        if arm is None:
            values = np.array([self.predict_normalized(contexts, arm=a)
                              for a in range(self.bandit.k)])
            return values[:, 0].T, values[:, 1].T

        if len(np.shape(contexts)) == 1:
            contexts = contexts[np.newaxis, :]
        mu = np.zeros(len(contexts)) + self.bandit.expected_reward

        sigma = np.ones(len(contexts))

        if len(self.context_history[arm]) <= 1:
            return mu, sigma

        con_mu = np.mean(self.context_history[arm], axis=0)
        con_std = np.std(self.context_history[arm], axis=0)

        context_history = normalize(
            self.context_history[arm], offset=con_mu, scale=con_std)
        normalized_contexts = normalize(contexts, offset=con_mu, scale=con_std)
        reward_history = normalize(
            self.reward_history[arm], offset=self.bandit.expected_reward, scale=EXPECTED_STD_REWARD)

        # Intermediary results below can occupy a lot of space if contexts is large, compute results by slices
        slice_size = min(MAX_SLICE_SIZE, len(normalized_contexts))
        for slice_index in range(ceil(len(normalized_contexts) / slice_size)):
            lo = slice_index * slice_size
            hi = lo + slice_size
            k_x = self.kernel(normalized_contexts[lo:hi], context_history)

            k_x_Kinv = k_x.dot(self.k_inv_history[arm])
            # print(np.shape(reward_history),np.shape(k_x_Kinv),np.shape(k_x_Kinv.dot(reward_history)))
            mu[lo:hi] = rescale(k_x_Kinv.dot(reward_history),
                                self.bandit.expected_reward, EXPECTED_STD_REWARD)
            sigma[lo:hi] = np.sqrt(np.maximum(0, 1 - (k_x_Kinv*k_x).sum(-1)))

        return mu, sigma

    def __str__(self):
        if self.name is not None:
            return self.name
        return "kernel"+str(self.kernel)

    def ucb_values(self, contexts=None):

        mu, sigma = np.array([self.value_estimates(
            contexts, return_std=True, arm=a) for a in range(self.bandit.k)]).T[0]

        return mu + self.learning * sigma * np.sqrt(self.beta)


class OracleExpert(Agent):
    def compute(self, contexts=None, mn=None, std=None):
        pass

    def predict_normalized(self, contexts, slice_size=None, arm=None, batch=False):
        assert np.shape(contexts)[1:] == (self.bandit.dims,)
        mu = self.prior_bandit.get(contexts)
        if arm is not None:
            mu = mu[..., arm]
        sigma = np.zeros_like(mu)
        return mu, sigma

    def prior_play(self, steps=0, bandit=None, spread=None):
        self.prior_bandit = bandit
