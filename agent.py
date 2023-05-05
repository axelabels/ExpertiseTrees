from itertools import product
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from time import time
from collections import defaultdict
from expert import *
import sys
sys.path.insert(0, "multiflow/")
sys.path.insert(0, "multiflow/skmultiflow")
try:
    from skmultiflow.trees import HoeffdingTreeClassifier
except:
    pass
from skmultiflow.trees import HoeffdingTreeRegressor
EXPECTED_AVG_REWARD = .5


def get_distance(i, spread_type, max_distance, n_experts):

    window_width = max(0, min(1-max_distance, max_distance)-.1)
    if spread_type in ('diverse', 'heterogeneous'):
        cluster_id = i / (n_experts-1)*2
    elif spread_type == 'polarized':
        cluster_id = 0 if i < n_experts//2 else 2
    else:
        cluster_id = 1
    desired_distance = max_distance + (cluster_id-1) * window_width
    return desired_distance



class Collective(Agent):
    def __init__(self, bandit, policy, n_experts,  gamma=None,  name=None,   alpha=1, beta=1, expert_spread='homogeneous'):

        super(Collective, self).__init__(bandit, policy)
        self.expert_spread = expert_spread

        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma

        self.k = self.bandit.k
        self.n = n_experts
        self.name = name
        self.advice = np.zeros((self.n, self.bandit.k))
        self._value_estimates = np.zeros(self.k)
        self._probabilities = np.zeros(self.k)

        self.confidences = np.ones((self.n, self.bandit.k))

        self.initialize_w()

    def initialize_w(self):
        pass

    def short_name(self):
        return "Average"

    @property
    def info_str(self):
        info_str = ""
        return info_str

    def observe(self, reward, arm):

        self.t += 1

    def get_weights(self, contexts):

        self.confidences = np.ones((self.n, self.bandit.k)) if contexts.get(
            'confidence', None) is None else contexts['confidence']
        w = safe_logit(self.confidences)

        return w

    def __str__(self):
        if self.name is not None:
            return self.name
        return self.short_name()

    def set_name(self, name):
        self.name = name

    @staticmethod
    def prior_play(experts, bias_steps, expert_spread, base_bandit, average_expert_distance=0, spread=1):
        options = KernelUCB.KERNELS
        np.random.shuffle(options)
        for e in experts:
            e.kernel = (np.random.choice(options))

        for i, e in (list(enumerate(experts))):
            e.index = i

            desired_distance = get_distance(i, expert_spread, average_expert_distance, len(
                experts))

            if expert_spread == 'polarized':
                if i == 0 or i == len(experts)//2:
                    cluster_bandit = base_bandit.from_bandit(
                        desired_distance=desired_distance)

                prior_bandit = cluster_bandit.from_bandit(
                    desired_distance=0.05)
            else:
                prior_bandit = base_bandit.from_bandit(
                    desired_distance=desired_distance)

            e.prior_play(steps=bias_steps, bandit=prior_bandit, spread=spread)

    def choose(self, advice, greedy=False):
        return self.policy.choose(self, advice, greedy=greedy)

    def probabilities(self, contexts):
        self.advice = np.copy(contexts['advice'])
        if isinstance(self, Exp4) and not np.allclose(np.sum(self.advice, axis=1), 1):

            self.advice = greedy_choice(self.advice, axis=1)

        w = self.get_weights(contexts)
        self._probabilities = np.sum(w * self.advice, axis=0)

        assert len(self._probabilities) == self.bandit.k
        return self._probabilities

    def value_estimates(self, contexts):
        self.advice = np.copy(contexts['advice'])

        self._value_estimates = np.sum(
            self.get_weights(contexts) * (self.advice - self.bandit.expected_reward), axis=0)

        return self._value_estimates

    def reset(self):
        super().reset()

        self.initialize_w()

class RandomAgent(Collective):
    
    def short_name(self):
        return "Random"

    def probabilities(self, contexts):
        return np.ones(self.bandit.k)/self.bandit.k

        
    def value_estimates(self, contexts):
        return np.ones(self.bandit.k)/self.bandit.k

class Exp4(Collective):
    def __init__(self, bandit, policy, n_experts, gamma, name=None,  expert_spread='homogeneous',
                 confidence_weight=100, weights_decay=0):
        super(Exp4, self).__init__(bandit, policy,
                                   n_experts,  name=name, gamma=gamma,   expert_spread=expert_spread)

        self.e_sum = 1
        self.w = np.ones(self.n)/self.n
        self.confidence_weight = confidence_weight
        self.weights_decay = weights_decay

    def copy(self):
        return Exp4(self.bandit, self.policy, self.n, self.gamma)

    def short_name(self):
        return f"EXP4.S(decay={self.weights_decay})"

    def initialize_w(self):
        self.e_sum = 1
        self.context_history = []
        self.w = np.ones(self.n)/self.n

    def reset(self):
        super(Exp4, self).reset()

    def get_weights(self, contexts):

        w = np.copy(self.w)

        w = np.repeat(w[:, np.newaxis], self.bandit.k, axis=1)

        return w

    def observe(self, reward, arm):

        assert np.allclose(np.sum(self.advice, axis=1),
                           1), "expected probability advice"
        x_t = self.advice[:, arm] * (1-reward)
        y_t = x_t / (self.policy.pi[arm]+self.gamma)

        self.e_sum = (1-self.weights_decay)*self.e_sum + \
            np.sum(np.max(self.advice, axis=0))

        numerator = self.e_sum
        lr = np.sqrt(np.log(self.n) / numerator)

        self.w *= np.exp(-lr * y_t)
        self.w /= np.sum(self.w)

        self.w = (1-self.weights_decay)*self.w + self.weights_decay/self.n

        np.testing.assert_allclose(
            self.w.sum(), 1, err_msg=f"{self.w.sum()} should be 1 {self.w.sum()-1} ")

        self.t += 1


class SquareCB(Collective):
    def __init__(self, bandit, policy, n_experts, name=None,  expert_spread='homogeneous',
                 confidence_weight=100, weights_decay=0, lr=2, variable_share=True,):
        super(SquareCB, self).__init__(bandit, policy,
                                       n_experts,  name=name,   expert_spread=expert_spread)

        self.weights_decay = weights_decay
        self.e_sum = 1
        self.variable_share = variable_share
        self.w = np.ones(self.n)/self.n
        self.confidence_weight = confidence_weight
        self.lr = lr

    def copy(self, ):
        return SquareCB(self.bandit, self.policy, self.n)

    def short_name(self):
        return f"SquareCB(decay={self.weights_decay})"

    def initialize_w(self):
        self.e_sum = 1
        self.context_history = []
        self.w = np.ones(self.n)/self.n
        self.w_pow = np.zeros(self.n)

    def reset(self):
        super(SquareCB, self).reset()

    def get_weights(self, contexts):

        w = np.copy(self.w)
        w = np.repeat(w[:, np.newaxis], self.bandit.k, axis=1)

        return w

    def observe(self, reward, arm):

        y_t = (self.advice[:, arm] - reward)**2

        self.w_pow += -self.lr * y_t
        self.w_pow -= np.max(self.w_pow)

        self.w = softmax(self.w_pow)
        if self.variable_share:
            pool = np.sum((1-(1-self.weights_decay)**y_t)*self.w)
            self.w = (1-self.weights_decay)**y_t*self.w+1/(self.n-1) * \
                (pool-(1-(1-self.weights_decay)**y_t)*self.w)
        else:
            self.w = self.w*(1-self.weights_decay) + \
                (self.weights_decay/(self.n-1))*(1-self.w)

        self.w_pow[self.w > 0] = np.log(self.w[self.w > 0])
        self.w_pow[self.w == 0] = -np.inf
        self.w_pow -= np.max(self.w_pow)

        np.testing.assert_allclose(self.w.sum(
        ), 1, err_msg=f"{self.w.sum()} should be 1 {self.w.sum()-1} safe: {self.w_pow} {softmax(self.w_pow)}")

        self.t += 1

    def value_estimates(self, contexts):
        self.advice = np.copy(contexts['advice'])

        self._value_estimates = np.sum(
            self.get_weights(contexts) * (self.advice), axis=0)

        return self._value_estimates


class MAB(Collective):

    def __init__(self, bandit, policy, experts,  include_time=False, include_ctx=True, expert_spread='homogeneous', weights_decay=0,
                 name=None,  gamma=None):

        super().__init__(bandit, policy,
                         experts, gamma=gamma,  name=name,  expert_spread=expert_spread)

        self.include_ctx = include_ctx
        self.include_time = include_time
        self.weights_decay = weights_decay

    def short_name(self):
        return f"Meta-MAB(decay={self.weights_decay})"

    def initialize_w(self):
        self.reward_history = []
        self.context_history = []

        self.betas = np.zeros(self.n)
        self.alphas = np.zeros(self.n)
        self.chosen_expert = np.random.randint(self.n)
        self.w = np.ones(self.n)/self.n

    def get_weights(self, contexts):

        expert_values = np.random.beta(self.alphas+1,  self.betas+1)
        self.chosen_expert = randargmax(expert_values)

        w = np.zeros((self.n, self.k))
        w[self.chosen_expert, :] = 1
        return w

    def observe(self, reward, arm):

        self.alphas *= (1-self.weights_decay)
        self.betas *= (1-self.weights_decay)

        self.alphas[self.chosen_expert] += reward
        self.betas[self.chosen_expert] += 1 - reward

        self.t += 1


class MetaCMAB(Collective):
    def __init__(self, bandit, policy, experts, beta=1, name=None, expert_spread='homogeneous', adaptive=False, weights_decay=0,
                 alpha=100, ):
        """
        weights_decay: discount factor (1-gamma from the original paper)
        """
        super().__init__(bandit, policy,
                         experts, name=name, alpha=alpha, beta=beta, expert_spread=expert_spread)
        self.adaptive = adaptive
        self._model = None
        self.context_dimension = experts+1
        self.weights_decay = weights_decay
        self.gamma = (1-weights_decay)

    def copy(self):
        return MetaCMAB(self.bandit, self.policy, self.n, beta=self.beta,
                        alpha=self.alpha)

    def short_name(self):
        if self.adaptive:
            s = f"Meta-CMAB(decay={(self.weights_decay)})"

        else:
            s = f"Meta-CMAB"
        return s

    @property
    def model(self):
        if self._model is None:

            self._model = self._init_model({})

        return self._model

    def predict(self, advice):
        pr = np.sum(self.get_weights(None, full=True) *
                    (advice - self.bandit.expected_reward), axis=0)
        return pr

    def _init_model(self, model):
        if self.adaptive:
            model['A'] = np.identity(self.context_dimension) * self.alpha
            model['A_ap'] = np.identity(self.context_dimension) * self.alpha
        model['A_inv'] = np.identity(self.context_dimension)/self.alpha
        model['b'] = np.zeros((self.context_dimension, 1))
        model['theta'] = np.zeros((self.context_dimension, 1))

        return model

    def initialize_w(self):
        self._model = None

        self.context_history = []
        self.reward_history = []
        self.action_history = []

    def get_values(self, contexts, return_std=True):

        estimated_rewards = (contexts*self.model['theta'][None, :, 0]).sum(-1)

        if return_std:
            if self.adaptive:
                matrix = np.array(self.model['A_inv'][None, ]) * \
                    (self.model['A_ap'][None, ])*(self.model['A_inv'][None, ])[0]

            else:
                matrix = self.model['A_inv']

            uncertainties = np.sqrt((contexts@matrix*contexts).sum(-1))

            return estimated_rewards, uncertainties
        else:
            return estimated_rewards

    def value_estimates(self, contexts):

        centered_advice = contexts['advice'] - self.bandit.expected_reward

        self.meta_contexts = np.concatenate(
            (centered_advice, np.ones((1, self.bandit.k))), axis=0).T

        if self.beta > 0:
            mu, sigma = self.get_values(self.meta_contexts)
            summed = mu + sigma*self.beta
        else:
            mu, sigma = self.get_values(self.meta_contexts, return_std=False)
            summed = mu

        return summed

    def reset(self):
        super(MetaCMAB, self).reset()

    def observe(self, reward, arm, p=None, w=None):
        if w is None:
            w = 1 if p is None else self.policy.pi[arm]/p
        action_context = self.meta_contexts[arm][..., None]*w
        if self.adaptive:
            self.model['A'] = self.model['A']*self.gamma + action_context.dot(
                action_context.T) + (1-self.gamma)*self.alpha*np.identity(self.context_dimension)
            self.model['A_ap'] = self.model['A_ap']*self.gamma**2 + action_context.dot(
                action_context.T) + (1-self.gamma**2)*self.alpha*np.identity(self.context_dimension)

            self.model['A_inv'] = np.linalg.inv(
                self.model['A'])

            self.model['b'] = self.model['b'] * \
                self.gamma + (reward - self.bandit.expected_reward) * \
                action_context * w
        else:
            self.model['A_inv'] = SMInv(
                self.model['A_inv'], action_context, action_context, 1)

            self.model['b'] += (reward -
                                self.bandit.expected_reward) * action_context * w

        self.model['theta'] = (
            self.model['A_inv'].dot(self.model['b']))

        self.t += 1

    def get_weights(self):
        return self.model['theta']


class TreeHeuristic(MAB):
    def __init__(self, bandit, policy, experts,  name=None, expert_spread='homogeneous', heuristic=True, grace_period=1, min_samples_split=1):

        super().__init__(bandit, policy,
                         experts, name=name, expert_spread=expert_spread)
        self.heuristic = heuristic
        self.min_samples_split = min_samples_split
        self.grace_period = grace_period

    def short_name(self):
        if self.heuristic and self.grace_period==1 and self.min_samples_split==1:
            return "TreeHeuristic"
        return f"Tree(heuristic={self.heuristic},no_preprune=False,grace_period={self.grace_period},pseudo_rewards=False,min_samples_split={self.min_samples_split})"

    def initialize_w(self):
        self.reward_history = []
        self.advice_history = []
        self.action_history = []
        self.context_history = []

        self.decision_trees = None
        self.betas = np.zeros(self.n)
        self.alphas = np.zeros(self.n)
        self.chosen_expert = np.random.randint(self.n)
        self.w = np.ones(self.n)/self.n

    def get_weights(self, contexts):
        self.decision_context = contexts['context']
        self.advice = contexts['advice']
        if self.decision_trees is None:
            self.decision_trees = [HoeffdingTreeRegressor(
                grace_period=self.grace_period, leaf_prediction='mean', no_preprune=False) for _ in range(self.n)]

        expert_values = [self.decision_trees[i].predict([self.decision_context])[
            0] for i in range(self.n)]

        if self.heuristic:
            nodes = [self.decision_trees[i].get_leaf_node(
                [self.decision_context]) for i in range(self.n)]
            heuristic_expert_values = []
            for n in nodes:
                n = n[0]
                if n is None:
                    pos_count = 0
                    neg_count = 0
                else:
                    pos_count = n.stats[1]
                    neg_count = n.stats[0]-pos_count
                if self.min_samples_split == 'auto':
                    weight = self.t/(1+pos_count+neg_count)
                    heuristic_expert_values.append(np.random.beta(
                        pos_count*weight+1,  neg_count*weight+1))
                else:
                    heuristic_expert_values.append(np.random.beta(
                        pos_count*self.min_samples_split+1,  neg_count*self.min_samples_split+1))
            expert_values = heuristic_expert_values

        expert_values = np.array(expert_values)

        self.chosen_expert = randargmax(expert_values)

        w = np.zeros((self.n, self.k))
        w[self.chosen_expert, :] = 1
        return w

    def observe(self, reward, arm):

        self.decision_trees[self.chosen_expert].partial_fit(
            [self.decision_context], [float(reward)])
            

        self.t += 1


class DTNode():
    def __init__(self, bounds, parent=None, base='MetaCMAB', master=None):
        self.left = None
        self.right = None
        self.split_dim = None
        self.split_threshold = None
        self.mask = None
        self.model = None
        self.bounds = bounds
        self.master = master
        self.parent = parent
        self.base = base
        self.pruned = False
        self.reset_model()

    @property
    def subvalue(self):
        if self.left is None:
            return self.value
        else:

            return max(self.value, self.left.subvalue+self.right.subvalue)

    def get_depth(self):
        if self.parent is None:
            return 0
        else:
            return self.parent.get_depth()+1

    def reset_model(self):
        if self.model is None:
            if self.base == 'MetaCMAB':
                self.model = MetaCMAB(
                    self.master.bandit, GreedyPolicy(), self.master.n)
            
            else:
                raise self.base + " base not supported"
        self.value = 0
        self.model.reset()
        self.reward_history = 0
        self.candidates = None
        self.mask = None
        return self

    def replay(self, reward_history,
               advice_history,
               action_history,
               context_history,
               probability_history, t, verbose=False
               ):
        played = 0 if self.mask is None else len(self.mask)
        self.mask = np.logical_and(
                context_history > np.array([b[0] for b in self.bounds])[None], context_history <= np.array([b[-1] for b in self.bounds])[None],).all(axis=-1)
        self.mask[t:] = False
        self.mask[:played] = False

        for r, advice, action, context, probability, t in zip(reward_history[self.mask],
                                                              advice_history[self.mask],
                                                              action_history[self.mask],
                                                              context_history[self.mask],
                                                              probability_history[self.mask], np.arange(len(reward_history))[self.mask]):

            choice = self.model.choose(
                {'advice': advice, 'context': context})
            pi = self.model.policy.pi
            if self.base == 'EXP4':
                self.model.policy.pi = probability+0

            if False:
                r_hat = (r-self.master.bandit.expected_reward)*pi[action]/probability[action]

            else:
                r_hat = (r-.5)*pi[action]/probability[action]
            
            self.model.observe(r, action)
            self.reward_history += (r_hat)

        return self.reward_history+0

    def size(self, context_history, t):

        mask = np.logical_and(
            context_history > np.array([b[0] for b in self.bounds])[None], context_history <= np.array([b[-1] for b in self.bounds])[None],).all(axis=-1)
        mask[t:] = False
        return np.sum(mask)

    def find_leaf(self, context, depth=0, incremental=False):
        if self.left is None or (self.value >= self.subvalue and not incremental):

            return self
        else:

            if self.left.contains(context):
                return self.left.find_leaf(context, depth=depth+1, incremental=incremental)
            else:
                assert self.right.contains(context)
                return self.right.find_leaf(context, depth=depth+1, incremental=incremental)

    def contains(self, context):
        return np.logical_and(context > np.array([b[0] for b in self.bounds]), context <= np.array([b[-1] for b in self.bounds]),).all()


class ExpertiseTree(Collective):
    def __init__(self, bandit, policy, experts, bounds, name=None, kappa=1,offset=False, pre_prune=True, incremental=False, hoeffding=1, expert_spread='homogeneous', heuristic=False, grace_period=1, min_samples_split=2, no_preprune=False, pseudo_rewards=False):

        self.kappa = kappa
        self.heuristic = heuristic
        self.hoeffding = hoeffding
        self.min_samples_split = min_samples_split
        self.pseudo_rewards = pseudo_rewards
        self.incremental = incremental
        self.grace_period = grace_period
        self.pre_prune = pre_prune
        self.offset = offset
        self.bounds = bounds
        self.min_samples_leaf=2
        super().__init__(bandit, policy,
                         experts, name=name, expert_spread=expert_spread)

    def short_name(self):
        if self.incremental:
            return 'Incremental Expertise Tree'+(' (prune)' if self.pre_prune else '')+(' (offset)' if self.offset else '')
        else:
            return 'Expertise Tree'+(' (prune)' if self.pre_prune else '')+(' (offset)' if self.offset else '')
        
    def initialize_w(self):
        self.reward_history = []
        self.advice_history = []
        self.action_history = []
        self.context_history = []
        self.probability_history = []

        self.tree = DTNode(self.bounds, master=self)

    def get_weights(self, contexts):
        self.decision_context = contexts['context']
        self.advice = contexts['advice']
        assert self.tree.contains(self.decision_context)
        leaf = self.tree.find_leaf(
            self.decision_context, incremental=self.incremental)
        return leaf.model.get_weights(contexts)

    def choose(self, contexts, greedy=False):

        self.advice = contexts['advice']
        self.decision_context = contexts['context']
        self.update(self.decision_context)
        assert self.tree.contains(self.decision_context),self.decision_context
        leaf = self.tree.find_leaf(
            self.decision_context, incremental=self.incremental)
        return leaf.model.choose(contexts)

    def observe(self, reward, arm):

        self.reward_history.append(reward)
        self.advice_history.append(self.advice)
        self.action_history.append(arm)
        self.context_history.append(self.decision_context)

        assert self.tree.contains(self.decision_context)
        leaf = self.tree.find_leaf(
            self.decision_context, incremental=self.incremental)
        self.probability_history.append(leaf.model.policy.pi)
        self.t += 1

    def update(self, target_context=None):

        node = self.tree

        histories = self.reward_history, self.advice_history, self.action_history, self.context_history, self.probability_history
        histories = [np.asarray(h) for h in histories]
        assert node.contains(target_context)
        if len(self.reward_history) == 0:
            return
        queue = [(node)]
        while len(queue) > 0:

            (node), queue = queue[0], queue[1:]
            if not node.contains(target_context):
                continue
            
            if node.size(histories[-2], self.t) < self.min_samples_split:
                continue
            if node.left is not None and self.incremental:
                if node.left.contains(target_context):
                    queue.append(node.left)
                else:
                    queue.append(node.right)
                continue
            if node.candidates is None:
                candidates = get_candidate_splits(node.bounds, self.kappa)
                node.candidates = []
                for left_bounds, right_bounds, dimension, threshold in candidates:

                    left_node = DTNode(left_bounds, parent=node, master=self)
                    right_node = DTNode(right_bounds, parent=node, master=self)
                    
                    node.candidates.append(
                        (left_node, right_node, dimension, threshold))

            node.value = (node.replay(*histories, self.t))

            if self.pre_prune:
                splits = [(node.value/(.5*node.model.t if self.offset else 1), 0, None, None, node, None)]
            else:
                splits = []
            for left_node, right_node, dimension, threshold in node.candidates:
                if left_node.size(histories[-2], self.t)<self.min_samples_leaf or right_node.size(histories[-2], self.t)<self.min_samples_leaf:
                    continue
                in_left= left_node.contains(target_context) 
                in_right = right_node.contains(target_context)
                assert in_left or in_right
                assert not (in_left and in_right)

                left_node.value = (left_node.replay(*histories, self.t))
                right_node.value = (right_node.replay(*histories, self.t))

                if left_node.model.t<self.min_samples_leaf or right_node.model.t<self.min_samples_leaf:
                    continue
                
                splits.append((left_node.value/(left_node.model.t if self.offset else 1), right_node.value/(right_node.model.t if self.offset else 1),
                              dimension, threshold, left_node, right_node))
            
            if len(splits)==0:
                node.left=None 
                node.right = None 
                return
            best_split = max(splits, key=lambda v: (v[0]+v[1], v[-1] is None))

            if best_split[-1] is not None:

                leftval, rightval, dimension, threshold, left_node, right_node = best_split

                node.left = left_node
                node.right = right_node
                queue.append((left_node))
                queue.append((right_node))
            else:
                node.left = None
                node.right = None


    def get_n_leaves(self):
        q = [self.tree]
        n_leaves=0
        while len(q)!=0:
            n,q = q[0],q[1:]
            if n is None:
                n_leaves+=1
                continue 

            q.append(n.left)
            q.append(n.right)

        return n_leaves

    def get_depth(self):
        q = [self.tree]
        depth=0
        while len(q)!=0:
            n,q = q[0],q[1:]
            if n is None:continue 

            q.append(n.left)
            q.append(n.right)

            depth = max(depth,n.get_depth())
        return depth

class NearestNeighbor(Collective):
    def __init__(self, bandit, policy, experts, bounds, name=None, kernel=None, adapt_factor=None, importance_sampling=False, expert_spread='homogeneous', cluster_size=.1, mode='cluster'):

        self.cluster_size = cluster_size
        self.mode = mode
        self.adapt_factor = adapt_factor
        self.importance_sampling = importance_sampling
        self.bounds = bounds
        self.kernel = kernel
        super().__init__(bandit, policy,
                         experts, name=name, expert_spread=expert_spread)

    def short_name(self):
        return f"Nearest Neighbor {self.cluster_size*100}%"
        return f"Cluster(size={self.cluster_size},mode={self.mode},kernel={self.kernel})"

    def initialize_w(self):
        self.reward_history = []
        self.advice_history = []
        self.action_history = []
        self.context_history = []
        self.probability_history = []
        self.model_context = None

    def get_weights(self, contexts):
        self.decision_context = contexts['context']
        self.advice = contexts['advice']
        if self.model_context is None or not np.allclose(self.model_context, self.decision_context):

            cluster_mask = self.get_cluster_mask(
                self.decision_context, self.cluster_size)
            self.model = self.fit(cluster_mask)
        self.model_context = self.decision_context
        return self.model.get_weights(contexts)

    def normalize(self, contexts):

        return contexts
        return (contexts-self.bounds[:, 0])/np.ptp(self.bounds, axis=-1)

    def get_cluster_mask(self, context, cluster_size):
        if len(self.context_history) == 0:
            return []
        context = self.normalize([context])
        context_history = self.normalize(np.asarray(self.context_history))
        if self.mode == 'cluster':
            if type(cluster_size) is int:
                n_clusters = len(self.reward_history)//cluster_size
            else:
                n_clusters = int(1/cluster_size)
            n_clusters = np.clip(n_clusters, 1, len(self.reward_history))

            clust = KMeans(n_clusters=n_clusters).fit(context_history)

            current_cluster = clust.predict(context)[0]
            labels = clust.labels_
            mask = labels == current_cluster
        else:
            if type(cluster_size) is int:
                n_clusters = cluster_size
            else:
                n_clusters = int(len(self.reward_history)*cluster_size)
            n_clusters = min(max(1, n_clusters), len(self.reward_history))
            distances = cdist(context, context_history)[0]
            assert len(distances) == len(context_history)
            threshold = sorted(distances)[n_clusters-1]
            mask = distances <= threshold
        return mask

    def choose(self, contexts, greedy=False):

        self.advice = contexts['advice']
        self.decision_context = contexts['context']
        if self.model_context is None or not np.allclose(self.model_context, self.decision_context):
            start = time()
            cluster_mask = self.get_cluster_mask(
                self.decision_context, self.cluster_size)
            self.model = self.fit(cluster_mask)
        self.model_context = self.decision_context
        return self.model.choose(contexts)

    def observe(self, reward, arm, p=None):

        if self.adapt_factor is not None:
            cluster_mask = self.get_cluster_mask(
                self.decision_context, self.cluster_size*self.adapt_factor)
            new_model = self.fit(cluster_mask)

            chosen_arm = new_model.choose({'advice': self.advice})

            reward_estimate = (reward - .5)*(chosen_arm == arm)
            cur_reward = reward - .5

            if reward_estimate > cur_reward:
                self.cluster_size *= self.adapt_factor
            elif reward_estimate < cur_reward:
                self.cluster_size /= self.adapt_factor
            self.cluster_size = min(1., self.cluster_size)

        self.reward_history.append(reward)
        self.advice_history.append(self.advice)
        self.action_history.append(arm)
        self.context_history.append(self.decision_context)
        self.probability_history.append(
            self.model.policy.pi[arm] if p is None else p)

        self.t += 1

    def fit(self, mask):

        def filt(history, mask):
            return [h for h, m in zip(history, mask) if m]

        model = MetaCMAB(self.bandit, GreedyPolicy(),self.n)

        for r, advice, action, context, probability, t in zip(filt(self.reward_history, mask),
                                                              filt(
                                                                  self.advice_history, mask),
                                                              filt(
                                                                  self.action_history, mask),
                                                              filt(
                                                                  self.context_history, mask),
                                                              filt(
                                                                  self.probability_history, mask),
                                                              filt(np.arange(len(self.reward_history)), mask)):
            choice = model.choose({'advice': advice, 'context': context})

            if self.kernel is not None:
                w = rbf_kernel([context], [self.decision_context], gamma=self.cluster_size if self.kernel == 'rbf' else (
                    1/len(self.decision_context))*self.cluster_size)[0][0]

            else:
                w = None
            if self.importance_sampling:
                model.observe(r, action, p=probability, w=w)
            else:
                model.observe(r, action, w=w)

        return model


def get_candidate_splits(bounds, kappa):

    candidates = []
    for f_i in range(len(bounds)):
        if len(bounds[f_i])>2:
            bounds=list(bounds)
            thresholds=np.quantile(bounds[f_i],np.linspace(0,1,kappa+2, endpoint=True)[1:-1]) #bounds[f_i]
        else:
            lo, hi = bounds[f_i]
            thresholds = np.linspace(lo, hi, num=kappa+2, endpoint=True)[1:-1]
   
        for t in thresholds:
            
            
            if len(bounds[f_i])>2:
                left_bounds =[ np.copy(b) for b in bounds]
                right_bounds = [ np.copy(b) for b in bounds]
                assert len(left_bounds[f_i][left_bounds[f_i]<=t])>0,(t,left_bounds[f_i])
                assert len(right_bounds[f_i][right_bounds[f_i]>=t])>0 ,(t,right_bounds[f_i])
                left_bounds[f_i] = left_bounds[f_i][left_bounds[f_i]<=t]
                right_bounds[f_i] = right_bounds[f_i][right_bounds[f_i]>=left_bounds[f_i][-1]]


            else:
                left_bounds =[ np.copy(b) for b in bounds]
                right_bounds = [ np.copy(b) for b in bounds]

                left_bounds[f_i][1] = t
                right_bounds[f_i][ 0] = t
            def equal(a,b):
                same= True
                for i in range(len(a)):
                    same = same and len(a[i])==len(b[i]) and np.allclose(a[i],b[i])

                return same
            
            if equal(left_bounds,right_bounds):continue
           
            candidates.append(((left_bounds), (right_bounds), f_i, t))

    return candidates
