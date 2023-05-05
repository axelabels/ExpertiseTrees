
import numpy as np

from tools import greedy_choice, softmax

class Policy(object):
    def __init__(self, b=1):
        self.b = b
        self.key = 'value'

    def __str__(self):
        return 'generic policy'

    def probabilities(self, agent, contexts):
        a = agent.value_estimates(contexts)
        self.pi = softmax(a*self.b)
        return self.pi
        

    def choose(self, agent, contexts, greedy=False):
        

        self.pi = self.probabilities(agent, contexts)
        
        if greedy:
            self.pi = greedy_choice(self.pi)
        np.testing.assert_allclose(np.sum(self.pi),1)
        action = np.searchsorted(np.cumsum(self.pi), np.random.rand(1))[0]

        return action
        


class RandomPolicy(Policy):

    def __init__(self):
        self.key = 'value'

    def __str__(self):
        return 'random'

    def probabilities(self, agent, contexts):
        self.pi = np.ones(agent.bandit.k)/agent.bandit.k
        return self.pi


class EpsilonGreedyPolicy(Policy):

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.key = 'value'

    def __str__(self):
        return 'eps'.format(self.epsilon)

    def probabilities(self, agent, contexts):
        v = agent.value_estimates(contexts)
        self.pi = greedy_choice(v)       
        self.pi *= (1-self.epsilon)
        self.pi += self.epsilon/agent.bandit.k
        return self.pi


class GreedyPolicy(EpsilonGreedyPolicy):

    def __init__(self):
        super().__init__(0)

    def __str__(self):
        return 'greedy'


class ProbabilityGreedyPolicy(Policy):
    def __init__(self, epsilon=0):
        self.epsilon = epsilon
        self.datas = []
        self.key = 'probability'

    def __str__(self):
        return 'PGP'.format(self.epsilon)

    def probabilities(self, agent, contexts):
        
        self.pi = greedy_choice(agent.probabilities(contexts))
        self.pi *= (1-self.epsilon)
        self.pi += self.epsilon/agent.bandit.k
        

        return self.pi


class UCBPolicy(Policy):

    def __init__(self):
        pass 
    def __str__(self):
        return 'GPUCB' 

    def probabilities(self, agent, contexts):
        self.pi = greedy_choice(agent.ucb_values(contexts))
        return self.pi

class Exp3Policy(Policy):
    def __init__(self, eps=0):
        self.eps = eps
        self.key = 'probability'

    def __str__(self):
        return 'E3P'

    def probabilities(self, agent, contexts):
        self.pi = agent.probabilities(contexts)
        self.pi = self.pi * (1 - self.eps) + self.eps / len(self.pi) 
        return self.pi

class SCBPolicy(Policy):

    def __init__(self, gamma=0):
        self.gamma = gamma
        self.key = 'probability'

    def __str__(self):
        return 'SCB'

    def probabilities(self, agent, contexts):

        values = agent.value_estimates(contexts)
        best_arm = np.argmax(values)
        self.pi = np.zeros_like(values)
        self.pi[:] = 1 / \
            (agent.bandit.k+self.gamma*(values[best_arm]-values))
        self.pi[best_arm] += (1-(np.sum(self.pi)))

        return self.pi

