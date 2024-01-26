import numpy as np
from src.mdp import Mdp

class Policy:
    def __init__(self, mdp:Mdp):
        self.mdp = mdp

    def get_action(self, state):
        raise NotImplementedError

class Uniform(Policy):
    def __init__(self, mdp:Mdp):
        super().__init__(mdp)

    def get_action(self, state):
        return np.random.randint(self.mdp.nb_actions)
    
class SoftmaxPolicy(Policy):
    def __init__(self, mdp, tau):
        self.mdp = mdp
        self.tau = tau

    def get_action(self, state, Q):
        softmax = np.exp(Q[state]/self.tau)/np.sum(np.exp(Q[state]/self.tau))
        return np.random.choice(Q.shape[1], p=softmax)
    
class EpsGreedyPolicy(Policy):
    def __init__(self, mdp, epsilon):
        self.mdp = mdp
        self.eps = epsilon

    def get_action(self, state, Q):
        if np.random.random() < self.eps:
            return np.random.randint(self.mdp.nb_actions)
        else:
            return np.argmax(Q[state])

class PolicyFromQ(Policy):
    def __init__(self, mdp, Q):
        super().__init__(mdp)
        self.Q = Q
        self.get_policy()

    def get_policy(self):
        self.policy = np.zeros(self.mdp.nb_states, dtype=int)
        for s in self.mdp.states:
            self.policy[s] = np.argmax(self.Q[s])

    def get_action(self, state):
        return int(self.policy[state])