import numpy as np

class Mdp:
    """
    defines a Markov Decision Process
    """
    def __init__(
        self,
        states: list,
        actions: list,
        initial_distribution: np.array, #P0
        transition_probability: np.array, #P
        reward_function: np.array, #R
        gamma: float = 0.9,
        terminal_states: list = [],
        horizon: int = 50,
    ):
        self.states = states
        self.nb_states = len(states)
        self.terminal_states = terminal_states

        self.actions = actions
        self.nb_actions = len(actions)

        self.P = transition_probability
        self.P0 = initial_distribution  # distribution used to draw the first state of the agent, ased in method reset()

        self.R = reward_function

        self.horizon = horizon  # maximum length of an episode
        self.gamma = gamma  # discount factor

        self.timestep = 0
        self.current_state = None

        assert self.check_P_is_distrib(), "the transition matrix is not a distribution over (S,A)"
        assert self.check_P0_is_distrib(), "initial state is not drawn according to a distribution"


    # TODO: write methods to check arrays are probability distributions
    def check_P_is_distrib(self):
        return np.all(self.P >= 0) and np.all(self.P <= 1) and np.allclose(np.sum(self.P, axis=-1), np.ones(self.P.shape[:-1]))
    def check_P0_is_distrib(self):
        return np.all(self.P0 >= 0) and np.all(self.P0 <= 1) and np.allclose(np.sum(self.P0, axis=-1), np.ones(self.P0.shape[:-1]))

    def reset(self):  # Initializes an episode and returns the state of the agent.
        self.current_state = np.random.choice(a=self.states, p=self.P0)
        self.timestep = 0
        return self.current_state

    def done(self):  # returns True if an episode is over
        # TODO: An episode is over if the current state is a terminal state,
        # OR if the length of the episode is greater than the horizon..
        return self.current_state in self.terminal_states or self.timestep > self.horizon

    def step(self, a):  # Given the current state, and an action, performs a transtion in the MDP,
        # TODO: Draw the next state according to the transition matrix.
        next_state = np.random.choice(np.arange(self.P.shape[-1]), p=self.P[self.current_state, a])
        # TODO: Get the reward of the transition.
        reward = self.R[self.current_state, a, next_state]

        self.timestep += 1
        self.current_state = next_state
        done = self.done()  # checks if the episode is over

        return next_state, reward, done

    def one_episode(self, policy):
        state = self.reset()
        done = False
        score = 0
        t = 0
        while not done:
            action = policy(state)
            next_state, r, done = self.step(action)
            score += r
            t += 1
            state = next_state
        return score


class MazeMdp(Mdp):
    def __init__(self, wall_ixs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wall_ixs = wall_ixs

    def one_episode(self, policy):
        state = self.reset()
        done = False
        score = 0
        t = 0
        path = [state]
        while not done:
            action = policy(state)
            next_state, r, done = self.step(action)
            score += r
            t += 1
            state = next_state
            path.append(state)
            if t > self.nb_states * 10:
                break
        return score, t + 1, path # We just update this method to also output the time needed to complete the task.


class StochasticMazeMdp(MazeMdp):
    def __init__(self, stoch:float, *args, **kwargs):
        super().__init__(stoch, *args, **kwargs)
    
    def step(self, a):
        if np.random.rand() < self.stoch:
            reward = self.R[self.current_state, a, self.current_state]
            self.timestep += 1
            done = self.done()
            return self.current_state, reward, done
        else:
            return super().step(a)
