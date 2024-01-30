import numpy as np

from src.maze_generator import get_maze_mdp
from src.mdp import StochasticMazeMdp, MazeMdp
from src.policies import EpsGreedyPolicy, SoftmaxPolicy
from src.learning import QLearning
from src.utils import get_one_episode

# ALPHA = .5
# SEED = 0
# NB_ITER = int(5*1e6)
# PATIENCE = NB_ITER//20

# sizes = [10, 20, 30]
# probs = [.2, .3, .4]
# n_repeat = 10

# n_conv = np.zeros((len(sizes), len(probs), n_repeat))
# path_lens = np.zeros((len(sizes), len(probs), n_repeat))

# for i, size in enumerate(sizes):
#     for j, prob in enumerate(probs):
#         print(f"Size: {size}, prob: {prob}")
#         for k, seed in enumerate(np.random.randint(0, 1000, n_repeat)):
#             print(f"Iteration: {k}")
            
#             mdp, grid, qstar = get_maze_mdp(size, size, prob, gamma=.95, seed=seed)
#             mdp.horizon = 10000
            
#             _, best_t, best_path = get_one_episode(mdp, qstar)
            
#             # pol = SoftmaxPolicy(mdp, tau=2.).get_action
#             pol = EpsGreedyPolicy(mdp, epsilon=.7).get_action
            
#             qhat, list_errors = QLearning(mdp, pol, qstar, best_t, patience=PATIENCE, alpha=ALPHA, nb_iter=NB_ITER)
#             _, t, path = get_one_episode(mdp, qhat)
            
#             n_conv[i, j, k] = len(list_errors)
#             path_lens[i, j, k] = t
            
#         print()

# name = "eps"
# np.save("conv_{name}.npy", n_conv)
# np.save("path_lens_{name}.npy", path_lens)

ALPHA = .5
SEED = 0
NB_ITER = int(1e6)
PATIENCE = NB_ITER//20

n_repeat = 10
betas = [.001, .01, .1]
errors = np.zeros((len(betas), NB_ITER, n_repeat))

for i, beta in enumerate(betas):
    print(f"Beta: {beta}")
    for k, seed in enumerate(np.random.randint(0, 1000, n_repeat)):
        print(f"Iteration: {k}")
        
        mdp, grid, qstar = get_maze_mdp(15, 15, .3, gamma=.95, seed=seed)
        mdp = StochasticMazeMdp(beta,
                                wall_ixs=mdp.wall_ixs,
                                states=mdp.states,
                                actions=np.arange(4), 
                                initial_distribution=mdp.P0,
                                transition_probability=mdp.P,
                                reward_function=mdp.R,
                                terminal_states=mdp.terminal_states,
                                gamma=mdp.gamma,
                                horizon=700)
        
        _, best_t, best_path = get_one_episode(mdp, qstar)
        
        pol = SoftmaxPolicy(mdp, tau=2.).get_action
        # pol = EpsGreedyPolicy(mdp, epsilon=.7).get_action
        
        qhat, list_errors = QLearning(mdp, pol, qstar, best_t, patience=PATIENCE, alpha=ALPHA, nb_iter=NB_ITER, stop_error=False, stop_patience=False)
        _, t, path = get_one_episode(mdp, qhat)
        
        errors[i, :, k] = list_errors

    np.save("errors.npy", errors)

print("Done")