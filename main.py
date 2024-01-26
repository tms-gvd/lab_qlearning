import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from argparse import ArgumentParser

from src.maze_generator import get_maze_mdp
from src.policies import EpsGreedyPolicy, SoftmaxPolicy
from src.learning import QLearning
from src.utils import get_one_episode, viz_path, plot_maze_q_star, plot_path_q

def main(args):
    
    mdp, grid, q_star = get_maze_mdp(nrows=args.nrows,
                       ncols=args.ncols,
                       p=args.p,
                       gamma=args.gamma,
                       seed=args.seed)
    
    # plot maze and q*
    plot_maze_q_star(grid, q_star)

    # Q learning
    if args.policy == "eps_greedy":
        behaviour_pol = EpsGreedyPolicy(mdp, epsilon=args.eps).get_action
    elif args.policy == "softmax":
        behaviour_pol = SoftmaxPolicy(mdp, tau=args.tau).get_action
    else:
        raise ValueError("Unknown policy.")

    q, list_errors, counts = QLearning(mdp, behaviour_pol, q_star, alpha=args.alpha, nb_iter=args.nb_iter)
    score, t, path = get_one_episode(mdp, q)
    
    plot_path_q(path, grid, q)
    
    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nrows", type=int, default=8, help="Number of rows in the maze.")
    parser.add_argument("--ncols", type=int, default=12, help="Number of columns in the maze.")
    parser.add_argument("--p", type=float, default=.5, help="Density of walls in the maze.")
    parser.add_argument("--gamma", type=float, default=.9, help="Discount factor.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--policy", type=str, default="eps_greedy", help="Policy to use.")
    parser.add_argument("--eps", type=float, default=.5, help="Epsilon for the epsilon-greedy policy.")
    parser.add_argument("--tau", type=float, default=.5, help="Tau for the softmax policy.")
    parser.add_argument("--alpha", type=float, default=.4, help="Learning rate.")
    parser.add_argument("--nb_iter", type=int, default=int(2*1e5), help="Number of iterations for Q learning.")
    args = parser.parse_args()
    main(args)
    plt.show()