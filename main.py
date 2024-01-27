import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from argparse import ArgumentParser

from src.maze_generator import get_maze_mdp, create_mdp
from src.policies import EpsGreedyPolicy, SoftmaxPolicy
from src.learning import QLearning, VI_Q
from src.utils import get_one_episode, plot_maze_path_q, plot_errors


def main(args):
    if args.path_maze is not None:
        maze = np.load(args.path_maze)
        walls_ixs = np.where(maze.ravel() == 0)[0]
        mdp, grid = create_mdp(walls_ixs, *maze.shape, gamma=0.9)
        qstar, _ = VI_Q(mdp, eps=1e-8)
    else:
        mdp, grid, qstar = get_maze_mdp(
            nrows=args.nrows,
            ncols=args.ncols,
            p=args.p,
            gamma=args.gamma,
            seed=args.seed,
        )

    _, best_t, best_path = get_one_episode(mdp, qstar)

    mdp.horizon = args.horizon

    print("Best path length: {}".format(best_t))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plot_maze_path_q(grid, qstar, best_path, axes=(ax1, ax2))
    fig.tight_layout()

    # Q learning
    if args.policy == "eps_greedy":
        behaviour_pol = EpsGreedyPolicy(mdp, epsilon=args.eps).get_action
    elif args.policy == "softmax":
        behaviour_pol = SoftmaxPolicy(mdp, tau=args.tau).get_action
    else:
        raise ValueError("Unknown policy.")

    qhat, list_errors = QLearning(
        mdp,
        behaviour_pol,
        qstar,
        best_t=best_t,
        alpha=args.alpha,
        nb_iter=args.nb_iter,
        patience=args.nb_iter // 20,
    )
    _, t, path = get_one_episode(mdp, qhat)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    plot_errors({args.policy: list_errors}, ax=ax1)
    plot_maze_path_q(grid, qhat, path, axes=(ax2, ax3))

    print("Q learning path length: {}".format(t))
    
    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path_maze",
        type=str,
        default=None,
        help="Path to the maze file.",
    )
    parser.add_argument(
        "--nrows", type=int, default=8, help="Number of rows in the maze."
    )
    parser.add_argument(
        "--ncols", type=int, default=12, help="Number of columns in the maze."
    )
    parser.add_argument(
        "--p", type=float, default=0.5, help="Density of walls in the maze."
    )
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--policy", type=str, default="eps_greedy", help="Policy to use."
    )
    parser.add_argument(
        "--horizon", type=int, default=10000, help="Horizon for the MDP."
    )
    parser.add_argument(
        "--eps", type=float, default=0.5, help="Epsilon for the epsilon-greedy policy."
    )
    parser.add_argument(
        "--tau", type=float, default=0.5, help="Tau for the softmax policy."
    )
    parser.add_argument("--alpha", type=float, default=0.4, help="Learning rate.")
    parser.add_argument(
        "--nb_iter",
        type=int,
        default=int(1e7),
        help="Number of iterations for Q learning.",
    )
    args = parser.parse_args()
    main(args)
    plt.show()
