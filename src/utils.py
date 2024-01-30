import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from src.policies import PolicyFromQ, EpsGreedyPolicy, SoftmaxPolicy
from src.learning import QLearning, VI_Q
from src.maze_generator import get_maze_mdp

def get_one_episode(mdp, q, verbose=False):
    # if q.sum() == 0:
    #     raise ValueError("The Q matrix is null.")
    pol_q = PolicyFromQ(mdp, q).get_action
    score, t, path = mdp.one_episode(pol_q)
    if verbose:
        print(f"One episode: time = {t}")
    return score, t, path


def viz_path(path, grid, ax=None):
    if ax is None:
        ax = plt.gca()
    _, ncols = grid.shape
    ax.imshow(grid, cmap='gray')

    for i in range(len(path)-1):
        x1, y1 = path[i]//ncols, path[i]%ncols
        x2, y2 = path[i+1]//ncols, path[i+1]%ncols
        ax.plot([y1, y2], [x1, x2], c='r', zorder=i)


def plot_maze(grid, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.imshow(grid, cmap='gray')
    ax.set_title("Maze")
    ax.set_xticks([])  # remove x-axis ticks
    ax.set_yticks([])  # remove y-axis ticks


def plot_q(q, grid, ax=None):
    qc = q.copy()
    if ax is None:
        ax = plt.gca()
        
    mask = qc.sum(axis=-1) > 0
    qc[~mask] = np.nan
    qc[mask] = np.log(qc[mask])
    img = ax.imshow(np.max(qc, axis=-1).reshape(grid.shape), cmap='Reds')
    ax.imshow(np.where(grid==1., np.nan, 1.), cmap='gray')
    ax.set_xticks([])  # remove x-axis ticks
    ax.set_yticks([])  # remove y-axis ticks

    # Create a new axes for the colorbar that is the same height as ax2
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    plt.colorbar(img, cax=cax)


def plot_maze_path_q(grid, q_star, path, axes=None):
    if axes is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    else:
        ax1, ax2 = axes
    
    plot_maze(grid, ax1)
    viz_path(path, grid, ax1)
    ax1.set_title("Maze and path")
    
    plot_q(q_star, grid, ax2)
    ax2.set_title(r"$\max_a Q^*(s)$")


def plot_errors(errors, step_plot=100, ax=None):
    if ax is None:
        ax = plt.gca()
    for name, errors in errors.items():
        n = len(errors)
        ax.plot(np.arange(0, n, step_plot), errors[::step_plot], label=name)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel(r"$||Q_t - Q^*||$")
    ax.set_yscale("log")
    ax.legend()


def plot_convergence(N=50, alpha=[.1, .3, .5, .7], tau=2, epsilon=.7, size_maze=15, proportion_maze=.3, Horizon=700):
    cmap = sns.color_palette()

    errors_tot_greedy = np.zeros((N, len(alpha), int(1e6)))
    errors_tot_softmax = np.zeros((N, len(alpha), int(1e6)))
    for i in range(N):
        maze_mdp, _, q_theory_maze = get_maze_mdp(nrows=size_maze, ncols=size_maze, p=proportion_maze, Horizon=Horizon)
        for j, a in enumerate(alpha):
            pol_behaviour = EpsGreedyPolicy(maze_mdp, epsilon=0.7).get_action
            q, errors = QLearning(maze_mdp, pol_behaviour, alpha=a, nb_iter=int(1e6), save=False, Q_theory=q_theory_maze)
            errors_tot_greedy[i, j] = np.array(errors)

            pol_behaviour = SoftmaxPolicy(maze_mdp, tau=2).get_action
            q, errors = QLearning(maze_mdp, pol_behaviour, alpha=a, nb_iter=int(1e6), save=False, Q_theory=q_theory_maze)
            errors_tot_softmax[i, j] = np.array(errors)

    fig = plt.figure(figsize=(8,4)) 
    for j, a in enumerate(alpha):
        plt.plot(range(len(errors)), np.mean(errors_tot_greedy[:,j], axis=0), label=f'epsilon={epsilon}, alpha={a}', c=cmap(j))
        plt.fill_between(range(len(errors)), np.mean(errors_tot_greedy[:,j], axis=0)+np.std(errors_tot_greedy[:,j], axis=0), np.mean(errors_tot_greedy[:,j], axis=0)-np.std(errors_tot_greedy[:,j], axis=0), alpha=.2, color=cmap(j))
    plt.suptitle("Effect of alpha value on convergence: $\epsilon$-greedy policy")
    plt.ylabel(r"$\|Q_t - Q^*\|_2$")
    plt.legend()
    plt.show()


    fig = plt.figure(figsize=(8,4)) 
    for j, (a,c) in enumerate(zip(alpha,['blue', 'orange', 'green', 'red'])):
        plt.plot(range(len(errors)), np.mean(errors_tot_softmax[:,j], axis=0), label=f'tau={tau}, alpha={a}', c=cmap(j))
        plt.fill_between(range(len(errors)), np.mean(errors_tot_softmax[:,j], axis=0)+np.std(errors_tot_softmax[:,j], axis=0), np.mean(errors_tot_softmax[:,j], axis=0)-np.std(errors_tot_softmax[:,j], axis=0), alpha=.2, color=cmap(j))
    plt.suptitle("Effect of alpha value on convergence: softmax policy")
    plt.ylabel(r"$\|Q_t - Q^*\|_2$")
    plt.legend()
    plt.show()


def plot_effect_epsilon(N=50, epsilon = [.3, .5, .7, .9], size_maze=15, proportion_maze=.4, Horizon=700, alpha=.5):
    policy_q_eval_maze = np.zeros((len(epsilon), N, 2))
    for i, e in enumerate(epsilon):
        for episodes in range(N):
            maze_mdp, _, q_theory = get_maze_mdp(nrows=size_maze, ncols=size_maze, p=proportion_maze, Horizon=Horizon)

            pol_behaviour = EpsGreedyPolicy(maze_mdp, epsilon=e).get_action
            q_maze, errors = QLearning(maze_mdp, pol_behaviour, alpha=alpha, nb_iter=int(1e6))

            maze_mdp.reset()
            pol_q_maze = PolicyFromQ(maze_mdp, q_maze).get_action

            scores_episode, t_episode, path = maze_mdp.one_episode(pol_q_maze)
            policy_q_eval_maze[i, episodes, 0] = t_episode
            policy_q_eval_maze[i, episodes, 1] = scores_episode

    fig, axs = plt.subplots(1,2, figsize=(14,7))
    axs[0].plot(epsilon, np.mean(policy_q_eval_maze[:, :, 1], axis=-1), '--o', label='Average success rate')
    axs[0].hlines(1, epsilon[0], epsilon[-1], linestyles='--', color='orange', label='max')
    axs[0].set_xlabel('Epsilon')
    axs[1].set_xlabel('Epsilon')
    axs[0].set_ylabel('Average success rate')
    axs[1].set_ylabel('Average number of steps')
    axs[1].plot(epsilon, np.mean(policy_q_eval_maze[:, :, 0], axis=-1), '--o', label='Average number of steps')
    axs[1].bar(epsilon, np.std(policy_q_eval_maze[:, :, 0], axis=-1), width=.01, label='1 std', color='red')
    axs[0].legend()
    axs[1].legend()
    fig.suptitle(r"$\epsilon$-greedy behavioural policy")
    plt.show()


def plot_effect_tau(N=50, tau = [.5, 1, 2, 5], size_maze=15, proportion_maze=.4, Horizon=700, alpha=.5):
    policy_q_eval_maze = np.zeros((len(tau), N, 2))
    for i, t in enumerate(tau):
        for episodes in range(N):
            maze_mdp, _, q_theory = get_maze_mdp(nrows=size_maze, ncols=size_maze, p=proportion_maze, Horizon=Horizon)

            pol_behaviour = SoftmaxPolicy(maze_mdp, tau=t).get_action
            q_maze, errors = QLearning(maze_mdp, pol_behaviour, alpha=alpha, nb_iter=int(1e6))

            maze_mdp.reset()
            pol_q_maze = PolicyFromQ(maze_mdp, q_maze).get_action

            scores_episode, t_episode, path = maze_mdp.one_episode(pol_q_maze)
            policy_q_eval_maze[i, episodes, 0] = t_episode
            policy_q_eval_maze[i, episodes, 1] = scores_episode

    fig, axs = plt.subplots(1,2, figsize=(14,7))
    axs[0].plot(tau, np.mean(policy_q_eval_maze[:, :, 1], axis=-1), '--o', label='Average success rate')
    axs[0].hlines(1, tau[0], tau[-1], linestyles='--', color='orange', label='max')
    axs[0].set_xlabel('Tau')
    axs[1].set_xlabel('Tau')
    axs[0].set_ylabel('Average success rate')
    axs[1].set_ylabel('Average number of steps')
    axs[1].plot(tau, np.mean(policy_q_eval_maze[:, :, 0], axis=-1), '--o', label='Average number of steps')
    axs[1].bar(tau, np.std(policy_q_eval_maze[:, :, 0], axis=-1), width=.01, label='1 std', color='red')
    axs[0].legend()
    axs[1].legend()
    fig.suptitle(r"Softmax behavioural policy")
    plt.show()


def plot_effect_sizes(N=50, sizes = [5,10,15,20], tau=2, epsilon=.7, proportion_maze=.4, Horizon=700, alpha=.5):
    policy_q_eval_maze_greedy = np.zeros((len(sizes), N, 2))
    policy_q_eval_maze_softmax = np.zeros((len(sizes), N, 2))
    for i, s in enumerate(sizes):
        for episodes in range(N):
            maze_mdp, _, q_theory = get_maze_mdp(nrows=s, ncols=s, p=proportion_maze, Horizon=Horizon)

            # Epsilon
            pol_behaviour = EpsGreedyPolicy(maze_mdp, epsilon=epsilon).get_action
            q_maze, list_errors, list_s = QLearning(maze_mdp, pol_behaviour, alpha=alpha, nb_iter=int(1e6))
            maze_mdp.reset()
            pol_q_maze = PolicyFromQ(maze_mdp, q_maze).get_action
            scores_episode, t_episode, path = maze_mdp.one_episode(pol_q_maze)
            policy_q_eval_maze_greedy[i, episodes, 0] = t_episode
            policy_q_eval_maze_greedy[i, episodes, 1] = scores_episode

            maze_mdp.reset()
            pol_behaviour = SoftmaxPolicy(maze_mdp, tau=tau).get_action
            q_maze, errors = QLearning(maze_mdp, pol_behaviour, alpha=alpha, nb_iter=int(1e6))
            maze_mdp.reset()
            pol_q_maze = PolicyFromQ(maze_mdp, q_maze).get_action
            scores_episode, t_episode, path = maze_mdp.one_episode(pol_q_maze)
            policy_q_eval_maze_softmax[i, episodes, 0] = t_episode
            policy_q_eval_maze_softmax[i, episodes, 1] = scores_episode

    fig, axs = plt.subplots(1,2, figsize=(14,7))
    axs[0].plot(sizes, np.mean(policy_q_eval_maze_greedy[:, :, 1], axis=-1), '--o', label='Average success rate')
    axs[0].hlines(1, sizes[0], sizes[-1], linestyles='--', color='orange', label='max')
    axs[0].set_xlabel('Size')
    axs[1].set_xlabel('Size')
    axs[0].set_ylabel('Average success rate')
    axs[1].set_ylabel('Average number of steps')
    axs[1].plot(sizes, np.mean(policy_q_eval_maze_greedy[:, :, 0], axis=-1), '--o', label='Average number of steps')
    axs[1].bar(sizes, np.std(policy_q_eval_maze_greedy[:, :, 0], axis=-1), width=.2, label='1 std', color='red')
    axs[0].legend()
    axs[1].legend()
    fig.suptitle(r'$\epsilon$-greedy behavioural policy')
    plt.show()

    fig, axs = plt.subplots(1,2, figsize=(14,7))
    axs[0].plot(sizes, np.mean(policy_q_eval_maze_softmax[:, :, 1], axis=-1), '--o', label='Average success rate')
    axs[0].hlines(1, sizes[0], sizes[-1], linestyles='--', color='orange', label='max')
    axs[0].set_xlabel('Size')
    axs[1].set_xlabel('Size')
    axs[0].set_ylabel('Average success rate')
    axs[1].set_ylabel('Average number of steps')
    axs[1].plot(sizes, np.mean(policy_q_eval_maze_softmax[:, :, 0], axis=-1), '--o', label='Average number of steps')
    axs[1].bar(sizes, np.std(policy_q_eval_maze_softmax[:, :, 0], axis=-1), width=.05, label='1 std', color='red')
    axs[0].legend()
    axs[1].legend()
    fig.suptitle(r'Softmax behavioural policy')
    plt.show()


def plot_effect_proportions(N=50, proportions = [.2, .3, .4, .5], tau=2, epsilon=.7, size_maze=.4, Horizon=700, alpha=.5):
    policy_q_eval_maze_greedy = np.zeros((len(proportions), N, 2))
    policy_q_eval_maze_softmax = np.zeros((len(proportions), N, 2))
    for i, p in enumerate(proportions):
        for episodes in range(N):
            maze_mdp, _, q_theory = get_maze_mdp(nrows=size_maze, ncols=size_maze, p=p, Horizon=Horizon)

            # Epsilon
            pol_behaviour = EpsGreedyPolicy(maze_mdp, epsilon=epsilon).get_action
            q_maze, list_errors, list_s = QLearning(maze_mdp, pol_behaviour, alpha=alpha, nb_iter=int(1e6))
            maze_mdp.reset()
            pol_q_maze = PolicyFromQ(maze_mdp, q_maze).get_action
            scores_episode, t_episode, path = maze_mdp.one_episode(pol_q_maze)
            policy_q_eval_maze_greedy[i, episodes, 0] = t_episode
            policy_q_eval_maze_greedy[i, episodes, 1] = scores_episode

            maze_mdp.reset()
            pol_behaviour = SoftmaxPolicy(maze_mdp, tau=tau).get_action
            q_maze, errors = QLearning(maze_mdp, pol_behaviour, alpha=alpha, nb_iter=int(1e6))
            maze_mdp.reset()
            pol_q_maze = PolicyFromQ(maze_mdp, q_maze).get_action
            scores_episode, t_episode, path = maze_mdp.one_episode(pol_q_maze)
            policy_q_eval_maze_softmax[i, episodes, 0] = t_episode
            policy_q_eval_maze_softmax[i, episodes, 1] = scores_episode

    fig, axs = plt.subplots(1,2, figsize=(14,7))
    axs[0].plot(proportions, np.mean(policy_q_eval_maze_greedy[:, :, 1], axis=-1), '--o', label='Average success rate')
    axs[0].hlines(1, proportions[0], proportions[-1], linestyles='--', color='orange', label='max')
    axs[0].set_xlabel('Proportion')
    axs[1].set_xlabel('Proportion')
    axs[0].set_ylabel('Average success rate')
    axs[1].set_ylabel('Average number of steps')
    axs[1].plot(proportions, np.mean(policy_q_eval_maze_greedy[:, :, 0], axis=-1), '--o', label='Average number of steps')
    axs[1].bar(proportions, np.std(policy_q_eval_maze_greedy[:, :, 0], axis=-1), width=.2, label='1 std', color='red')
    axs[0].legend()
    axs[1].legend()
    fig.suptitle(r'$\epsilon$-greedy behavioural policy')
    plt.show()

    fig, axs = plt.subplots(1,2, figsize=(14,7))
    axs[0].plot(proportions, np.mean(policy_q_eval_maze_softmax[:, :, 1], axis=-1), '--o', label='Average success rate')
    axs[0].hlines(1, proportions[0], proportions[-1], linestyles='--', color='orange', label='max')
    axs[0].set_xlabel('Proportion')
    axs[1].set_xlabel('Proportion')
    axs[0].set_ylabel('Average success rate')
    axs[1].set_ylabel('Average number of steps')
    axs[1].plot(proportions, np.mean(policy_q_eval_maze_softmax[:, :, 0], axis=-1), '--o', label='Average number of steps')
    axs[1].bar(proportions, np.std(policy_q_eval_maze_softmax[:, :, 0], axis=-1), width=.05, label='1 std', color='red')
    axs[0].legend()
    axs[1].legend()
    fig.suptitle(r'Softmax behavioural policy ($\alpha =.5$)')
    plt.show()