import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.policies import PolicyFromQ


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