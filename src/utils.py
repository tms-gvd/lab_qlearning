import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.policies import PolicyFromQ

def get_one_episode(mdp, q):
    if q.sum() == 0:
        raise ValueError("The Q matrix is null.")
    pol_q = PolicyFromQ(mdp, q).get_action
    score, t, path = mdp.one_episode(pol_q)
    print(f"One episode: score = {score}, time = {t}")
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

def plot_maze_q_star(grid, q_star):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    img1 = ax1.imshow(grid, cmap='gray')
    ax1.set_title("Maze")
    ax1.set_xticks([])  # remove x-axis ticks
    ax1.set_yticks([])  # remove y-axis ticks

    img2 = ax2.imshow(np.max(q_star, axis=-1).reshape(grid.shape), cmap='hot_r')
    ax2.set_title(r"$\max_a \ Q^*(s, a)$")
    ax2.set_xticks([])  # remove x-axis ticks
    ax2.set_yticks([])  # remove y-axis ticks

    # Create a new axes for the colorbar that is the same height as ax2
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(img2, cax=cax)
    
    fig.tight_layout()

def plot_path_q(path, grid, q_hat):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    img1 = ax1.imshow(np.max(q_hat, axis=-1).reshape(grid.shape), cmap='hot_r')
    ax1.set_title(r"$\max_a \ \hat{Q}(s, a)$")
    ax1.set_xticks([])  # remove x-axis ticks
    ax1.set_yticks([])  # remove y-axis ticks

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(img1, cax=cax)
    
    viz_path(path, grid, ax2)
    ax2.set_title("Path found")
    ax2.set_xticks([])  # remove x-axis ticks
    ax2.set_yticks([])  # remove y-axis ticks

    fig.tight_layout()

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