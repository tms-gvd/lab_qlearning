import numpy as np
from tqdm import tqdm
from src.learning import VI, VI_Q
from src.mdp import MazeMdp

def get_maze_mdp(nrows=5, ncols=5, p=.4, gamma=.9, seed=None):
    assert (nrows + ncols) / (nrows * ncols) <= 1 - p, "The maze is too small to have a path from the initial state to the terminal state with this density of walls."

    n = nrows * ncols
    S = np.arange(n)
    np.random.seed(seed)
    seeds = np.random.randint(0, 1000, size=1000)
    ix = 0
    
    while True:
        maze = np.ones(n)
        if ix >= len(seeds):
            raise ValueError("No maze found with this density of walls.")
        np.random.seed(seeds[ix])
        ix += 1
        wall_ixs = np.random.choice(S[1:-1], size=int(n*p), replace=False)
        maze[wall_ixs] = 0
        
        grid = maze.reshape(nrows, ncols)
        
        P = np.zeros((n, 4, n))
        # 0: right, 1: left, 3: up, 2: down for the actions
        for i in S:
            
            _i, _j = i//ncols, i%ncols
            l, r, u, d = _j - 1, _j + 1, _i - 1, _i + 1
            
            if l >= 0 and maze[i - 1] == 1:
                P[i, 1, i - 1] = 1
            else:
                P[i, 1, i] = 1
            
            if r < ncols and maze[i + 1] == 1:
                P[i, 0, i + 1] = 1
            else:
                P[i, 0, i] = 1
            
            if u >= 0 and maze[i - ncols] == 1:
                P[i, 3, i - ncols] = 1
            else:
                P[i, 3, i] = 1
                
            if d < nrows and maze[i + ncols] == 1:
                P[i, 2, i + ncols] = 1
            else:
                P[i, 2, i] = 1
        
        P0 = np.zeros_like(S)
        P0[0] = 1

        R = np.zeros((len(S), 4, len(S)))
        R[-2, 0, -1],  R[-(ncols+1), 2, -1] = 1, 1
        
        terminal_states = np.array([S[-1]])
        terminal_states = np.append(wall_ixs, S[-1])

        maze_mdp = MazeMdp(wall_ixs,
                           states=S,
                           actions=np.arange(4), 
                           initial_distribution=P0,
                           transition_probability=P,
                           reward_function=R,
                           terminal_states=terminal_states,
                           gamma=gamma,
                           horizon=5*(nrows+ncols))
        
        V, _ = VI(maze_mdp, eps=.0001)
        if V[0]>0:
            break
    
    q_theory, _ = VI_Q(maze_mdp)

    return maze_mdp, grid, q_theory