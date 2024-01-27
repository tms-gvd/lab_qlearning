import numpy as np
from tqdm import tqdm
from src.utils import get_one_episode

def VI(mdp, eps = 1e-2): #Value Iteration using the state value V

    V = np.zeros((mdp.nb_states)) #initial state values are set to 0
    list_error = []
    quitt = False

    while quitt==False:
        Vold = V.copy()
        for s in mdp.states: #for each state s
            if s in mdp.terminal_states:
                continue
            else:
                V[s] = np.max(np.dot(mdp.P[s, :, :], (mdp.R[s, :, :].T + mdp.gamma * Vold.reshape(-1, 1))))

        # Test if convergence has been reached
        error = np.linalg.norm(V-Vold)
        list_error.append(error)
        if error < eps :
            quitt = True

    return V, list_error


def VI_Q(mdp, eps=1e-2): #Value Iteration using the state-action value Q

    Q = np.zeros((mdp.nb_states, mdp.nb_actions)) #initial state values are set to 0
    list_error = []
    quitt = False

    while quitt==False:
        Qold = Q.copy()
        for s in mdp.states: #for each state s
            if s in mdp.terminal_states:
                    continue
            else:
                Q[s] = np.sum(mdp.P[s] * (mdp.R[s] + mdp.gamma * np.max(Qold, axis=1)), axis=1)
        # Test if convergence has been reached
        error = np.linalg.norm(Q-Qold)
        list_error.append(error)
        if error < eps :
            quitt = True
    return Q, list_error


def QLearning(mdp, behaviour_pol, q_theory, best_t, patience=1000, alpha=0.1, nb_iter=int(1e6)):
    # Qhat = np.random.rand(mdp.nb_states, mdp.nb_actions)
    Qhat = np.zeros((mdp.nb_states, mdp.nb_actions))
    list_errors = [np.inf]
    s = mdp.reset()
    step_patience = 0
    
    print("Start Q learning with", str(behaviour_pol).split()[2].replace('.get_action', ''))
    with tqdm(range(nb_iter), ncols=150) as pbar:
        for i in pbar:
            a = behaviour_pol(s, Qhat)
            s_next, r, done = mdp.step(a)
            if done:
                if s_next in mdp.terminal_states:
                    assert s_next == mdp.terminal_states[-1], f"Wrong terrminal state {s_next}"
                    Qhat[s, a] = r
                else:
                    assert mdp.timestep == mdp.horizon + 1, f"Not in terminal states and timestep != horizon, {mdp.timestep} != {mdp.horizon}"
                    delta = r + mdp.gamma*np.max(Qhat[s_next]) - Qhat[s, a]
                    Qhat[s, a] = Qhat[s, a] + alpha*delta
                s = mdp.reset()
            else:
                delta = r + mdp.gamma*np.max(Qhat[s_next]) - Qhat[s, a]
                Qhat[s, a] = Qhat[s, a] + alpha*delta
                s = s_next
            
            list_errors.append(np.linalg.norm((Qhat - q_theory)[~mdp.wall_ixs]))
            if list_errors[-1] == list_errors[-2]:
                step_patience += 1
            if step_patience > patience or list_errors[-1] < 1e-4:
                score, t, _ = get_one_episode(mdp, Qhat)
                pbar.set_postfix({"min path": t})
                if t == best_t and score == 1:
                    break
                else:
                    step_patience = 0
    
    return Qhat, np.array(list_errors[1:])
