import numpy as np
from tqdm import tqdm

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


def QLearning(mdp, behaviour_pol, q_theory, alpha=0.1, nb_iter=int(1e6)):
    # Qhat = np.random.rand(mdp.nb_states, mdp.nb_actions)
    Qhat = np.zeros((mdp.nb_states, mdp.nb_actions))
    list_errors = []
    counts = {"terminated": 0, "truncated": 0, "horizon": mdp.horizon}
    s = mdp.reset()
    
    print("Start Q learning with", str(behaviour_pol).split()[2].replace('.get_action', ''))
    for i in tqdm(range(nb_iter)):
        a = behaviour_pol(s, Qhat)
        s_next, r, done = mdp.step(a)
        if done:
            if s_next in mdp.terminal_states:
                assert s_next == mdp.terminal_states[-1], f"Wrong terrminal state {s_next}"
                Qhat[s, a] = r
                counts["terminated"] += 1
            else:
                assert mdp.timestep == mdp.horizon + 1, f"Not in terminal states and timestep != horizon, {mdp.timestep} != {mdp.horizon}"
                counts["truncated"] += 1
            s = mdp.reset()
        else:
            delta = r + mdp.gamma*np.max(Qhat[s_next]) - Qhat[s, a]
            Qhat[s, a] = Qhat[s, a] + alpha*delta
            s = s_next
        
        error = np.linalg.norm((Qhat - q_theory)[~mdp.wall_ixs])
        list_errors.append(error)
        if error < 1e-6:
            print(f"Converged after {i} iterations")
            break
        

    print(f"Reached {counts['terminated']} terminal states during learning")
    return Qhat, np.array(list_errors), counts
