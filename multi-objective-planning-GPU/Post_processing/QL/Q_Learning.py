from grid_world import timeOpt_grid
from utils.custom_functions import *
from utils.plot_functions import  plot_learned_policy
from os.path import join
from os import getcwd
# from definition import N_inc
# ALPHA =0.1


def Run_Q_learning_episode(g, Q, N, vel_field_data, ALPHA, rzn, eps):
    """
    Runs a general Q learning episode in the kth realisation environment.
    """
    # print()
    # print("new episode")

    s1 = g.start_state
    t, i, j = s1
    g.set_state(s1)
    dummy_policy = None   #stochastic_action_eps_greedy() here, uses Q. so policy is ingnored anyway
    # a1 = stochastic_action_eps_greedy(policy, s1, g, eps, Q=Q)
    count = 0
    max_delQ = 0

    # while not g.is_terminal() and g.if_within_TD_actionable_time():
    while not g.is_terminal(s1) and not g.if_edge_state(s1) and g.if_within_actionable_time():
        """Will have to change this for general time"""
        
        t, i, j = s1
        a1 = stochastic_action_eps_greedy(dummy_policy, s1, g, eps, Q=Q)
        vx, vy = extract_velocity(vel_field_data, t, i, j, rzn)
        r = g.move_exact(a1, vx, vy)
        # r = g.move_exact(a1, Vx[i,j], Vy[i,j])
        s2 = g.current_state()
        # if g.is_terminal() or (not g.if_within_actionable_time()):

        alpha = ALPHA / N[s1][a1]
        N[s1][a1] += N_inc

        #maxQsa = 0 if next state is a terminal state/edgestate/outside actionable time
        max_q_s2_a2= 0
        if not g.is_terminal(s2) and not g.if_edge_state(s2) and g.if_within_actionable_time():
            a2, max_q_s2_a2 = max_dict(Q[s2])

        old_qsa = Q[s1][a1]
        Q[s1][a1] = Q[s1][a1] + alpha*(r + max_q_s2_a2 - Q[s1][a1])

        if np.abs(old_qsa - Q[s1][a1]) > max_delQ:
            max_delQ = np.abs(old_qsa - Q[s1][a1])

        #action for next iteration
        # if (s2[1], s2[2]) != g.endpos:

        # if (s2[0], s2[1]) != g.endpos:
        # if s2 != g.endpos and not g.if_edge_state(s2):
        # if not g.is_terminal(s2) and not g.if_edge_state(s2) and g.if_within_actionable_time():
        #     a1 = stochastic_action_eps_greedy(policy, s2, g, eps, Q=Q)
        # else:
        #     break

        s1 = s2
        # t, i, j = s1

    return Q, N, max_delQ

#eps_dec_method 1
def f_1_pt05(qiter, QIters, eps_0):
    # at qiter = 0, f = 1, eps = eps0
    # at qiter = Qiter, f = 20, eps= 0.05* eps0
    return eps_0 / ( (19.0 * qiter/QIters) + 1 ) 

#eps_dec_method 2
def f_1_pt5(qiter, QIters, eps_0):
    return eps_0 / ( (1.0 * qiter/QIters) + 1 ) 

def Q_learning_Iters(Q, N, g, policy, vel_field_data, nmodes, train_id_list, n_inc, alpha = 0.5, QIters=10000, stepsize=1000, post_train_size = 1000, eps_0=0.5, eps_dec_method = 1):

    global N_inc
    N_inc = n_inc
    
    print("$$$$$$$$$$$ CHECk in Q_Learning: N_inc = ", N_inc)
    max_delQ_list=[]
    dt_size = len(train_id_list)
    print("-------- CHECK: train_id_list: ", dt_size)
    # t=1
    for k in range(QIters):
        # alpha = 1/(k+1)
        # if k%(QIters/500)==0:
        #     t+=0.04
        # eps goes from eps0 to its half
        # t += 1/QIters
        if eps_dec_method == 1:
            eps = f_1_pt05(k, QIters, eps_0)
        else:
            eps =f_1_pt5(k, QIters, eps_0)

        if k%500==0:
            print("Qlearning Iters: iter, eps =", k, eps)

        # if QIters are large then, keep looping over rzns
        # Loop over rzns in train_id_list
        # TODO:xxDone: HCparam
        rzn = train_id_list[k%dt_size]
        # Vx = vx_rlzns[rzn,:,:]
        # Vy = vy_rlzns[rzn,:,:]
        
        Q, N, max_delQ = Run_Q_learning_episode(g, Q, N,vel_field_data, alpha, rzn, eps)
        if k%500==0:
            max_delQ_list.append(max_delQ)

    if QIters!=0:
        for s in Q.keys():
            newa, _ = max_dict(Q[s])
            # if policy[s] != newa:
            #     print("s, old policy, new policy",s, policy[s], newa)
            policy[s]=newa

    return Q, N, policy, max_delQ_list


# output_path, exp_num = create_new_dir()          #dirs Exp/1, Exp/2, ...
# DP_path = join(output_path,'DP')                 #dirs Exp/1/DP
# QL_path = join(output_path,'QL')


def test_QL(QIters=10000):
    QL_path = getcwd()
    xs = np.arange(10)
    ys = xs
    dt=1
    vStream_x = np.zeros((5000, len(ys), len(xs)))
    vStream_y = np.zeros((5000, len(ys), len(xs)))
    stream_speed =0.75
    vStream_x[0,4:7,:]=stream_speed
    F=1

    X, Y = my_meshgrid(xs, ys)
    print(X.shape, Y.shape, vStream_x.shape, vStream_y.shape)

    g = timeOpt_grid(xs, ys, dt, 100, F, (8,4), (1,4), num_actions=16 )

    action_states = g.ac_state_space()

    # initialise policy
    policy = initialise_policy(g)

    # initialise Q and N
    init_Q=0
    Q, N = initialise_Q_N(g, init_Q, 1)

    print("Teswt")
    Q, policy, max_delQ_list = Q_learning_Iters(Q, N, g, policy, vStream_x, vStream_y, QIters=QIters, eps_0 = 1)

    print("shapes of X, Y, vStream",X.shape, Y.shape, vStream_x.shape, vStream_x[0,:,:].shape)
    traje, return_value = plot_exact_trajectory(g, policy, X, Y, vStream_x[0,:,:], vStream_y[0,:,:], QL_path, fname='QLearning', lastfig=True)
    ALPHA=None
    label_data = [ F, stream_speed, ALPHA, init_Q, QIters ]
    plot_learned_policy(g, Q, policy, init_Q, label_data)


# test_QL()