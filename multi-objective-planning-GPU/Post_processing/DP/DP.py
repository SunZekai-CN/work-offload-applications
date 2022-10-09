import time
import numpy as np
from utils.custom_functions import picklePolicy, calc_mean_and_std, read_pickled_File
import pickle
from utils.plot_functions import plot_exact_trajectory_set, plot_learned_policy, plot_exact_trajectory_set_DP
from definition import ROOT_DIR
import math

# action_state_space = []
"""
__________________________________FUNCTIONS_______________________________________________________--
"""
# initialise value functions and policy
def initialise_policy_and_V(g):
    policy = {}
    V = {}
    for s in g.state_space():
        V[s] = 0
        policy[s] =(0,0)

    return policy, V

def value_iteration_update(g, V, Trans_prob):
    delV = -10
    state_flag = None
    for s in action_state_space:
        old_V = V[s]
        g.set_state(s)
        best_val = -float('inf')
        # if s[0] == g.nt - 1:
        #     print("- - - at t nt-1: ,", s, a[1], s_new)
        # print("*--- ",s)
        # print("** ",Trans_prob[s],'\n')
        for a in g.actions:
            val = 0
            # print("****   ",Trans_prob[s][a])
            for s_new in Trans_prob[s][a]:
                try:
                    prob, r = Trans_prob[s][a][s_new]
                    val += prob * (r + V[s_new])
                except:
                    print("Exception")
                    print(s, np.round(a[1], 3), s_new)
                # prob, _ = Trans_prob[s][a][s_new]
                # val += prob * (g.R(s, s_new) + V[s_new])
            if val > best_val:
                best_val = val
        V[s] = best_val
        #         print((delV, np.abs(V[s]-old_V)))
        if delV <= np.abs(V[s]-old_V):
            state_flag = s
            delV = np.abs(V[s]-old_V)

        # delV = max(delV, np.abs(V[s] - old_V))
    return V, delV, state_flag


def compute_Policy(g, policy, Trans_prob, V):
    for s in action_state_space:
        g.set_state(s)
        best_val = -float('inf')
        new_a = None
        if s == (0,7,5):
            print("**** check at st ", s, ': ')
        for a in g.actions:
            val = 0
            if s == (0,7,5):
                print("\n**check at ac ", a)
                print("*snew*")
            for s_new in Trans_prob[s][a]:
                try:
                    prob, r = Trans_prob[s][a][s_new]
                    val += prob * (r + V[s_new])
                    if s == (0,7,5):
                        print(s_new)
                        if math.isnan(prob) or math.isnan(r):
                            print("nan", prob, r)
                except:
                    print("Exception in compute_policy()")
                    print(s, np.round(a[1], 3), s_new)
                # prob, _ = Trans_prob[s][a][s_new]
                # val += prob * (g.R(s, s_new) + V[s_new])

            if val > best_val:
                best_val = val
                new_a = a
            if s == (0,7,5):
                print("val data for ac: ", val, best_val, new_a)
        
        policy[s] = new_a

    return policy

def pickle_V(V):
    with open('ValueFunc.p', 'wb') as fp:
        pickle.dump(V, fp, protocol=pickle.HIGHEST_PROTOCOL)

def write_list_to_file(list, file):
    outputfile = open(file, 'w+')
    print(list, file=outputfile)
    outputfile.close()



"""
__________________________________MAIN_______________________________________________________--
"""

def run_DP(setup_grid_params, prob_file, output_file, output_path, threshold = 1e-3, eg_rzn =1):
    #TODO: use appropriate plot funtions that utilise vel_field_data instead of vxrzns and vyrzns
    #Set up grid
    g, xs, ys, X, Y, vel_field_data, nmodes, num_rzns, path_mat, params, param_str = setup_grid_params
    print("g.nt: ", g.nt)
    global action_state_space
    action_state_space = g.ac_state_space()
    # print(action_state_space)

    #Read transition probability
    prob_full_filename = ROOT_DIR + '/DP/Trans_matxs_3D/' + prob_file + '/' + prob_file
    Trans_prob = read_pickled_File(prob_full_filename)

    #Initialise Policy and V
    policy, V = initialise_policy_and_V(g)
    countb = 0

    print("Parameters:\n",
            "xs: ", xs, '\n',
            "ys: ", ys, '\n',
            "nmodes: ", nmodes, '\n',
            "num_rzns: ", num_rzns, '\n'
        )
    for i in range(len(params)):
        print(param_str[i], ": ", params[i])


    start = time.time()
    #Iterate VI updates
    while True:
        countb += 1
        # print("iter: ", countb)
        loop_time = time.time()

        V, del_V_max, flagged_state = value_iteration_update(g, V, Trans_prob)

        if countb % 1 == 0:
            print("--- iter: ", countb, '\n',
                  "del_V_max: ",del_V_max,'\n',
                  "flagged_state: ", flagged_state,'\n',
                  "cumul_time: ", (time.time()-start)/60, ' mins\n',
                  "iter_time: ",  (time.time()-loop_time)/60, ' mins'
                )

        if del_V_max< threshold:
            print("Converged after ", countb, " iterations")
            break

        # if countb % 50 == 0:
        #     print("compute policy learned upitil iter ", countb )
        #     policy = compute_Policy(g, policy, Trans_prob, V)
        #     print("pickle policy")
        #     picklePolicy(policy, output_path + '/partial_policy_till_iter' + str(countb))

        print()

    # Compute policy
    print("launch compute policy after convergence")
    policy = compute_Policy(g, policy, Trans_prob, V)
    end = time.time()
    DP_compute_time = end - start
    # Save policy to file
    print("pickle policy")
    picklePolicy(policy, output_path + '/policy')

    # TODO: make corrections to plot_exact_trajcetory()
    # print("plot exct trajectory")
    # trajectory, G = plot_exact_trajectory(g, policy, X, Y, Vx_rzns[eg_rzn,:,:], Vy_rzns[eg_rzn,:,:], output_path, fname='Sample_Traj_with_policy_in rzn_'+ str(eg_rzn), lastfig = True)
   
    print("plot exct trajectory set")
    test_id_list = [i for i in range(4500,5000)]

    t_list_all, t_list_reached, G_list, bad_count_tuple= plot_exact_trajectory_set_DP(g, policy, X, Y, vel_field_data, test_id_list, output_path, fname='Test_Traj_set' + output_file)
    badcount = bad_count_tuple[0]
    print("t_list_all ", t_list_all)

    # Plot Policy
    print("plot policy")
    plot_learned_policy(g, DP_params = [policy, output_path], vel_field_data=vel_field_data)

    print("write list to file")
    write_list_to_file(t_list_all, output_path+'/t_list_all')
    write_list_to_file(G_list, output_path +'/G_list')

    print("calc mean and std")
    mean_tlist,_std_tlist, _, _ = calc_mean_and_std(t_list_all)
    mean_glist, _, _, _ = calc_mean_and_std(G_list)

    return V[g.start_state], mean_tlist, np.std(t_list_all), badcount, DP_compute_time, mean_glist