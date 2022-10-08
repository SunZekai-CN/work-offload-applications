from utils.custom_functions import Calculate_action, max_dict, initialise_policy, picklePolicy, extract_velocity
import numpy as np
from definition import Sampling_interval
import random
from _collections import OrderedDict

SAMPLING_Interval = Sampling_interval
max_delQ_threshold = 0.001

# Helper func to compute cell state of pt in trajectory and add it to dic
def compute_cell(grid, s):
    remx = (s[0] - grid.xs[0]) % grid.dj
    remy = -(s[1] - grid.ys[-1]) % grid.di
    xind = (s[0] - grid.xs[0]) // grid.dj
    yind = -(s[1] - grid.ys[-1]) // grid.di

    if remx >= 0.5 * grid.dj and remy >= 0.5 * grid.di:
        xind+=1
        yind+=1
    elif remx >= 0.5 * grid.dj and remy < 0.5 * grid.di:
        xind+=1
    elif remx < 0.5 * grid.dj and remy >= 0.5 * grid.di:
        yind+=1

    return int(yind), int(xind)


#Checks if pts p1 and p2 lie within the same cell
def within_same_spatial_cell(grid , point1, point2):
    state1 = compute_cell(grid, point1)
    state2 = compute_cell(grid, point2)
    if(state1 == state2) :
        return 1
    else:
        return 0

def build_experience_buffer(grid, vel_field_data, nmodes, paths, sampling_interval, train_path_ids, num_actions ):
    exp_buffer_all_trajs = []
    # print("$$$ CHECK  train_path_ids: ", train_path_ids)
    
    s_next_badcount = 0 #for counting how many times we dont reach terminal state
    double_count = 0
    s_next_bad_idlist = []
    doouble_idlist =[]
    coord_traj_5k =[]
    state_traj_5k =[]
    outbound_count = 0
    for k in train_path_ids:
        exp_buffer_kth_traj = []
        # Vxt = Vx_rzns[k, :, :]  #kth rzn velocity data
        # Vyt = Vy_rzns[k, :, :]
        trajectory = paths[k, 0] #kth trajectory, array of shape (n,2)

        # append starting point to traj
        s_t = int(0)
        coord_traj = []
        state_traj = []
        for i in range(0, len(trajectory), sampling_interval):
            coord_traj.append((s_t, trajectory[i][0], trajectory[i][1]))  #add first coordinate from the trajectory to coord_traj
            s_i, s_j = compute_cell(grid, trajectory[i])         # compute indices corrsponding to first coord
            state_traj.append((s_t, s_i, s_j))  #add first index to the list continating trajectory as state indices
            s_t += 1

        """
        # make dictionary states mapping to coords. and choose middle coord to append to traj
        traj_dict = OrderedDict()
        for j in range(0, len(trajectory)):
            s_i, s_j = compute_cell(grid, trajectory[j])
            s = (s_i, s_j)
            c = (trajectory[j][0], trajectory[j][1])
            if not traj_dict.get(s):
                traj_dict[s] = [c]
            else:
                traj_dict[s].append(c)
        keys = list(traj_dict.keys())
        keys.remove(keys[0])        #remove first and last keys (states).
        keys.remove(keys[-1])          #They are appended separately

        for s in keys:
            state_traj.append(s)
            l = len(traj_dict[s])
            coord_traj.append(traj_dict[s][int(l//2)])
        """
        # add last point to the trajectories
        coord_traj.append((s_t, trajectory[-1][0], trajectory[-1][1]))
        s_i, s_j = compute_cell(grid, trajectory[-1])
        state_traj.append((s_t, s_i, s_j))

        #  reverse order now to save it from doing so while leaning through "reverse" method
        state_traj.reverse()
        coord_traj.reverse()
        
        coord_traj_5k.append(coord_traj)
        state_traj_5k.append(state_traj)
        
        #build buffer
        # print("check warning, rzn: ", k)
        # print("s1, p1, p2, Vxt, Vyt")
        for i in range(len(state_traj)-1): # till len -1 because there is i+1 inside the loop
            s1=state_traj[i+1]
            s2=state_traj[i] #IMP: perhaps only used as dummy
            if not (grid.is_terminal(s1)):
                t ,m, n = s1
                # m, n = s1
                #vx=Vxt[t,i,j]
                # TODO: the below is only valid as this is a stationary vel feild. hence, t idx not needed
                try:
                    vx, vy = extract_velocity(vel_field_data, t, m, n, k)
                except:
                    print("$$$ CHECK t,i,j: ", t, m, n)
                    outbound_count+=1

                # vx = Vxt[m, n]
                # vy = Vyt[m, n]

                p1=coord_traj[i+1]
                p2=coord_traj[i]
                grid.set_state(s1, xcoord=p1[1], ycoord=p1[2])

                """COMMENTING THIS STATEMENT BELOW"""
                # if (s1[1],s1[2])!=(s2[1],s2[2]):
                # print(s1,p1,p2, vx, vy)
                a1 = Calculate_action(s1, s2, p1, p2, vx, vy, grid, coord_traj_theta= False)
                r1 = grid.move_exact(a1, vx, vy, k)
                s_next = grid.current_state()
                # if grid.current_state() != s2:
                #     print("**** mismatch: ",s1, a1, s2, grid.current_state())
                if i == 0:
                    if grid.is_terminal(s_next):
                        s_next_badcount += 1
                        s_next_bad_idlist.append((k,s1,a1,r1,s_next,s2))
                exp_buffer_kth_traj.append([s1, a1, r1, s_next])
            else:
                double_count += 1
                doouble_idlist.append((k, s1, s2))

        # if k == 0:
        #     # print("$$$$ CHECk state_traj: ", state_traj)
        #     pass
        #     # print("$$$$ CHECk 0th buffer: ")
        #     # for sars in exp_buffer_kth_traj:
        #     #     print(sars)

        #append kth-traj-list to master list
        exp_buffer_all_trajs.append(exp_buffer_kth_traj)

    # picklePolicy(doouble_idlist, 'doouble_idlist' )
    # picklePolicy( s_next_bad_idlist, 's_next_bad_idlist')
    # picklePolicy( coord_traj_5k, 'coord_traj_5k')
    # picklePolicy( state_traj_5k,'state_traj_5k ')
    print("$$$$$$$ s_next_badcount: ", s_next_badcount)
    print("$$$$$$$ double_count: ", double_count)
    print("$$$$$ dlen(train_path_ids) ", len(train_path_ids))
    print('$$$$$ outbound_count= ', outbound_count)

    return exp_buffer_all_trajs


def Q_update(Q, N, max_delQ, sars, ALPHA, grid, N_inc):
    s1, a1, r1, s2 = sars
    if not grid.is_terminal(s1) and grid.if_within_actionable_time(s1):     # if (s1[1], s1[2]) != grid.endpos:
        N[s1][a1] += N_inc
        alpha1 = ALPHA / N[s1][a1]
        q_s1_a1 = r1
        if not grid.is_terminal(s2) and grid.if_within_actionable_time(s2): # if (s2[1], s2[2]) != grid.endpos:
            _, val = max_dict(Q[s2])
            q_s1_a1 = r1 + val
        old_qsa = Q[s1][a1]
        Q[s1][a1] += alpha1 * (q_s1_a1 - Q[s1][a1])
        delQ = np.abs(old_qsa - Q[s1][a1])
        if delQ > max_delQ:
            max_delQ = delQ
    return Q, N, max_delQ


def learn_Q_from_exp_buffer(grid, exp_buffer, Q, N, ALPHA, method='reverse_order', num_passes =1):
    """
    Learns Q values after building experience buffer. Contains 2 types of methods- 1.reverse pass through buffer  2.random pass through buffer
    :param grid:
    :param exp_buffer:
    :param Q:
    :param N:
    :param ALPHA:
    :param method:
    :param num_passes:
    :return:
    """
    # print("$$$$ CHECK: ")
    # for kth_traj_buffer in exp_buffer:
    #     print("* *  * *  * *")
    #     for i in range(3):
    #         print(kth_traj_buffer[i])

    if not (method == 'reverse_order' or method == 'iid'):
        print("No such method learning Q values from traj")
        return

    print("In Build_Q_...   learning method = ", method)
    max_delQ_list = []

    if method == 'reverse_order':
        for Pass in range(num_passes):
            print("in Build_Q_.. : pass ", Pass)
            max_delQ = 0
            for kth_traj_buffer in exp_buffer:
                for sars in kth_traj_buffer:
                    Q, N, max_delQ = Q_update(Q, N, max_delQ, sars, ALPHA, grid, N_inc)

            max_delQ_list.append(max_delQ)
            print('max_delQ= ',max_delQ)
            # print("Q[start] = ", Q[grid.start_state])
            print('Q[s]: best a, val =', max_dict(Q[grid.start_state]))
            if max_delQ < max_delQ_threshold:
                print("Qs converged")
                break

    if method == 'iid':
        flatten = lambda l: [item for sublist in l for item in sublist]
        exp_buffer = flatten(exp_buffer)
        idx_list= np.arange(len(exp_buffer))
        print(len(exp_buffer))

        for Pass in range(num_passes):
            print("in Build_Q_.. : pass ", Pass)
            random.shuffle(idx_list)
            max_delQ = 0
            for i in idx_list:
                sars = exp_buffer[i]
                Q, N, max_delQ = Q_update(Q, N, max_delQ, sars, ALPHA, grid, N_inc)

            max_delQ_list.append(max_delQ)
            print('max_delQ= ', max_delQ)
            # print("Q[start] = ", Q[grid.start_state])
            print('Q[s]: best a, val =', max_dict(Q[grid.start_state]))
            if max_delQ < max_delQ_threshold:
                print("Qs converged")
                break

    return Q, N, max_delQ_list


def learn_Q_from_trajs(paths, grid, Q, N,  vel_field_data, nmodes, train_path_ids, num_actions, ALPHA, sampling_interval, method= 'reverse_order', num_passes= 1):
    #build experience buffer
    exp_buffer = build_experience_buffer(grid, vel_field_data, nmodes, paths, sampling_interval, train_path_ids, num_actions)
    print(" Built Experience Buffer")

    # use experience buffer to learn Q values
    Q, N, max_delQ_list = learn_Q_from_exp_buffer(grid, exp_buffer, Q, N, ALPHA, method=method, num_passes = num_passes)
    print("learned Q from Trajs")
    return Q, N, max_delQ_list



def Learn_policy_from_data(paths, g, Q, N, vel_field_data, nmodes, train_path_ids, n_inc, num_actions = 36, ALPHA=0.5, method = 'reverse_order', num_passes = 1):

    global N_inc
    N_inc = n_inc
    print("$$$$$$$$$$$ CHECk in buildQ: N_inc = ", N_inc)

    sampling_interval = SAMPLING_Interval
    # Q = EstimateQ_with_parallel_trajs(paths, g, pos_const, sampling_interval, Q, N, Vx, Vy, train_path_ids)
    # Q, max_Qdel_list= EstimateQ_mids_mids2(paths, g, Q, N, Vx_rzns, Vy_rzns, num_of_paths, num_actions, ALPHA, sampling_interval )
    Q, N, max_Qdel_list= learn_Q_from_trajs(paths, g, Q, N, vel_field_data, nmodes, train_path_ids, num_actions, ALPHA, sampling_interval, method = method, num_passes= num_passes)
    #Compute policy
    policy=initialise_policy(g)
    for s in Q.keys():
        newa, _ = max_dict(Q[s])
        policy[s] = newa

    return Q, N,  policy, max_Qdel_list







"""
## original . doesnt do away with copies. paths input not pruned and padded.
# Calculate theta (outgoing angle) between last point in 1st cell and first point in next cell
def build_experience_buffer(grid, Vx_rzns, Vy_rzns, paths, sampling_interval, num_of_paths, num_actions ):
    exp_buffer_all_trajs = []
    for k in range(num_of_paths):
        exp_buffer_kth_traj = []
        Vxt = Vx_rzns[k, :, :]
        Vyt = Vy_rzns[k, :, :]
        trajectory = paths[0, k]
        state_traj = []
        coord_traj = []

        #build sub sampled trajectory and reverse it
        for j in range(0, len(trajectory) - 1, sampling_interval):  # the len '-1' is to avoid reading NaN at the end of path data
            s_i, s_j = compute_cell(grid, trajectory[j])

            # state_traj.append((s_t, s_i, s_j))
            # coord_traj.append((grid.ts[s_t],trajectory[j][0], trajectory[j][1]))
            state_traj.append((s_i, s_j))
            coord_traj.append((trajectory[j][0], trajectory[j][1]))
        state_traj.reverse()
        coord_traj.reverse()

        # Append first state to the sub sampled trajectory
        m, n = grid.start_state
        x0 = grid.xs[n]
        y0 = grid.ys[grid.ni - 1 - m]
        state_traj.append(grid.start_state)
        # coord_traj.append((grid.ts[p],x0,y0))
        coord_traj.append((x0, y0))

        #build buffer
        for i in range(len(state_traj)-1):
            s1=state_traj[i+1]
            s2=state_traj[i]
            # t ,m,n=s1
            m, n = s1
            p1=coord_traj[i+1]
            p2=coord_traj[i]
       
            # if (s1[1],s1[2])!=(s2[1],s2[2]):
            #vx=Vxt[t,i,j]
            a1 = Calculate_action(s1,p1,p2, Vxt, Vyt, num_actions)
            r1 = grid.move_exact(a1, Vxt[m, n], Vyt[m, n])
            exp_buffer_kth_traj.append([s1, a1, r1, s2])

        #append kth-traj-list to master list
        exp_buffer_all_trajs.append(exp_buffer_kth_traj)

    return exp_buffer_all_trajs
"""






# def EstimateQ_mids_mids2(paths, grid, Q, N,  Vx_rzns, Vy_rzns, num_of_paths, num_actions, ALPHA, sampling_interval):
#     # considers transition from middle of state to middle of state
#     # chooses correct actions by taking into consideration velocity field
#     # generates velocity field realization here

#     max_delQ_list=[]
#     #pick trajectory from paths and store in reverse order
#     for k in range(num_of_paths):
#         if k%500 == 0:
#             print("traj_",k)

#         max_delQ = 0
#         # setup corresponding realisation of velocity field
#         """may have to build the realisation here!!!!"""
#         # Vxt = Vx_rzns[k,:,:,:]
#         # Vyt = Vy_rzns[k,:,:,:]

#         """Jugaad"""
#         Vxt = Vx_rzns[k,:,:]
#         Vyt = Vy_rzns[k,:,:]


#         # for all trajectories in the list of paths
#         trajectory = paths[0,k]
#         state_traj = []
#         coord_traj = []

#         test_trajx = []
#         test_trajy = []

#         #*********ASSUMING THAT 5DT IN TRAJ DATA IS 1 SECOND********
#         # s_t = 1
#         s_i = None
#         s_j = None


#         for j in range(0, len(trajectory) - 1, sampling_interval):  # the len '-1' is to avoid reading NaN at the end of path data
#             s_i, s_j = compute_cell(grid, trajectory[j])

#             # state_traj.append((s_t, s_i, s_j))
#             # coord_traj.append((grid.ts[s_t],trajectory[j][0], trajectory[j][1]))
#             state_traj.append((s_i, s_j))
#             coord_traj.append((trajectory[j][0], trajectory[j][1]))

#             # test_trajx.append(trajectory[j][0])
#             # test_trajy.append(trajectory[j][1])
#             # s_t+=1

#         # if the last sampled point is not endpoint of trajectory, include it in the state/coord_traj
#         # s_i_end, s_j_end = compute_cell(grid, trajectory[-2])
#         # if (s_i, s_j) != (s_i_end, s_j_end):
#         #     state_traj.append((s_t, s_i, s_j))
#         #     coord_traj.append((grid.ts[s_t], trajectory[-2][0], trajectory[-2][1]))
#         #     test_trajx.append(trajectory[-2][0])
#         #     test_trajy.append(trajectory[-2][1])
#         #Reverse trajectory orders
#         state_traj.reverse()
#         coord_traj.reverse()
#         test_trajx.reverse()
#         test_trajy.reverse()


#         # since traj data does not contain start point info, adding it explicitly
#         # p, m, n = grid.start_state

#         m, n = grid.start_state
#         x0 = grid.xs[n]
#         y0 = grid.ys[grid.ni - 1 - m]
#         state_traj.append(grid.start_state)
#         # coord_traj.append((grid.ts[p],x0,y0))
#         coord_traj.append((x0,y0))

#         # test_trajx.append(x0)
#         # # test_trajy.append(y0)
#         # if k%500==0:
#         #     plt.plot(test_trajx, test_trajy, '-o')
#         #Update Q values based on state and possible actions

#         for i in range(len(state_traj)-1):
#             s1=state_traj[i+1]
#             s2=state_traj[i]
#             # t ,m,n=s1
#             m, n = s1
#             p1=coord_traj[i+1]
#             p2=coord_traj[i]
#             """COMMENTING THIS STATEMENT BELOW"""
#             # if (s1[1],s1[2])!=(s2[1],s2[2]):

#             #vx=Vxt[t,i,j]
#             a1 = Calculate_action(s1,p1,p2, Vxt, Vyt, grid)
#             # print("EstQ: YO")
#             if (s1[0],s1[1])!= grid.endpos:

#                 N[s1][a1] += N_inc
#                 alpha1 = ALPHA / N[s1][a1]

#                 #update Q considering a1 was performed
#                 grid.set_state(s1,xcoord=p1[0], ycoord=p1[1])
#                 r1 = grid.move_exact(a1, Vxt[m,n], Vyt[m,n])
#                 q_s_a1 = r1
#                 next_s = grid.current_state()

#                 if (next_s[0], next_s[1]) != grid.endpos:
#                     _, val = max_dict(Q[next_s])
#                     q_s_a1 = r1 + val

#                 old_qsa = Q[s1][a1]
#                 Q[s1][a1] += alpha1*(q_s_a1 - Q[s1][a1])

#                 if np.abs(old_qsa - Q[s1][a1]) > max_delQ:
#                     max_delQ = np.abs(old_qsa - Q[s1][a1])

#         max_delQ_list.append(max_delQ)

#     return Q, max_delQ_list
