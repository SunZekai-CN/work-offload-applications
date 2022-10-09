from QL.Build_Q_from_Trajs import Learn_policy_from_data
from utils.plot_functions import plot_max_delQs, plot_exact_trajectory_set, plot_max_Qvalues, plot_learned_policy,plot_and_return_exact_trajectory_set_train_data
from QL.Q_Learning import Q_learning_Iters
from utils.custom_functions import initialise_policy, initialise_Q_N, initialise_guided_Q_N, initialise_policy_from_initQ, createFolder, calc_mean_and_std, append_summary_to_summaryFile,get_rzn_ids_for_training_and_testing, calc_mean_and_std_train_test, print_sorted_Qs_kvs, picklePolicy
import time
import math
import numpy as np
from definition import ROOT_DIR
from os.path import join
Pi = math.pi

Num_interleaves = 5

def run_QL(setup_grid_params, QL_params, QL_path, exp_num):
    
    exp =  QL_path

    Training_traj_size_list, ALPHA_list, esp0_list, QL_Iters_multiplier_list, init_Q, with_guidance, method, num_passes_list, eps_dec_method, N_inc = QL_params
    
    #Read data from files
    #setup_params (from setup_grid.py)= [num_actions, nt, dt, F, startpos, endpos]
    g, xs, ys, X, Y, vel_field_data, nmodes, useful_num_rzns, paths, setup_params, setup_param_str = setup_grid_params
    print("In TQLearn: ", len(setup_params), setup_params)
    num_actions, nt, dt, F, startpos, endpos = setup_params
    

    #print QL Parameters to file
    total_cases = len(Training_traj_size_list)*len(ALPHA_list)*len(esp0_list)*len(num_passes_list)*len(QL_Iters_multiplier_list)
    str_Params = ['with_guidance','Training_traj_size_list', 'ALPHA_list', 'esp0_list', 'QL_Iters', 'num_actions', 'init_Q', 'dt', 'F']
    Params = [with_guidance, Training_traj_size_list, ALPHA_list, esp0_list, QL_Iters_multiplier_list, num_actions, init_Q, dt, F]
    Param_filename = exp +'/Parmams.txt'
    outputfile = open(Param_filename, 'w+')
    for i in range(len(Params)):
        print(str_Params[i]+':  ',Params[i], file=outputfile)
    outputfile.close()
    

    #Create Sub-directories for different hyper parameters
    for num_passes in num_passes_list:
        for QL_Iters_x in QL_Iters_multiplier_list:
            for eps_0 in esp0_list:
                for ALPHA in ALPHA_list:
                    for dt_size in Training_traj_size_list:
                        directory = exp + '/num_passes_' + str(num_passes) + '/QL_Iter_x' + str(QL_Iters_x) + '/dt_size_' + str(dt_size) + '/ALPHA_' + str(ALPHA) + '/eps_0_' + str(eps_0)
                        createFolder(directory)


    case =0 #initilise case. Each case is an experiment with a particular combination of eps_0, ALPHA and dt_size
    output_parameters_all_cases = []    # contains output params for runQL for all the cases

    t_start_RUN_QL = time.time()
    query_state = (58, 20, 41)
    for num_passes in num_passes_list:
        for QL_Iters_x in QL_Iters_multiplier_list:
            for eps_0 in esp0_list:
                for ALPHA in ALPHA_list:
                    for dt_size in Training_traj_size_list:

                        # test_size = useful_num_rzns - dt_size   #number of trajetcories to be used for testing
                        t_start_case = time.time()
                        dir_path = exp + '/num_passes_' + str(num_passes) + '/QL_Iter_x' + str(QL_Iters_x) + '/dt_size_' + str(dt_size) + '/ALPHA_' + str(ALPHA) + '/eps_0_' + str(eps_0) +'/'
                        case+=1
                        QL_Iters =  QL_Iters_x * dt_size
                        print("*******  CASE: ", case, '/', total_cases, '*******')
                        print("num_passes= ", num_passes)
                        print("QL_Iters_x= ", QL_Iters_x)
                        print("with_guidance= ", with_guidance)
                        print('eps_0 = ', eps_0)
                        print('ALPHA =', ALPHA)
                        print('dt_size = ', dt_size)
                        print("N_inc= ", N_inc)
                        print("num_actions= ", num_actions)

                        # get respective indices fo trajectories for training and testing:
                        train_id_list, test_id_list, train_id_set, test_id_set, goodlist = get_rzn_ids_for_training_and_testing()

                        print("$$$$ check in TQ : train_id_list", train_id_list[0:20])
                        # print("test_size= ", test_size)
                        print("len_goodlist \n", len(goodlist))


                        # Reset Variables and environment
                        # (Re)initialise Q and N based on with_guidance paramter
                        # HCparams
                        if with_guidance==True:
                            Q, N = initialise_guided_Q_N(g, init_Q, init_Q/2,  1) #(g, init_Qval, guiding_Qval,  init_Nval)
                        else:
                            Q, N = initialise_Q_N(g,init_Q, 1) #(g, init_Qval, init_Nval)

                        g.set_state(g.start_state)
                        print("Q and N intialised!")

                        print("$$$$ CHECK Q[g.start_state]= ",Q[g.start_state])
                        print_sorted_Qs_kvs(g, Q, query_state)
                        #Learn Policy From Trajectory Data
                        #if trajectory data is given, learn from it. otherwise just initilise a policy and go to refinemnet step. The latter becomes model-free QL
                        if dt_size != 0:
                            # for n_intrleave in range(Num_interleaves):
                                # Q, N, policy, max_delQ_list_1 = Learn_policy_from_data(paths, g, Q, N, vel_field_data, nmodes, train_id_list, N_inc, num_actions =num_actions, ALPHA=ALPHA, method = method, num_passes = num_passes//Num_interleaves)
                                # Q, N, policy, max_delQ_list_2 = Q_learning_Iters(Q, N, g, policy, vel_field_data, nmodes, train_id_list, N_inc, alpha=ALPHA, QIters=QL_Iters//Num_interleaves,
                                                    # eps_0=eps_0, eps_dec_method = eps_dec_method)
                            Q, N, policy, max_delQ_list_1 = Learn_policy_from_data(paths, g, Q, N, vel_field_data, nmodes, train_id_list, N_inc, num_actions =num_actions, ALPHA=ALPHA, method = method, num_passes = num_passes)
                            
                            print("Learned Policy from data")

                            #Save policy
                            Policy_path = dir_path + 'Policy_01'
                            picklePolicy(policy, Policy_path)
                            print("Policy written to file")

                            # plot_max_Qvalues(Q, policy, X, Y, fpath = dir_path, fname = 'max_Qvalues', showfig = True)
                            print("Plotted max Qvals")

                            #Plot Policy
                            # Fig_policy_path = dir_path+'Fig_'+ 'Policy01'+'.png'
                            label_data = [F, ALPHA, init_Q, QL_Iters]
                            QL_params_plot = policy, Q, init_Q, label_data, dir_path, 'pol_plot_1'
                            plot_learned_policy(g, QL_params = QL_params_plot)
                            # plot_all_policies(g, Q, policy, init_Q, label_data, full_file_path= Fig_policy_path )
                            
                            #plot max_delQ
                            plot_max_delQs(max_delQ_list_1, filename=dir_path + 'delQplot1')
                            print("plotted learned policy and max_delQs")

                            print_sorted_Qs_kvs(g, Q, query_state)



                        else:
                            if with_guidance == True:
                                policy = initialise_policy_from_initQ(Q)
                            else:
                                policy = initialise_policy(g)


                        # Times and Trajectories based on data and/or guidance
                        t_list1, G0_list1, bad_count1 = plot_exact_trajectory_set(g, policy, X, Y, vel_field_data, nmodes, train_id_set, test_id_set, goodlist,
                                                                            fpath = dir_path, fname = 'Trajectories_before_exp')
                        print("plotted exacte trajectory set")

                        # Policy Refinement Step: Learn from Experience
                        # Q, N, policy, max_delQ_list_2 = Q_learning_Iters(Q, N, g, policy, vel_field_data, nmodes, train_id_list, N_inc, alpha=ALPHA, QIters=QL_Iters,
                        #                             eps_0=eps_0, eps_dec_method = eps_dec_method)

                        print("Policy refined")
                        #save Updated Policy
                        Policy_path = dir_path + 'Policy_02'
                        # Fig_policy_path = dir_path + 'Fig_' + 'Policy02' + '.png'

                        picklePolicy(policy, Policy_path)
                        QL_params_plot = policy, Q, init_Q, label_data, dir_path, 'pol_plot_2'
                        plot_learned_policy(g, QL_params = QL_params_plot) 
                        print("Refined policy written to file")

                        #plots after Experince
                        # plot_max_delQs(max_delQ_list_2, filename= dir_path + 'delQplot2' )
                        t_list2, G0_list2, bad_count2 = plot_exact_trajectory_set(g, policy, X, Y, vel_field_data, nmodes, train_id_set, test_id_set, goodlist,
                                                                    fpath = dir_path, fname =  'Trajectories_after_exp')
                        t_list3, G0_list3, bad_count3 = plot_and_return_exact_trajectory_set_train_data(g, policy, X, Y, vel_field_data, nmodes, train_id_list,
                                                                    fpath = dir_path, fname =  'Train_Trajectories_after_exp')
                        t_list4, G0_list4, bad_count4 = plot_and_return_exact_trajectory_set_train_data(g, policy, X, Y, vel_field_data, nmodes, test_id_list,
                                                                    fpath = dir_path, fname =  'Test_Trajectories_after_exp')
                        print("plotted max delQs and exact traj set AFTER REFINEMENT")

                        picklePolicy(Q, dir_path + 'Q2')
                        picklePolicy(N, dir_path + 'N2')


                        print_sorted_Qs_kvs(g, Q, query_state)

                        #Results to be printed
                        # avg_time1 = np.mean(t_list1)
                        # std_time1 = np.std(t_list1)
                        # avg_G01 = np.mean(G0_list1)
                        # avg_time2 = np.mean(t_list2)
                        # std_time2 = np.std(t_list2)
                        # avg_G02 = np.mean(G0_list2)
                        avg_time1, std_time1, cnt1 , none_cnt1, none_cnt_perc1 = calc_mean_and_std_train_test(t_list1, train_id_set, test_id_set)
                        avg_G01, _, _, _ ,_= calc_mean_and_std_train_test(G0_list1, train_id_set, test_id_set)
                        
                        avg_time2, std_time2, cnt2 , none_cnt2, none_cnt_perc2 = calc_mean_and_std_train_test(t_list2, train_id_set, test_id_set)
                        avg_G02, _, _, _, _ = calc_mean_and_std_train_test(G0_list2,train_id_set, test_id_set)

                        
                        overall_bad_count1 = 'dummy_init'
                        overall_bad_count2 = 'dummy_init'
                        if QL_Iters!=0:
                            overall_bad_count1 = (bad_count1, str(bad_count1*100/dt_size)+'%')
                            overall_bad_count2 = (bad_count2, str(bad_count2*100/dt_size) + '%')

                        t_end_case = time.time()
                        case_runtime = round( (t_end_case - t_start_case) / 60, 2 ) #mins

                        #Print results to file
                        picklePolicy(train_id_list, dir_path +'train_id_list')
                        picklePolicy(test_id_list, dir_path +'test_id_list')


                        str_Results1 = ['avg_time1','std_time1', 'overall_bad_count1', 'avg_G01']
                        Results1 = [avg_time1, std_time1, overall_bad_count1, avg_G01]
                        str_Results2 = ['avg_time2','std_time2', 'overall_bad_count2', 'avg_G02']
                        Results2 = [avg_time2, std_time2, overall_bad_count2, avg_G02]

                        Result_filename = dir_path + 'Results.txt'
                        outputfile = open(Result_filename, 'w+')
                        print("Before Experince ", file=outputfile)
                        for i in range(len(Results1)):
                            print(str_Results1[i] + ':  ', Results1[i], file=outputfile)

                        print(end="\n" * 3, file=outputfile)
                        print("After Experince ", file=outputfile)
                        for i in range(len(Results2)):
                            print(str_Results2[i] + ':  ', Results2[i], file=outputfile)

                        print(end="\n" * 3, file= outputfile)
                        print("Parameters: ", file = outputfile)
                        for i in range(len(Params)):
                            print(str_Params[i] + ':  ', Params[i], file=outputfile)
                        outputfile.close()

                        #Print out times to file
                        TrajTimes_filename = dir_path + 'TrajTimes1.txt'
                        outputfile = open(TrajTimes_filename, 'w+')
                        print(t_list1, file=outputfile)
                        outputfile.close()

                        Returns_filename = dir_path + 'G0list1.txt'
                        outputfile = open(Returns_filename, 'w+')
                        print(G0_list1, file=outputfile)
                        outputfile.close()

                        TrajTimes_filename = dir_path + 'TrajTimes2.txt'
                        outputfile = open(TrajTimes_filename, 'w+')
                        print(t_list2, file=outputfile)
                        outputfile.close()

                        Returns_filename = dir_path + 'G0list2.txt'
                        outputfile = open(Returns_filename, 'w+')
                        print(G0_list2, file=outputfile)
                        outputfile.close()

                        output_paramaters_ith_case = [exp_num, method, num_actions, nt, dt, F, startpos, endpos, eps_0, ALPHA,
                                                        eps_dec_method, N_inc, dt_size, with_guidance, init_Q, num_passes, QL_Iters,
                                                        avg_time1[0], std_time1[0], avg_G01[0], none_cnt1[0], cnt1[0], none_cnt_perc1[0], #train stats
                                                        avg_time2[0], std_time2[0], avg_G02[0], none_cnt2[0], cnt2[0], none_cnt_perc2[0], #train stats
                                                        avg_time1[1], std_time1[1], avg_G01[1], none_cnt1[1], cnt1[1], none_cnt_perc1[1], #test stats
                                                        avg_time2[1], std_time2[1], avg_G02[1], none_cnt2[1], cnt2[1], none_cnt_perc2[1], #test stats                                            
                                                        overall_bad_count1, overall_bad_count2, case_runtime ]
                        # Exp No	Method	Num_actions	nt	dt	F	start_pos	end_pos	Eps_0	ALPHA	dt_size_(train_size)	V[start_pos]	Mean_Time_over_5k	Variance_Over_5K	Bad Count	DP_comput_time	Mean_Glist
                        # useless line now since append summary is done here itself
                        output_parameters_all_cases.append(output_paramaters_ith_case) 
                        print("output_paramaters_ith_case\n")
                        print(output_paramaters_ith_case)
                        append_summary_to_summaryFile( join(ROOT_DIR, 'Experiments/Exp_summary_QL.csv'),  output_paramaters_ith_case)
                        picklePolicy(output_paramaters_ith_case, join(dir_path, 'output_paramaters') )
                        RUN_QL_elpased_time = round((time.time() - t_start_RUN_QL)/60, 2)
                        #Terminal Print
                        print('Case_runtime= ', case_runtime)
                        print('RUN_QL_elpased_time= ', RUN_QL_elpased_time, ' mins', end="\n" * 3)

    t_end_RUN_QL = time.time()
    RUN_QL_runtime = round((t_end_RUN_QL - t_start_RUN_QL)/60, 2)
    print("RUN_QL_runtime: ", RUN_QL_runtime, " mins")

    return output_parameters_all_cases


            