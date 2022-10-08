from grid_world import timeOpt_grid
from utils.custom_functions import my_meshgrid
from definition import ROOT_DIR
import scipy.io
import numpy as np
from os import getcwd
from os.path import join
import math


def get_filler_coords(traj, start_pos):
    x0, y0 = start_pos
    x, y = traj[0]
#     num_points = int(np.linalg.norm(traj[0] - np.array([x0, y0]), 2) // np.linalg.norm(traj[0] - traj[1], 2))
    num_points = int(80) # changed to specifically suit DG trajs

    filler_xy = np.linspace((x0, y0), (x, y), int(num_points), endpoint=False)
    return filler_xy


def prune_and_pad_paths(path_ndarray, start_xy, end_xy):
    xf, yf = end_xy
    # _, num_rzns = path_ndarray.shape
    num_rzns, _ = path_ndarray.shape

    for n in range(num_rzns):
        # prune path
        l = len(path_ndarray[n, 0])
        idx_list = []
        for i in range(l - 100, l):
            x, y = path_ndarray[n, 0][i]
            if x < xf or y > yf:
#             if x < xf and x>xf:
                idx_list.append(i)
            elif math.isnan(x) or math.isnan(y):
                idx_list.append(i)
        path_ndarray[n, 0] = np.delete(path_ndarray[n, 0], idx_list, axis=0)

        # pad path
        filler = get_filler_coords(path_ndarray[n, 0], start_xy)
        path_ndarray[n, 0] = np.append(filler, path_ndarray[n, 0], axis=0)
    return path_ndarray


# IMP: default values of nt, dt, F, startpos, endpos are taken from DG2. 
# startpos and enpos are based on coords start_coord = (0.1950, 0.2050), end_coord = (0.4, 0.8) / (0.41, 0.8)
#                                                                                     (20, 40) / (20, 41)

def setup_grid(num_actions =16, nt = 60, dt =40e-5, F =20.202, startpos = (79, 19), endpos = (19, 40), Test_grid= False):
# def setup_grid(num_actions =16, nt = 60, dt = 40e-5, F =20.202, startpos = (35, 50), endpos = (19, 40), Test_grid= False):

    # TODO: check default arguments for startpos and endpos
    #Read data from files

    data_path = join(ROOT_DIR, 'Input_data_files/nT_' + str(nt) +'/')
    all_u_mat = np.load(data_path +'all_u_mat.npy')
    all_ui_mat = np.load(data_path +'all_ui_mat.npy')
    all_v_mat = np.load(data_path +'all_v_mat.npy' )
    all_vi_mat = np.load(data_path +'all_vi_mat.npy')
    all_Yi = np.load(data_path +'all_Yi.npy' )
    vel_field_data = [all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi]
    grid_mat = scipy.io.loadmat(join(ROOT_DIR, 'Input_data_files/param.mat'))
    path_mat = scipy.io.loadmat(join(ROOT_DIR, 'Input_data_files/headings_unitvec_3.mat'))

    # TODO: check path_mat[] and other arguments
#     paths = path_mat['pathStore']
    paths = prune_and_pad_paths(path_mat['pathStore'], (0.1950, 0.2050), (0.4, 0.81))

    XP = grid_mat['XP']
    YP = grid_mat['YP']
    # Vx_rzns = np.load(join(ROOT_DIR,'Input_data_files/Velx_5K_rlzns.npy'))
    # Vy_rzns = np.load(join(ROOT_DIR,'Input_data_files/Vely_5K_rlzns.npy'))

    nT, _, nmodes = all_Yi.shape
    useful_num_rzns = 5000
    if nt == None:
        nt = nT

    param_str = ['num_actions', 'nt', 'dt', 'F', 'startpos', 'endpos']
    params = [num_actions, nt, dt, F, startpos, endpos]

    #Set up Grid
    xs = XP[1,:]
    ys_temp = YP[:,1]
    ys = np.flip(ys_temp)
    X, Y = my_meshgrid(xs, ys)

    if Test_grid == True:
        test_gsize = 10
        xs = xs[0:test_gsize]
        ys = ys[0:test_gsize]
        print("xs: ", xs)
        print("ys", ys)
        startpos = (7,5)
        endpos = (2,5)

    g = timeOpt_grid(xs, ys, dt, nt, F, startpos, endpos, num_actions=num_actions)

    print("Grid Setup Complete !")

    # CHANGE RUNNER FILE TO GET PARAMS(9TH ARG) IF YOU CHANGE ORDER OF RETURNS HERE
    return g, xs, ys, X, Y, vel_field_data, nmodes, useful_num_rzns, paths, params, param_str