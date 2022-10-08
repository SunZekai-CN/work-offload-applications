import numpy as np 
from math import cos, sin, pi, floor
import matplotlib.pyplot as plt
import imageio
from scipy.interpolate import griddata
def extract_velocity(vel_field_data, t, i, j, rzn):
    # all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi = vel_field_data
    #    0          1           2           3           4
    # vx = all_u_mat[t,i,j] + np.matmul(all_ui_mat[t, :, i,j], all_Yi[t, rzn,:])
    # vy = all_v_mat[t,i,j] + np.matmul(all_vi_mat[t, :, i,j], all_Yi[t, rzn,:])
    vx = vel_field_data[0][t, i, j] + np.matmul(vel_field_data[2][t, :, i, j],vel_field_data[4][t, rzn,:])
    vy = vel_field_data[1][t, i, j] + np.matmul(vel_field_data[3][t, :, i, j], vel_field_data[4][t, rzn,:])

    return vx,vy






class DG_scalar_field:

    def __init__(self, gsize, nt, dt, dxy, A, eps, op_nrzns, n_wsamples, w_range, interpolate_degree=1):
        self.gsize = gsize
        self.n = gsize*gsize
        self.nt = nt
        self.dt = dt
        self.op_nrzns = op_nrzns
        self.n_wsamples = n_wsamples
        self.eps = eps
        self.A = A
        self.w_range = w_range
        self.dxy = dxy
        self.xs = self.get_xs_ys(dxy)
        self.ys = self.get_xs_ys(dxy)
        self.ts = [i*dt for i in range(nt)]
        self.interpolate_degree = interpolate_degree
        print("self.dxy= ", self.dxy)
        print("self.xs[-1]= ", self.xs[-1])
        print("self.dt = ", dt, nt, self.ts[-1])

    def get_xs_ys(self,dxy):
        return [(0.5 + i)*dxy for i in range(self.gsize)]

    def f(self , x, t):
        w = pi/2
        # length = len(self.xs)
        # term1 = self.eps * sin(w*t) * (x**2)
        # term2 = (1 - 2*self.eps*sin(w*t)) * x
        T = t*self.interpolate_degree/2
        l = self.xs[-1]/2
        high_val = 1
        low_val = 0
        dom_size = self.xs[-1] + self.dxy
        if ((x + (T*self.dxy)) <= l):
            return high_val
        elif (l < (x + (T*self.dxy)) < l+(0.2*dom_size)):
            # return -0.5*(x + (T*self.dxy)) + 5
            slope = (high_val-low_val)/(-0.2*dom_size)
            return high_val + (slope*(x - (l- T*self.dxy)))
        else:
            return low_val


    def phi(self, x, y, t):
        # *sin(2*pi*t/20))
        # return self.A*(1+ sin(0.25*pi*self.f(x,t))*cos(0.05*pi*(y+1)))
        return self.A*(self.f(x,t))
        
    def sample_w(self, k):
        # not used when generating only mean scalar field
        w_i, w_f = self.w_range
        r = k/(self.n_wsamples - 1)
        w = w_i + r*(w_f - w_i)
        # if k==1:
        #     print("(k,w)=",(k,w*10/pi))
        return w

    def get_ij_from_xy(self, i, j):
        x = self.xs[j]
        y = self.ys[self.gsize - 1 - i]
        return x, y

    def generate_phi(self):
        # generate mean scalar field
        phi = np.empty((self.nt, self.n),  dtype = np.float32)
        for tid in range(self.nt):
            # t = self.ts[tid]
            t= tid
            for i in range(self.gsize):
                for j in range(self.gsize):
                    col = self.gsize*i + j
                    x, y = self.get_ij_from_xy(i,j)
                    phi[tid, col] = self.phi(x,y,t)

        g= self.gsize
        phi_grid = np.empty((self.nt,g,g), dtype = np.float32)
        for tid in range(self.nt):
            phi_grid[tid,:,:]= np.resize(phi[tid,:],(g,g) )

        return phi_grid




class DG_velocity_field:

    def __init__(self, gsize, nt, dt, dxy, A, eps, wx, wy, op_nrzns, n_wsamples, w_range, interpolate_degree=1):
        self.gsize = gsize
        self.n = gsize*gsize 
        self.nt = nt
        self.dt = dt
        self.op_nrzns = op_nrzns
        self.n_wsamples = n_wsamples
        self.eps = eps
        self.A = A
        self.w_range = w_range
        self.dxy = dxy
        self.xs = self.get_xs_ys(dxy, gsize)
        self.ys = self.get_xs_ys(dxy, gsize)
        self.ts = [i*dt for i in range(nt)]
        self.interpolate_degree = interpolate_degree
        self.xgrid, self.ygrid = np.meshgrid(self.xs, np.flip(self.ys))
        self.wy = wy
        self.wx = wx

    def get_xs_ys(self, dxy, size):
        return [(0.5 + i)*dxy for i in range(size)]
        
    def f(self , x, t, w):
        term1 = self.eps * sin(w*t) * (x**2)
        term2 = (1 - 2*self.eps*sin(w*t)) * x
        return  term1 + term2 

    def fx(self, x, t, w):
        fx = (2*self.eps*sin(w*t)*(x-1)) + 1
        return fx

    def phi(self, x, y, t, w):
        return self.A*sin(pi*self.f(x,t,w))*sin(self.wy*y)

    def vx_vy(self, x, y, t, w):
        fx = self.fx(x,t,w)
        f = self.f(x,t,w)
        vx = -( 0.5 * self.A * pi * sin(self.wx*f) * cos(self.wy *y) )
        vy = self.A * pi * sin(self.wy * y) * fx * cos(self.wx*f)
        return vx, vy

    def sample_w(self, k):
        w_i, w_f = self.w_range
        r = k/(self.n_wsamples - 1)
        w = w_i + r*(w_f - w_i)
        # if k==1:
        #     print("(k,w)=",(k,w*10/pi))
        return w

    def get_ij_from_xy(self, i, j):
        x = self.xs[j]
        y = self.ys[self.gsize - 1 - i]
        return x, y

    def generate_R(self):
        # Reealisation matrix
        R_vx = np.empty((self.nt, self.n_wsamples, self.n))
        R_vy = np.empty((self.nt, self.n_wsamples, self.n))
        phi = np.empty((self.nt, self.n_wsamples, self.n))
        assert(R_vx.shape == (self.nt, self.n_wsamples, self.n))
        print("tid: ")
        for tid in range(self.nt):
            if tid%25 == 0:
                print(tid)
            t = self.ts[tid]
            for k in range(self.n_wsamples):
                w = self.sample_w(k)
                for i in range(self.gsize):
                    for j in range(self.gsize):
                        row = k
                        col = self.gsize*i + j
                        x, y = self.get_ij_from_xy(i,j)
                        vx, vy = self.vx_vy(x, y, t, w)
                        R_vx[tid, row, col] = vx
                        R_vy[tid, row, col] = vy
                        phi[tid, row, col] = self.phi(x,y,t,w)
        print()
        return R_vx, R_vy, phi

    def find_max_vels(self, R_vx, R_vy):
        R_vx_sq = np.square(R_vx)
        R_vy_sq = np.square(R_vy)
        R_vsq = R_vx_sq + R_vy_sq
        vx_sq_max = np.max(R_vx_sq)
        vy_sq_max = np.max(R_vy_sq)
        vmax_sq = np.max(R_vsq)
        max_vels = [vx_sq_max**0.5, vy_sq_max**0.5, vmax_sq**0.5]
        min_vels = [np.min(R_vx_sq)**0.5, np.min(R_vy_sq)**0.5, np.min(R_vsq)**0.5]
        mean_vels = [np.mean(R_vx_sq)**0.5, np.mean(R_vy_sq)**0.5, np.mean(R_vsq)**0.5]
        return max_vels, min_vels, mean_vels

    def rank_reduction(self, u, s, vh, n_modes):
        u1 = u[:, 0:n_modes]
        s1_mat = np.diag(s[0:n_modes])
        vh1 = vh[0:n_modes, :]
        C = np.matmul(u1, s1_mat)
        M = vh1
        assert(C.shape == (self.n_wsamples, n_modes))
        # assert(M.shape == (n_modes, self.n*2))
        return C,M

    def compute_modes_and_coeffs(self, n_modes):
        Rx, Ry, phi = self.generate_R()
        R = np.concatenate((Rx, Ry), axis = 2)
        # no. of modes wanted is not more than min matrix dim
        assert(n_modes <= np.min(R.shape))

        R_mean = np.mean(R, axis =1)
        R_mean_full = np.empty_like(R)
        for k in range(self.n_wsamples):
            R_mean_full[:,k,:] = R_mean    

        assert(R.shape == (self.nt, self.n_wsamples, self.n*2))
        assert(R_mean_full.shape == R.shape)
        assert(R_mean.shape == (self.nt, self.n*2))

        X = R - R_mean_full
        print("n_modes= ", n_modes)
        C = np.empty((self.nt, self.n_wsamples, n_modes),dtype = np.float32)
        M = np.empty((self.nt, n_modes, self.n*2),dtype = np.float32)
        sn = 5 # to print first sn singular values
        print("first ",sn, "singular valus ")
        for tid in range(self.nt):
            t = self.ts[tid]
            u, s, vh = np.linalg.svd(X[tid,:,:])
            if tid%50==0:
                print(" out of ",len(s),"at tid= ",tid,"; t= ", t, ": \n", np.round(s[0:sn],2),"\n")
            Ct, Mt = self.rank_reduction(u,s,vh, n_modes)
            C[tid,:,:] = Ct
            M[tid,:,:] = Mt 
        return C, M, R_mean_full, R_mean

    def compute_phi_modes_and_coefs(self, n_modes):
        Rx, Ry, phi = self.generate_R()
        phi_mean = np.mean(phi, axis=1) #mean along n_samples
        print("phi.shape = ", phi.shape)
        print("phi_mean.shape = ", phi_mean.shape)
        phi_mean_full = np.empty_like(phi)
        for k in range(self.n_wsamples):
            phi_mean_full[:,k,:] = phi_mean 

        del_phi = phi - phi_mean_full
        C = np.empty((self.nt, self.n_wsamples, n_modes),dtype = np.float32)
        M = np.empty((self.nt, n_modes, self.n), dtype = np.float32)
        sn = 5 # to print first sn singular values~
        for tid in range(self.nt):
            t = self.ts[tid]
            u, s, vh = np.linalg.svd(del_phi[tid,:,:])
            print(" out of ",len(s),"at tid= ",tid,"; t= ", t, ": \n", np.round(s[0:sn],2),"\n")
            Ct, Mt = self.rank_reduction(u,s,vh, n_modes)
            C[tid,:,:] = Ct
            M[tid,:,:] = Mt 
        return C, M, phi_mean_full, phi_mean


    def reshape_matrices_for_square_grid(self,C, M, phi_M, R_mean, phi_mean, n_modes):
        # means
        all_u_mat = np.empty((self.nt, self.gsize, self.gsize), dtype = np.float32)
        all_v_mat = np.empty_like(all_u_mat, dtype = np.float32)
        all_phi_mat = np.empty((self.nt, self.gsize, self.gsize))
        # modes
        all_ui_mat = np.empty((self.nt, n_modes, self.gsize, self.gsize),dtype = np.float32)
        all_vi_mat = np.empty_like(all_ui_mat, dtype = np.float32)
        all_phi_i_mat = np.empty((self.nt, n_modes, self.gsize, self.gsize),dtype = np.float32)

        # coeffs
        all_Yi_mat = C
        g = self.gsize
        n = self.n
        print("check C dtype", C.dtype)
        for tid in range(self.nt):
            all_u_mat[tid,:,:] = np.reshape(R_mean[tid,0:n],(g,g))
            all_v_mat[tid,:,:] = np.reshape(R_mean[tid,n:2*n],(g,g))
            all_phi_mat[tid,:,:] = np.reshape(phi_mean[tid,0:n],(g,g))
            for m in range(n_modes):
                all_ui_mat[tid,m,:,:] = np.reshape(M[tid,m,0:n],(g,g))
                all_vi_mat[tid,m,:,:] = np.reshape(M[tid,m,n:2*n],(g,g))
                all_phi_i_mat[tid,m,:,:] = np.reshape(phi_M[tid,m,0:n],(g,g))
        return [all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi_mat, all_phi_mat, all_phi_i_mat]

    def interpolate_data_t(self, field_data):
        d = self.interpolate_degree
        new_size = d*self.gsize 
        new_dxy = self.dxy/d
        new_xs = self.get_xs_ys(new_dxy, new_size)
        new_ys = self.get_xs_ys(new_dxy, new_size)
        # print("lenghts ",len(new_xs), len(self.xs))
        assert(len(new_xs) == d * len(self.xs))
        new_xgrid, new_ygrid = np.meshgrid(new_xs, np.flip(new_ys))
        x=self.xgrid.flatten()
        y=self.ygrid.flatten()
        # print("new_xgrid.shape: ",new_xgrid.shape)
        points = np.empty((len(x),2))
        points[:,0] =x
        points[:,1] =y
        assert(points.shape == (self.gsize*self.gsize, 2))
        new_field_data = griddata(points, field_data.flatten(), (new_xgrid, new_ygrid), method='linear')
        # print("new_field_data.shape= ",new_field_data.shape)
        return new_field_data

    def interpolate_data_to_refined_grid(self, all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_phi_mat, all_phi_i_mat, n_modes):
        d = self.interpolate_degree

        new_all_u_mat = np.empty((self.nt, d*self.gsize, d*self.gsize), dtype = np.float32)
        new_all_v_mat = np.empty_like(new_all_u_mat, dtype = np.float32)
        new_all_phi_mat = np.empty((self.nt, d*self.gsize, d*self.gsize), dtype = np.float32)
        print("new_all_phi_mat.shape= ", new_all_phi_mat.shape)
        print("new_all_phi_mat.shape=", new_all_phi_mat[1,:,:].shape)
        # modes
        new_all_ui_mat = np.empty((self.nt, n_modes, d*self.gsize, d*self.gsize),dtype = np.float32)
        new_all_vi_mat = np.empty_like(new_all_ui_mat, dtype = np.float32)
        new_all_phi_i_mat = np.empty((self.nt, n_modes, d*self.gsize, d*self.gsize),dtype = np.float32)

        for tid in range(self.nt):
            new_all_u_mat[tid,:,:] = self.interpolate_data_t(all_u_mat[tid,:,:])
            new_all_v_mat[tid,:,:] = self.interpolate_data_t(all_v_mat[tid,:,:])
            new_all_phi_mat[tid,:,:] = self.interpolate_data_t(all_phi_mat[tid,:,:])

            for m in range(n_modes):
                new_all_ui_mat[tid,m,:,:] = self.interpolate_data_t(all_ui_mat[tid,m,:,:])
                new_all_vi_mat[tid,m,:,:] = self.interpolate_data_t(all_vi_mat[tid,m,:,:])
                new_all_phi_i_mat[tid,m,:,:] = self.interpolate_data_t(all_phi_i_mat[tid,m,:,:])

        return [new_all_u_mat, new_all_v_mat, new_all_ui_mat, new_all_vi_mat, new_all_phi_mat,new_all_phi_i_mat]










def test_generate_R_plots(dg1, rzn_list, nt, gsize):
    Rx, Ry, phi = dg1.generate_R()
    print(Rx.shape)

    X,Y = np.meshgrid(dg1.xs, np.flip(dg1.ys))
    # print(X, Y)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    plt.gca().set_aspect('equal', adjustable='box')
    for r in rzn_list:
        images= []
        for t in range(nt):
            vx = np.reshape(Rx[t,r,:],(gsize,gsize))
            vy = np.reshape(Ry[t,r,:],(gsize,gsize))
            phi_tr = np.reshape(phi[t,r,:],(gsize,gsize))
            # plt.quiver(X, Y, vx, vy, color = 'c', alpha = 0.5)
            plt.streamplot(X, Y, vx, vy, color='k', linewidth=2)
            plt.contourf(X, Y, phi_tr, cmap='RdBu')
            plt.colorbar()
            fname = "test.png"
            plt.savefig(fname)
            plt.clf()
            images.append(imageio.imread(fname))
        gifname = "movie" + "_r" + str(r) + ".gif"
        imageio.mimsave(gifname, images, duration = 1)

    return


def test_rank_reduced_field(dg1,t,kth_rzn,n_modes):
    # n_modes = np.min((self.n*2, self.n_wsamples)) #criteria
    assert(t<dg1.nt)
    C, M, R_mean_full, _ = dg1.compute_modes_and_coeffs(n_modes)
    Ct, Mt = C[t,:,:], M[t,:,:]
    mean_Rt = R_mean_full[t,:,:]
    red_Xt = np.matmul(Ct, Mt)
    red_Rt = mean_Rt + red_Xt
    assert(red_Xt.shape == mean_Rt.shape)
    print("red_Rt.shape = ",red_Rt.shape)

    old_Rx, old_Ry,phi = dg1.generate_R()
    old_Rx_t = old_Rx[t,:,:]
    old_Ry_t = old_Ry[t,:,:]

    old_Rt = np.concatenate((old_Rx_t, old_Ry_t), axis = 1)
    assert(old_Rt.shape == (dg1.n_wsamples, dg1.n*2))
    assert(old_Rt.shape == red_Rt.shape)
    print("arrays are nearly equal: ", np.allclose(red_Rt, old_Rt))

    k = kth_rzn  #kth realisation
    red_Rx_k = red_Rt[k,0:dg1.n]
    red_Ry_k = red_Rt[k,dg1.n:dg1.n*2]
    print(red_Rx_k.shape, red_Ry_k.shape)
    red_vx =np.reshape(red_Rx_k, (dg1.gsize, dg1.gsize))
    red_vy =np.reshape(red_Ry_k, (dg1.gsize, dg1.gsize))
    X,Y = np.meshgrid(dg1.xs, np.flip(dg1.ys))
    phi_tr = np.reshape(phi[t,k,:], (dg1.gsize, dg1.gsize))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.streamplot(X, Y, red_vx, red_vy, color='k', linewidth=2)
    plt.contourf(X, Y, phi_tr, cmap='RdBu')
    plt.colorbar()
    plt.savefig("reduced_rank"+"t"+str(t))
    return


def plot_modes_of_rank_reduced_field(dg, C, M, t, n_modes):
    assert(n_modes <= np.min((dg.n_wsamples, dg.n*2)))
    # C, M, R_mean_full, _= dg.compute_modes_and_coeffs(n_modes)
    phi_C, phi_M =  dg.compute_phi_modes_and_coefs(n_modes)


    fig = plt.figure(figsize=(10, 10))
    _, Mt = C[t,:,:], M[t,:,:]
    _, phi_Mt = phi_C[t,:,:], phi_M[t,:,:]
    X,Y = np.meshgrid(dg.xs, np.flip(dg.ys))

    for k in range(n_modes):
        ax = fig.add_subplot(2, 2, k+1)
        Mt_k = Mt[k,:]
        phi_Mt_k = phi_Mt[k,:]
        mode_vx = np.reshape(Mt_k[0:dg.n], (dg.gsize,dg.gsize))
        mode_vy = np.reshape(Mt_k[dg.n:dg.n*2], (dg.gsize,dg.gsize))
        mode_phi = np.reshape(phi_Mt_k[0:dg.n], (dg.gsize,dg.gsize))
        ax.streamplot(X, Y, mode_vx, mode_vy, color='k', linewidth=2)
        plt.contourf(X, Y, mode_phi, cmap='RdBu')
        ax.set_ylabel(str(k))
    
    plt.savefig('modes', bbox_inches = "tight", dpi=300)
    return phi_C, phi_M



def fill_obstacles(obstacle_mask, final_gsize, nt, dyn_obstacle_data):
    # row/col up/down init/final
    # obs_upper_row, obs_left_col, obs_width have range 0 to final_gsize
    # obs_speed = [vx, vy] units= cells/timestep

    mode = dyn_obstacle_data[-1]

    if mode == 'None':
        obstacle_mask[:, :, :] = 0

    if mode == 'static':
        ru = int(0.4*final_gsize)
        rd = int(0.5*final_gsize)
        cl = int(0.5*final_gsize)
        cr = int(0.6*final_gsize)
        obstacle_mask[:, ru:rd, cl:cr] = 1

    elif mode == 'dynamic':
        obs_upper_row, obs_left_col, obs_width, obs_speed, _ = dyn_obstacle_data
        assert(0<obs_left_col<final_gsize and 0<obs_upper_row<final_gsize)
        ru_i = obs_upper_row
        # rd_i = ru_i + obs_width
        cl_i = obs_left_col
        # cr_i = cl_i + obs_width
        for t in range(nt):
            cl = floor(cl_i + (t * obs_speed[0]))
            cr = int(cl + obs_width)
            ru = floor(ru_i + (t * obs_speed[1]))
            rd = int(ru + obs_width)
            # if (cl == 0 or cl == final_gsize -1 or cr == 0 or cr == final_gsize -1 or ru == 0 or ru == final_gsize -1  or rd == 0 or rd == final_gsize -1):
            # if obstacle is in domain
            if (0 < cl < final_gsize-1) and (0 < cr < final_gsize-1) and (0 < ru < final_gsize-1) and (0 < rd < final_gsize-1):
                obstacle_mask[t, ru:rd, cl:cr] = 1
            else:
                obstacle_mask[t, :, :] = 0

    return obstacle_mask 


def test_scalar_field(all_s_mat, X, Y, nt):
    fig = plt.figure(figsize=(10, 10))
    images = []
    fname = "test.png"

    for t in range(nt):
        s_arr = all_s_mat[t,:,:]
        plt.contourf(X, Y, s_arr, cmap = "YlOrRd", alpha = 0.5, zorder = -1e5)
        plt.colorbar()
        plt.savefig(fname)
        plt.clf()
        images.append(imageio.imread(fname))
    imageio.mimsave('scalar_field.gif', images, duration = 0.5)


# init_gsize = 25
# nt = 10
# dt = 10/nt
# dxy = 2/init_gsize
# A = 0.5
# A_sc = 3
# eps = 0.1
# op_nrzns = 10
# n_wsamples = 1000
# w_range = ( pi/10, 8*pi/10 )
# wy = 0.5*pi
# wx = pi
# # interpolates 
# interpolate_degree = 8
# n_modes = 3

def build_environment_DG_with_westward_clouds(init_gsize, nt, dt, dxy, A, A_sc, eps,
                                                op_nrzns, n_wsamples, w_range, wx,  
                                                interpolate_degree, n_modes, dyn_obstacle_data = None):
    wy = 0.5*pi
    final_gsize = init_gsize*interpolate_degree
    print("gsize=", final_gsize)
    print("nt=", nt)
    print("nr=", n_wsamples)
    print("nmodes=",n_modes)
    print("dxy=",dxy)

    print("initialising objects")
    dg1 = DG_velocity_field(init_gsize, nt, dt, dxy, A, eps, wx, wy, op_nrzns, n_wsamples, w_range, interpolate_degree)
    cloud = DG_scalar_field(final_gsize, nt, dt, dxy/interpolate_degree, A_sc, eps, op_nrzns, n_wsamples, w_range)

    print("Generating realisations")
    R_vx, R_vy, _ = dg1.generate_R()
    max_vels, min_vels, mean_vels = dg1.find_max_vels(R_vx, R_vy)
    print("max_vels= ", max_vels)
    print("min_vels", min_vels)
    print("mean_vels", mean_vels)
    # print()

    print("Computing modes and coeffs")
    C, M, _, R_mean = dg1.compute_modes_and_coeffs(n_modes)
    phi_C, phi_M, _, phi_mean = dg1.compute_phi_modes_and_coefs(n_modes)
    # plot_modes_of_rank_reduced_field(dg1, C, M, 4, n_modes)
    # test_rank_reduced_field(dg1,1,0,n_modes)

    print("Reshaping matrices for grid")
    init_vel_field_data = dg1.reshape_matrices_for_square_grid(C, M, phi_M, R_mean, phi_mean, n_modes)

    print("Interpolating")
    all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi_mat, all_phi_mat, all_phi_i_mat = init_vel_field_data
    print("all_phi_i_mat.shape =", all_phi_i_mat.shape)
    print("all_phi_mat.shape =", all_phi_mat.shape)
    print("all_phi_mat[t,:,:].shape =", all_phi_mat[1,:,:].shape)

    vel_field_data = dg1.interpolate_data_to_refined_grid(all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_phi_mat, all_phi_i_mat, n_modes)
    vel_field_data.insert(4,all_Yi_mat)
    vel_field_data.append(phi_C)  #phi_C = all_phi_Yi_mat
    for x in vel_field_data:
        x = np.nan_to_num(x, copy=False)
    print()

    # save velocity files
    print("saving velocity files")
    files = ["all_u_mat.npy", "all_v_mat.npy", "all_ui_mat.npy", "all_vi_mat.npy", "all_Yi.npy","all_phi_mat.npy", "all_phi_i_mat.npy", "all_phi_Yi_mat.npy"]
    for i in range(len(files)):
        np.save(files[i], vel_field_data[i])
        print("Saved ", files[i], "shape= ", vel_field_data[i].shape,
                                    " type=", vel_field_data[i].dtype,
                                    "max_val=", np.max(vel_field_data[i]))

    del vel_field_data
    print()


    print("generating scalar field")
    all_s_mat = cloud.generate_phi()
    assert(all_s_mat.shape == (nt,final_gsize,final_gsize))
    # X,Y = np.meshgrid(np.array(cloud.xs)/interpolate_degree, np.flip(np.array(cloud.ys)/interpolate_degree))
    X,Y = np.meshgrid(np.array(cloud.xs), np.flip(np.array(cloud.ys)))
    test_scalar_field(all_s_mat, X, Y, nt)
    # print(C.shape, M.shape, R_mean.shape)

    obstacle_mask = np.zeros((nt, final_gsize, final_gsize), dtype = np.int32)
    obstacle_mask = fill_obstacles(obstacle_mask, final_gsize, nt, dyn_obstacle_data)

    scalar_field_data = [all_s_mat, obstacle_mask]
    scalar_files = ["all_s_mat.npy", "obstacle_mask.npy"]


    for i in range(len(scalar_field_data)):
        np.save(scalar_files[i], scalar_field_data[i])
        print("Saved ", scalar_files[i],"shape= ", scalar_field_data[i].shape,
                                    " type=", scalar_field_data[i].dtype,
                                    "max_val=", np.max(scalar_field_data[i]))

    print("\n ----- Saved field files! ------\n")

