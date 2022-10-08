#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include "cnpy.h"
#include <cmath>
#include <stdlib.h>
#include <chrono>
using namespace std::chrono;

#include <iostream>

long long int GPUmem = 8*1000*1000*1000LL; // using 1000 instead of 1024 
int nchunks = 1;
int chunk_size;
int last_chunk_size;
int thrust_fraction = 1; //expected  memory use for thrust method calls in terms of fraction of master_arr size
int avg_nnz_per_row = 10; // avg num  of nnz per row
long long int ncells;
long long int CszNrNa;  // abv for chunk_size*nrzns*num_actions
long long int CszNr;    // abv for chunk_size*num_actions
long long int CszNa;
long long int NcNrNa;   // abv for ncells*nrzns*num_actions


__device__ int32_t get_thread_idx(){
    // assigns idx to thread with which it accesses the flattened 3d vxrzns matrix
    // for a given T and a given action. 
    // runs for both 2d and 3d grid
    // TODO: may have to change this considering cache locality
    // here i, j, k refer to a general matrix M[i][j][k]
    int32_t i = threadIdx.x;
    int32_t j = blockIdx.y;
    int32_t k = blockIdx.x;
    int32_t idx = k + (j*gridDim.x)  + (i*gridDim.x*gridDim.y)+ blockIdx.z*blockDim.x*gridDim.x*gridDim.y;
    return idx;
}

__device__ long long int state1D_from_thread(int32_t T, int32_t sp_id, long long int ncells){   
    // j ~ blockIdx.x
    // i ~ blockIdx.y 
    // The above three consitute a spatial state index from i and j of grid
    // last term is for including time index as well.

        // return value when full spatial grid was used
        // return (blockIdx.x + (blockIdx.y*gridDim.x) + (T*gridDim.x*gridDim.y) ); 
    
    // return value for chunks concept
    return sp_id + (T*ncells);
}


__device__ long long int state1D_from_ij(int32_t*  posid, int32_t T, int32_t gsize){
    // posid = {i , j}
    // state id = j + i*dim(i) + T*dim(i)*dim(j)

        // return value when full spatial grid was used
        // return (posid[1] + posid[0]*gridDim.x + (T*gridDim.x*gridDim.y) ) ; 

    // return value for chunks concept
    return (posid[1] + posid[0]*gsize + (T*gsize*gsize) ) ; 

}


__device__ int32_t get_rzn_id(){

    return (blockIdx.z * blockDim.x)  + threadIdx.x;
}

__device__ void get_posids_from_sp_id(long long int sp_id, int gsize, int32_t* posids){

    posids[0] = sp_id/gsize;
    posids[1] = sp_id%gsize;
    return;
}

__device__ long long int get_sp_id(int chunkNum, int chunk_size){

    return (chunkNum*chunk_size)*1LL + blockIdx.x;
}


__device__ bool is_edge_state(int32_t i, int32_t j){
    // n = gsize -1 that is the last index of the domain assuming square domain
    int32_t n = gridDim.x - 1;
    if (i == 0 || i == n || j == 0 || j == n ) 
        return true;
    else 
        return false;
}


__device__ bool is_terminal(int32_t i, int32_t j, float* params){
    // terminal state indices (of UL corner of terminal subgrid if term_subgrid_size>1)
    int32_t i_term = params[8];         
    int32_t j_term = params[9];
    int tsgsize = params[12]; //term_subgrid_size

    if( (i >= i_term && i < i_term + tsgsize)  && (j >= j_term && j < j_term + tsgsize) )
        return true;
    else return false;
}


__device__ bool my_isnan(int s){
    // By IEEE 754 rule, NaN is not equal to NaN
    return s != s;
}


__device__ void get_xypos_from_ij(int32_t i, int32_t j, int32_t gsize ,float* xs, float* ys, float* x, float* y){
    *x = xs[j];
        // *y = ys[gridDim.x - 1 - i];
    *y = ys[gsize - 1 - i];

    return;
}


__device__ float get_angle_in_0_2pi(float theta){
    float f_pi = 3.141592;
    if (theta < 0)
        return theta + (2*f_pi);
    else
        return theta;
}


__device__ float calculate_reward_const_dt(float* xs, float* ys, int32_t i_old, int32_t j_old, float xold, float yold, int32_t* newposids, float* params, float vnet_x, float vnet_y ){
    // xold and yold are centre of old state (i_old, j_old)
    int gsize = params[0];
    float dt = params[4];
    float r1, r2, theta1, theta2, theta, h;
    float dt_new;
    float xnew, ynew;
    if (newposids[0] == i_old && newposids[1] == j_old)
        dt_new = dt;
    else
    {
        get_xypos_from_ij(newposids[0], newposids[1], gsize, xs, ys, &xnew, &ynew); //get centre of new states
        h = sqrtf((xnew - xold)*(xnew - xold) + (ynew - yold)*(ynew - yold));
        r1 = h/(sqrtf((vnet_x*vnet_x) + (vnet_y*vnet_y)));
        theta1 = get_angle_in_0_2pi(atan2f(vnet_y, vnet_x));
        theta2 = get_angle_in_0_2pi(atan2f(ynew - yold, xnew - xold));
        theta = fabsf(theta1 -theta2);
        r2 = fabsf(sinf(theta));
        dt_new = r1 + r2;
        if (threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 1 && blockIdx.y == 1)
        {
            params[24] = r1;
            params[25] = r2;
        }
    }
    return -dt_new;
}


__device__ void move(float ac_angle, float vx, float vy, int32_t T, float* xs, float* ys, int32_t* posids, float* params, float* r ){
    int32_t gsize = params[0];
    int32_t n = params[0] - 1;      // gsize - 1
    // int32_t num_actions = params[1];
    // int32_t nrzns = params[2];
    float F = params[3];
    float dt = params[4];
    float r_outbound = params[5];
    float r_terminal = params[6];
    // int32_t nT = params[10];
    float Dj = fabsf(xs[1] - xs[0]);
    float Di = fabsf(ys[1] - ys[0]);
    float r_step = 0;
    *r = 0;
    int32_t i0 = posids[0];
    int32_t j0 = posids[1];
    float vnetx = F*cosf(ac_angle) + vx;
    float vnety = F*sinf(ac_angle) + vy;
    float x, y;
    get_xypos_from_ij(i0, j0, gsize, xs, ys, &x, &y); // x, y stores centre coords of state i0,j0
    float xnew = x + (vnetx * dt);
    float ynew = y + (vnety * dt);
    
    //checks TODO: remove checks once verified
    if (threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 1 && blockIdx.y == 1)
    {
        params[12] = x;
        params[13] = y;
        params[14] = vnetx;
        params[15] = vnety;
        params[16] = xnew;
        params[17] = ynew;
        params[18] = ac_angle;
    }
    if (xnew > xs[n])
        {
            xnew = xs[n];
            *r += r_outbound;
        }
    else if (xnew < xs[0])
        {
            xnew = xs[0];
            *r += r_outbound;
        }
    if (ynew > ys[n])
        {
            ynew =  ys[n];
            *r += r_outbound;
        }
    else if (ynew < ys[0])
        {
            ynew =  ys[0];
            *r += r_outbound;
        }
    // TODO:xxDONE check logic wrt remainderf. remquof had issue
    int32_t xind, yind;
    //float remx = remquof((xnew - xs[0]), Dj, &xind);
    //float remy = remquof(-(ynew - ys[n]), Di, &yind);
    float remx = remainderf((xnew - xs[0]), Dj);
    float remy = remainderf(-(ynew - ys[n]), Di);
    xind = ((xnew - xs[0]) - remx)/Dj;
    yind = (-(ynew - ys[n]) - remy)/Di;
    if ((remx >= 0.5 * Dj) && (remy >= 0.5 * Di))
        {
            xind += 1;
            yind += 1;
        }
    else if ((remx >= 0.5 * Dj && remy < 0.5 * Di))
        {
            xind += 1;
        }
    else if ((remx < 0.5 * Dj && remy >= 0.5 * Di))
        {
            yind += 1;
        }
    if (!(my_isnan(xind) || my_isnan(yind)))
        {
            posids[0] = yind;
            posids[1] = xind;
            if (is_edge_state(posids[0], posids[1]))     //line 110
                {
                    *r += r_outbound;
                }
            
            if (threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 1 && blockIdx.y == 1)
            {
                // params[26] = 9999;
            }
        }
    // r_step = calculate_reward_const_dt(xs, ys, i0, j0, x, y, posids, params, vnetx, vnety);
    //TODO: change back to normal when needed
    r_step = -dt;
    *r += r_step; //TODO: numerical check remaining
    if (is_terminal(posids[0], posids[1], params))
        {
            *r += r_terminal;
        }
    
    if (threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 1 && blockIdx.y == 1)
    {
        params[19] = xnew;
        params[20] = ynew;
        params[21] = yind;
        params[22] = xind;
        // params[23] = *r;
        //params[17] = ynew;
        //params[18] = ac_angle;
    }
}


__device__ void extract_velocity(int32_t* posids, long long int sp_id, long long int ncells, float* vx, float* vy,
                                 int32_t T, float* all_u_mat, float* all_v_mat, float* all_ui_mat, 
                                 float* all_vi_mat, float* all_Yi, float* params){
    int32_t nrzns = params[2];
    int32_t nmodes = params[7];    
    int32_t gsize = params[0];          

    long long int sp_uvi, str_uvi, sp_Yi; //startpoints and strides for accessing all_ui_mat, all_vi_mat and all_Yi
    // int str_Yi;
    float sum_x = 0;
    float sum_y = 0;
    float vx_mean, vy_mean;
    //thread index. also used to access resultant vxrzns[nrzns, gsize, gsize]
    int32_t idx = get_thread_idx();
    //rzn index to identify which of the 5k rzn it is. used to access all_Yi.
    int32_t rzn_id = get_rzn_id() ;
    //mean_id is the index used to access the flattened all_u_mat[t,i,j].
    long long int mean_id = state1D_from_thread(T, sp_id, ncells);
    //to access all_ui_mat and all_vi_mat
        //str_uvi = gridDim.x * gridDim.y;
        // sp_uvi = (T * nmodes * str_uvi) + (gridDim.x * blockIdx.y) + (blockIdx.x);
    str_uvi = gsize*gsize*1LL;
    sp_uvi = (T * nmodes * str_uvi) + (gsize * posids[0]) + (posids[1]);

    // to access all_Yi
    sp_Yi = (T * nrzns * nmodes * 1LL) + (rzn_id * nmodes);
    vx_mean = all_u_mat[mean_id];
    for(int i = 0; i < nmodes; i++)
    {
        sum_x += all_ui_mat[sp_uvi + (i*str_uvi)]*all_Yi[sp_Yi + i];
    }
    vy_mean = all_v_mat[mean_id];
    for(int i = 0; i < nmodes; i++)
    {
        sum_y += all_vi_mat[sp_uvi + (i*str_uvi)]*all_Yi[sp_Yi + i];
    }

    *vx = vx_mean + sum_x;
    *vy = vy_mean + sum_y;

    return;
}



//test: changer from float* to float ac_angle
__global__ void transition_calc(float* T_arr, int chunkNum, int chunk_size, int eff_chunk_size, long long int ncells, 
                            float* all_u_mat, float* all_v_mat, float* all_ui_mat, float* all_vi_mat, float* all_Yi,
                            float ac_angle, float* xs, float* ys, float* params, float* sumR_sa, long long int* results)
                                    // resutls directions- 1: along S2;  2: along S1;    3: along columns towards count
{
    int32_t gsize = params[0];          // size of grid along 1 direction. ASSUMING square grid.
    int32_t nrzns = params[2];
           
    int32_t is_stationary = params[11];
    int32_t T = (int32_t)T_arr[0];      // current timestep
    int32_t idx = get_thread_idx();
    long long int new_idx;
    float vx, vy;
    long long int sp_id = get_sp_id(chunkNum, chunk_size);      //sp_id is space_id. S1%(gsize*gsize)

    if(idx < gridDim.x*gridDim.y*nrzns && sp_id < ncells) //or idx < arr_size
    {
        // int32_t posids[2] = {(int32_t)blockIdx.y, (int32_t)blockIdx.x};    //static declaration of array of size 2 to hold i and j values of S1. 
        int32_t posids[2];    //static declaration of array of size 2 to hold i and j values of S1. 
        get_posids_from_sp_id(sp_id, gsize, posids);
        int32_t rzn_id = get_rzn_id();
        if(idx == 0){
            params[23] = posids[0];
            params[24] = posids[1];
        }
        if(idx == 500){
            params[25] = posids[0];
            params[26] = posids[1];
        }
            
        //  Afer move() these will be overwritten by i and j values of S2
        float r=0;              // to store immediate reward
        extract_velocity(posids, sp_id, ncells, &vx, &vy, T, all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi, params);
        
        if (is_terminal(posids[0], posids[1], params) == false)
            move(ac_angle, vx, vy, T, xs, ys, posids, params, &r);

        if(idx == 0){
            params[27] = posids[0];
            params[28] = posids[1];
        }
        if(idx == 500){
            params[29] = posids[0];
            params[30] = posids[1];
        }

            // int32_t S1, S2;
        // long long int S1;
        long long int S2;

        if (is_stationary == 1)
        {
            T = 0;
            // S1 = state1D_from_thread(T, sp_id, ncells);     //get init state number corresponding to thread id
            S2 = state1D_from_ij(posids, T, gsize);   //get successor state number corresponding to posid and next timestep T+1        
        }
        else
        {
            // S1 = state1D_from_thread(T, sp_id, ncells);     //get init state number corresponding to thread id
            S2 = state1D_from_ij(posids, T+1, gsize);   //get successor state number corresponding to posid and next timestep T+1        
            // sp_id = S1%(gsize*gsize);
                // new_idx = rzn_id + (sp_id*nrzns);
            new_idx = rzn_id + ( (sp_id - (chunkNum * chunk_size)) * nrzns );
            // TODO: Corner case for last chunk
        }
        //writing to sumR_sa. this array will later be divided by nrzns, to get the avg
            // float a = atomicAdd(&sumR_sa[sp_id], r); 
        float a = atomicAdd(&sumR_sa[(sp_id - (chunkNum * chunk_size))], r); 

        // results[idx] = S2; // each chunk is ncells of one rzn. 
        results[new_idx] = S2;  // each chunk all rzns of one S1
        __syncthreads();
        // if (threadIdx.x == 0 && blockIdx.z == 0)
        //     sumR_sa[sp_id] = sumR_sa[sp_id]/nrzns;    //TODO: xxdone sumR_sa is now actually meanR_sa!

    }//if ends
    return;
}



__global__ void compute_mean(float* D_master_sumRsa_arr, int size, int nrzns) {

    int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
    if (tid < size)
        D_master_sumRsa_arr[tid] =  D_master_sumRsa_arr[tid]/nrzns;

    return;
}


__global__ void count_kernel(long long int* D_master_S2_arr_ip, int nrzns, long long int* num_uq_s2_ptr) {

    int tid = blockIdx.x;
    int nblocks = gridDim.x;  //ncells*num_actions  //chunk_size*num_action
    float count = 0;
    long long int old_s2 = -1;
    long long int new_s2;
    long long int start_idx = tid*nrzns;

    if (tid < nblocks){
        
        for(int i = 0; i < nrzns; i++){
            new_s2 = D_master_S2_arr_ip[start_idx + i];

            if ( new_s2 != old_s2){
                count++;
            }
            
            old_s2 = new_s2;
        }

        num_uq_s2_ptr[tid] = count;
    }

    return;
}


__global__ void reduce_kernel(long long int* D_master_S2_arr_ip, int t, int chunkNum, int chunk_size, int eff_chunk_size, long long int ncells, int nrzns, int nnz_xa_pc, 
                            long long int* D_coo_s1_arr, long long int* D_coo_s2_arr, 
                            float* D_coo_cnt_arr, long long int* num_uq_s2_ptr, long long int* prSum_num_uq_s2_ptr){
   
    int tid = blockIdx.x;
    int nblocks = gridDim.x;  // eff_chunk_size*num_actions 
    int start_idx = tid*nrzns; // to access tid'th threads 0-pos in ip_arr

    int n_uqs = num_uq_s2_ptr[tid]; //number of unique S2s for tid'th block
    int op_st_id = prSum_num_uq_s2_ptr[tid];   //sum of number of uniqeu S2s uptil tid'th block. to access tid'th thread's 0-pos in op_arr

    int ith_nuq = 0; //ranges from 0 to n_uqs , to index number between 0 and n_uqs

    long long int old_s2 = D_master_S2_arr_ip[start_idx];
    long long int new_s2;
    float count = 0; //first if eval will lead to else condition and do  count++ 


    if (tid < nblocks){

            // int32_t s1 = (tid%ncells) + (t*ncells); // TODO:xxdone change this to nbe a function of a arguments: sp_id and t
        long long int s1 = chunkNum*chunk_size + (tid%chunk_size) + (t*ncells);
        for(int i = 0; i< n_uqs; i++)
            D_coo_s1_arr[op_st_id + i] = s1;

        for(int i = 0; i< nrzns; i++){
            new_s2 = D_master_S2_arr_ip[start_idx + i];
            if (new_s2 != old_s2){                              // on encountering new value in the sorted array
                D_coo_s2_arr[op_st_id + ith_nuq] = old_s2;         // store old_s2 value in the [.. + ith] position
                D_coo_cnt_arr[op_st_id + ith_nuq] = count/nrzns;   // store prob value in the [.. + ith] position
                ith_nuq++;                                      // increment i
                count = 1;      //restart count on encounter new element
            }
            else
                count++;

            old_s2 = new_s2;

        }

        // to store information about the last of n_uqs S2s
        if (ith_nuq < n_uqs ){   //this condition should always be true because i assert ith_nuq == n_uqs - 1
            D_coo_s2_arr[op_st_id + ith_nuq] = old_s2;         // store old_s2 value in the [.. + ith] position
            D_coo_cnt_arr[op_st_id + ith_nuq] = count/nrzns;   // store prob value in the [.. + ith] position
            ith_nuq++;                                      // increment i
        }

   }
   return;
}
template<typename dType>
void print_array(dType* array, int num_elems,std::string array_name, std::string end){
    std::cout << array_name << std::endl;
    for(int i = 0; i < num_elems; i++)
        std::cout << array[i] << " " << end;
    std::cout << std::endl;
}

// template<typename dType>
void print_device_vector( thrust::device_vector<long long int> &array, int start_id, int end_id, std::string array_name, std::string end, int method){
    std::cout << array_name << "  from id " << start_id << "  to  " << end_id << std::endl;
    if (method == 1){
        float temp = -10000000;
        for(int i = start_id; i < end_id; i++){
            if (array[i] != temp){
                std::cout << i << "\n";
                std::cout << array[i] << " " << end;
                std::cout << "\n";
                temp = array[i];
            }
        }
    }

    else if (method == 0){
        for(int i = start_id; i < end_id; i++)
            std::cout << array[i] << " " << end;
    }

    else
        std::cout << "Invalid input for argument: method";


    std::cout << std::endl;
}

// build_sparse_transition_model_at_T(t, bDimx, D_tdummy, D_all_u_arr, D_all_v_arr 
//     ,D_all_ui_arr, D_all_vi_arr, D_all_yi_arr,
//     D_params, D_xs, D_ys, H_ac_angles, D_master_vals,
//     H_coo_len_per_ac,
//     H_Aarr_of_cooS1, H_Aarr_of_cooS2, H_Aarr_of_cooProb,
//     H_Aarr_of_Rs);

void build_sparse_transition_model_at_T(int t, int bDimx, thrust::device_vector<float> &D_tdummy, 
                                        float* D_all_u_arr, float* D_all_v_arr, float* D_all_ui_arr,
                                        float*  D_all_vi_arr, float*  D_all_yi_arr,
                                        thrust::device_vector<float> &D_params, thrust::device_vector<float> &D_xs, 
                                        thrust::device_vector<float> &D_ys, thrust::host_vector<float> &H_ac_angles,
                                        thrust::device_vector<long long int> &D_master_vals,
                                        thrust::host_vector<int32_t> &H_coo_len_per_ac,
                                        thrust::host_vector<long long int>* H_Aarr_of_cooS1,
                                        thrust::host_vector<long long int>* H_Aarr_of_cooS2,
                                        thrust::host_vector<float>* H_Aarr_of_cooProb,
                                        thrust::host_vector<float>* H_Aarr_of_Rs
                                        );
// void concatenate_results_across_time();

void build_sparse_transition_model_at_T(int t, int bDimx, thrust::device_vector<float> &D_tdummy, 
                                float* D_all_u_arr, float* D_all_v_arr, float* D_all_ui_arr,
                                float*  D_all_vi_arr, float*  D_all_yi_arr,
                                thrust::device_vector<float> &D_params, thrust::device_vector<float> &D_xs, 
                                thrust::device_vector<float> &D_ys, thrust::host_vector<float> &H_ac_angles,
                                thrust::device_vector<long long int> &D_master_vals,
                                thrust::host_vector<int32_t> &H_coo_len_per_ac,
                                thrust::host_vector<long long int>* H_Aarr_of_cooS1,
                                thrust::host_vector<long long int>* H_Aarr_of_cooS2,
                                thrust::host_vector<float>* H_Aarr_of_cooProb,
                                thrust::host_vector<float>* H_Aarr_of_Rs
                                ){

    int gsize = (int) D_params[0];
    int num_actions =  (int)D_params[1];
    int nrzns = (int) D_params[2];
    int nt = (int) D_params[10];

    // // check velocity data and vector data
    // std::cout << "D_paramas" << std::endl;
    // for (int i = 0; i< 10; i ++)
    //     std::cout << D_params[i] << std::endl;

    // I think doing it this way does not issue a memcpy at the backend. thats why it fails
    // std::cout << "D_all_u_arr" << std::endl;
    // for (int i = 0; i< 10; i ++)
    //     std::cout << D_all_u_arr[i] << std::endl;                                 

    // raw pointer casts
    float* D_T_arr = thrust::raw_pointer_cast(&D_tdummy[0]);
    float* xs = thrust::raw_pointer_cast(&D_xs[0]);
    float* ys = thrust::raw_pointer_cast(&D_ys[0]);
    float* params = thrust::raw_pointer_cast(&D_params[0]);

    // print_array<float>(xs, 10, "xs", "");
    std::cout << "D_xs= " ;
    for (int i = 0; i< 10; i++)
        std::cout << D_xs[i] << " " ;

        // int arr_size = chunk_size * nrzns;
        // thrust::host_vector<float> H_S2_vec(arr_size, 0); //eqv of results
        // thrust::host_vector<float> H_sumR_sa(ncells, 0);
    

    //Define Kernel launch parameters for transition calculation kernel
    int DimGrid_z = (nrzns/bDimx)+1;
    if (nrzns % bDimx == 0)
        DimGrid_z = (nrzns/bDimx);

    // checks
    if (t == nt-2){
        std::cout << "t = " << t << "\n nt = " << nt << "\n" ; 
        std::cout<<"gisze= " << gsize << std::endl;
        std::cout<<"DimGrid_z = " << DimGrid_z << std::endl;
        std::cout<<"bDimx = " <<  bDimx << std::endl;
    }
 
    // std::cout << "pre/post move posids (i,j) at 0" << "\n";

    int eff_chunk_size = chunk_size;
    long long int efCszNa, efCszNr, efCszNrNa;
    efCszNa = eff_chunk_size*num_actions;
    efCszNr = eff_chunk_size * nrzns;
    efCszNrNa = eff_chunk_size * nrzns * num_actions;

    // initialse master S2 array -  S2_array concated across all actions
        // int master_arr_size = arr_size*num_actions;
        //thrust::device_vector<int32_t> D_master_S2_vector(master_arr_size);
    thrust::device_vector<long long int> D_master_S2_vector(efCszNrNa);
    long long int* D_master_S2_arr = thrust::raw_pointer_cast(&D_master_S2_vector[0]);

    for (int chunkNum = 0; chunkNum < nchunks; chunkNum++){
        
        std::cout << "***   chunkNum = " << chunkNum << "\n";
        if (chunkNum == nchunks - 1){
            eff_chunk_size = last_chunk_size;
            efCszNa = eff_chunk_size*num_actions;
            efCszNr = eff_chunk_size * nrzns;
            efCszNrNa = eff_chunk_size * nrzns * num_actions;

            // inside if to prevenent reeinitialisation costs at each chunknum. reinitiaise only at last chunknum
            thrust::device_vector<long long int> D_master_S2_vector(efCszNrNa);
            long long int* D_master_S2_arr = thrust::raw_pointer_cast(&D_master_S2_vector[0]);
        }
           
        std::cout << "***   eff_chunk_size = " << eff_chunk_size << "\n";
        // initialise master sum_Rsa array - sumRsa's concated across all actions
        // Important to initialise it with 0
            // thrust::device_vector<float> D_master_sumRsa_vector(ncells*num_actions, 0);
        // need to intiilase with 0 at for each chunknum
        thrust::device_vector<float> D_master_sumRsa_vector(efCszNa, 0);
        float* D_master_sumRsa_arr = thrust::raw_pointer_cast(&D_master_sumRsa_vector[0]);


        // define kerel block and grid configuration
            // dim3 DimGrid(gsize, gsize, DimGrid_z);
        dim3 DimGrid(eff_chunk_size, 1, DimGrid_z);
        dim3 DimBlock(bDimx, 1, 1);

        for(int n = 0; n < num_actions; n++){

            // std::cout <<  std::endl <<"     a = " << n << std::endl;
            float ac_angle = H_ac_angles[n];

            // launch kernel for @a @t
            transition_calc<<< DimGrid, DimBlock  >>> (D_T_arr, chunkNum, chunk_size, eff_chunk_size, 
                ncells, D_all_u_arr, D_all_v_arr, D_all_ui_arr, D_all_vi_arr, D_all_yi_arr,
                ac_angle, xs, ys, params, D_master_sumRsa_arr + n*eff_chunk_size, D_master_S2_arr + n*efCszNr);

            cudaDeviceSynchronize();

            // // CHECK copy data back to host for check
            // std::cout << "a" << n <<"\n vx at s1=0: " << D_params[31] << std::endl;
            // std::cout <<"\n vx at s1=0: " << D_params[30] << std::endl;
            // std::cout << "----a" << n <<"\n";
            // std::cout <<"pre move " << "\n";
            // std::cout<<"r1\n"<< D_params[23] << "," << D_params[24] << std::endl;
            // std::cout<<"r2\n"<< D_params[25] << "," << D_params[26] << std::endl;
            // std::cout <<"post move " << "\n";
            // std::cout<<"r1\n"<< D_params[27] << "," << D_params[28] << std::endl;
            // std::cout<<"r2\n"<< D_params[29] << "," << D_params[30] << std::endl;

            // thrust::copy(D_master_S2_vector.begin() + n*arr_size, D_master_S2_vector.begin() + (n+1)*arr_size, H_S2_vec.begin());
            // thrust::copy(D_master_sumRsa_vector.begin() + n*ncells, D_master_sumRsa_vector.begin() + (n+1)*ncells, H_sumR_sa.begin());
            // std::cout << "post kernel" << std::endl;
            // for(int i = 0; i < 10; i ++)
            //     std::cout << H_sumR_sa[i] << std::endl;
            // for(int i = 0; i < 10; i ++)
            //     std::cout << H_S2_vec[i] << std::endl;
            
        }

        int Nthreads = D_master_sumRsa_vector.size();
        int threads_per_block = 1024;
        int blocks_per_grid = (Nthreads/threads_per_block) + 1;

        std::cout<<"cmpute_mean\n Nthreads = " << Nthreads << "\n threads_per_block= " << threads_per_block << "\n blocks_per_grid= " <<blocks_per_grid << "\n";
        if ( blocks_per_grid * threads_per_block < Nthreads)
            std::cout << "NOOOOOOOOOOOOOOOO----------------------------\n";

        compute_mean<<< blocks_per_grid, threads_per_block >>>(D_master_sumRsa_arr, Nthreads, nrzns);

        // TODO: in optimazation phase move this line after initilisation num_uq_S2 vectors.
        cudaDeviceSynchronize();
        //initialising vectors for counting nnzs or number of uniqe S2s for S1s
        //Hopefully, this will go on parallelly with the last kernel
        
            // thrust::device_vector<int32_t> D_num_uq_s2(ncells*num_actions,0);
            // thrust::device_vector<int32_t> D_prSum_num_uq_s2(ncells*num_actions);
        //TODO: corner case for last chunk size
        thrust::device_vector<long long int> D_num_uq_s2_pc(efCszNa,0);
        thrust::device_vector<long long int> D_prSum_num_uq_s2_pc(efCszNa);
        long long int* num_uq_s2_ptr = thrust::raw_pointer_cast(&D_num_uq_s2_pc[0]);
        long long int* prSum_num_uq_s2_ptr = thrust::raw_pointer_cast(&D_prSum_num_uq_s2_pc[0]);
            
            int tprint = nt - 2;
            if (t == tprint)
                print_device_vector(D_master_S2_vector,0, 10, "pre-sort: D_master_S2_vector", " ", 0);

        // Sort master_data
        // float* D_master_S2_arr_ip = thrust::raw_pointer_cast(&D_master_S2_vector[0]);
        thrust::stable_sort_by_key(D_master_S2_vector.begin(), D_master_S2_vector.end(), D_master_vals.begin());
        thrust::stable_sort_by_key(D_master_vals.begin(), D_master_vals.end(), D_master_S2_vector.begin());

            if (t == tprint)
                print_device_vector(D_master_S2_vector,0, 10, "post-sort: D_master_S2_vector", " ", 0);

        // launch kernel to count nnzs
            // int nblocks = ncells*num_actions;
        count_kernel<<<efCszNa, 1>>>(D_master_S2_arr, nrzns, num_uq_s2_ptr);
        cudaDeviceSynchronize();


        // std::cout << "D_num_uq_s2_pc\n";
        // int tempflag = 0;
        // int tempnum;
        // int cnt2 = 0;
        // int cnt1 = 0;
        // for (int i =0; i < efCszNa; i++){
        //     tempnum = D_num_uq_s2_pc[i];
        //     if (tempnum == 1)
        //         cnt1++;
        //     else if (tempnum == 2)
        //         cnt2++;
        //     else
        //         std::cout << " --------------------------- WRONG-----------\n";
        // }
        
        // std::cout << "cnt1 = " << cnt1 << "\ncnt2 = " << cnt2 <<"\n";




        // calc nnz_xa_pc: number of non zero elements(or unique S2s) across(multiplied by) num_actions actions for a given chunk
        long long int nnz_xa_pc = thrust::reduce(D_num_uq_s2_pc.begin(), D_num_uq_s2_pc.end(), (float) 0, thrust::plus<float>());
        // get prefix sum of D_num_uq_s2_pc. This helps threads to access apt COO indices in reduce_kernel
        thrust::exclusive_scan(D_num_uq_s2_pc.begin(), D_num_uq_s2_pc.end(), D_prSum_num_uq_s2_pc.begin());
        std::cout << "nnz_xa_pc = " << nnz_xa_pc << "\n";


        //initilise coo arrays (concated across actions)
        thrust::device_vector<long long int> D_coo_s1(nnz_xa_pc);
        thrust::device_vector<long long int> D_coo_s2(nnz_xa_pc);
        thrust::device_vector<float> D_coo_count(nnz_xa_pc); // TODO: makde this int32_t and introduce another array for prob
        long long int* D_coo_s1_arr = thrust::raw_pointer_cast(&D_coo_s1[0]);
        long long int* D_coo_s2_arr = thrust::raw_pointer_cast(&D_coo_s2[0]);
        float* D_coo_cnt_arr = thrust::raw_pointer_cast(&D_coo_count[0]);


        // reduce operation to fill COO arrays
        reduce_kernel<<<efCszNa, 1>>>(D_master_S2_arr, t, chunkNum, chunk_size, eff_chunk_size, 
                                    ncells, nrzns, nnz_xa_pc, D_coo_s1_arr, D_coo_s2_arr, D_coo_cnt_arr, 
                                    num_uq_s2_ptr, prSum_num_uq_s2_ptr);
        cudaDeviceSynchronize();

        print_device_vector(D_coo_s1, 0, 10, "D_coo_s1", " ", 0);
        print_device_vector(D_coo_s2, 0, 10, "D_coo_s2", " ", 0);

        //reduce D_num_uq_s2_pc in chunks of actions - to find nnz or len_coo_arr for each action
        for (int n = 0; n < num_actions; n++){
                // H_coo_len_per_ac[n] = thrust::reduce(D_num_uq_s2_pc.begin() + n*ncells, D_num_uq_s2_pc.begin() +  (n+1)*ncells, (float) 0, thrust::plus<float>());
            H_coo_len_per_ac[n] = thrust::reduce(D_num_uq_s2_pc.begin() + n*eff_chunk_size, D_num_uq_s2_pc.begin() + (n+1)*eff_chunk_size, (float) 0, thrust::plus<float>());
        }
        thrust::inclusive_scan(H_coo_len_per_ac.begin(), H_coo_len_per_ac.end(), H_coo_len_per_ac.begin());


        //check
        std::cout << "H_coo_len_per_ac" << std::endl;
        for (int n = 0; n < num_actions; n++)
          std::cout << H_coo_len_per_ac[n] << std::endl;


        // Copy Device COO rusults to Host COO vectors across actions and append vectors across time
        H_Aarr_of_cooS1[0].insert(H_Aarr_of_cooS1[0].end(), D_coo_s1.begin(), D_coo_s1.begin() + H_coo_len_per_ac[0]);
        for (int n = 1; n < num_actions; n++){
            H_Aarr_of_cooS1[n].insert(H_Aarr_of_cooS1[n].end(), D_coo_s1.begin() + H_coo_len_per_ac[n-1], D_coo_s1.begin() + H_coo_len_per_ac[n]);
        }

        H_Aarr_of_cooS2[0].insert(H_Aarr_of_cooS2[0].end(), D_coo_s2.begin(), D_coo_s2.begin() + H_coo_len_per_ac[0]);
        for (int n = 1; n < num_actions; n++){
            H_Aarr_of_cooS2[n].insert(H_Aarr_of_cooS2[n].end(), D_coo_s2.begin() + H_coo_len_per_ac[n-1], D_coo_s2.begin() + H_coo_len_per_ac[n]);
        }

        H_Aarr_of_cooProb[0].insert(H_Aarr_of_cooProb[0].end(), D_coo_count.begin(), D_coo_count.begin() + H_coo_len_per_ac[0]);
        for (int n = 1; n < num_actions; n++){
            H_Aarr_of_cooProb[n].insert(H_Aarr_of_cooProb[n].end(), D_coo_count.begin() + H_coo_len_per_ac[n-1], D_coo_count.begin() + H_coo_len_per_ac[n]);
        }

        for (int n = 0; n < num_actions; n++){
                // H_Aarr_of_Rs[n].insert(H_Aarr_of_Rs[n].end(), D_master_sumRsa_vector.begin() + n*ncells, D_master_sumRsa_vector.begin() + (n+1)*ncells);
            H_Aarr_of_Rs[n].insert(H_Aarr_of_Rs[n].end(), D_master_sumRsa_vector.begin() + n*eff_chunk_size, D_master_sumRsa_vector.begin() + (n+1)*eff_chunk_size);
        }

        // std::cout << "H_Aarr_of_cooS1" << std::endl;
        // for (int n = 0; n < num_actions; n++){
        //     for (int i = 0; i < H_Aarr_of_cooS1[n].size(); i++)
        //         std::cout << H_Aarr_of_cooS1[n][i] << " , " << H_Aarr_of_cooS2[n][i] << " , " << H_Aarr_of_cooProb[n][i] << std::endl;
        //     std::cout << std::endl;
        // }

        // std::cout << "H_Aarr_of_Rs" << std::endl;
        // for (int n = 0; n < num_actions; n++){
        //     for (int i = 0; i < ncells; i++)
        //         std::cout << H_Aarr_of_Rs[n][i] << std::endl;
        //     std::cout << std::endl;
        // }


        // // array of num_actions decive_vvectors for sum_Rsa_vec
        // // initialasation with 0 is important. because values are added to this
        // thrust::host_vector<float> H_arr_sumR_sa[num_actions];
        // for(int n = 0; n < num_actions; n++){
        //     H_arr_sumR_sa[n] = thrust::host_vector<float>(nnz[i]);
    }

}





void get_cell_chunk_partition(int gsize, int ncells, int nrzns, int num_actions,
                int nmodes, int nt, int thrust_fraction, int avg_nnz_per_row,
                int* nchunks, int* chunk_size, int* last_chunk_size){
    // reads dimensions of input data related to the problem and returns sizes
    // and number of chunks into which ncells (spatial grid) is divided.
    // So as to be able to fit all necesarry data structures in GPU memory

    // long long int max_s_val = ncells*nt*num_actions*1LL;
    long long int master_arr_size_term ;
    // if (mas_s_val < 2147483647) //TODO: dynamic datatype allocation int vs long long int based on 
                                    // max_s_value possible in master_s2_array
    master_arr_size_term = 16*nrzns*num_actions; // 2 long long int arrays
    long long int vdata_size_term = 8*nt*(nmodes+1);
    long long int coo_term = 8*3*avg_nnz_per_row*num_actions;
    int k = thrust_fraction;
    long long int denom = ((1+k)*master_arr_size_term) + vdata_size_term + coo_term;

    std::cout << "master_arr_size_term= " << master_arr_size_term << "\n" 
        << "vdata_size_term = " <<vdata_size_term << "\n"
        << "denom = " << denom << "\n" ;

    int local_chunk_size = (int) (GPUmem/denom);
    if (local_chunk_size < ncells){
        *chunk_size = local_chunk_size;
        *nchunks = (ncells/local_chunk_size) + 1;
        *last_chunk_size = ncells - ( (local_chunk_size)*(*nchunks - 1) );
    }
    else{
        *chunk_size = ncells/2;
        *last_chunk_size = ncells/2;
        *nchunks = 2;
    }

    std::cout << "nchunks = " << *nchunks << "\n" 
    << "chunk_size = " << *chunk_size << "\n" 
    << "last_chunk_size = " << *last_chunk_size << "\n";

return;
}
       

// function declarations
cnpy::NpyArray read_velocity_field_data( std::string file_path_name, int* n_elements);
void define_xs_or_ys(float* xs, float dx, float x0, int gsize);
void populate_ac_angles(float* ac_angles, int num_actions);
void save_master_Coos_to_file(std::string op_FnamePfx, int num_actions, 
    thrust::host_vector<long long int> &H_master_cooS1, 
    thrust::host_vector<long long int> &H_master_cooS2, 
    thrust::host_vector<float> &H_master_cooVal,
    thrust::host_vector<float> &H_master_R,
    thrust::host_vector<long long int>* H_Aarr_of_cooS1,
    thrust::host_vector<long long int>* H_Aarr_of_cooS2,
    thrust::host_vector<float>* H_Aarr_of_cooProb,
    thrust::host_vector<float>* H_Aarr_of_Rs,
    thrust::host_vector<float> &prob_params,
    long long int* DP_relv_params,
    unsigned long int num_DP_params);


// global variables
std::string str_x = "x";
std::string str_g = "g";
std::string str_r = "r";
std::string str_a = "a";
std::string str_us = "_";
   // std::string str_probDim = str_g + std::to_string(gsize) + 
    //                         str_x + std::to_string(gsize) + str_x +
    //                         str_x + std::to_string(nt) + str_us; //"gsizexgsizexnt_"
    // std::string str_probRzns = str_r + std::to_string(nrzns) + str_us;
    // std::string str_probAc = str_a + std::to_string(num_actions);

    // std::string str_prob_type = "all_jet_";
    // std::string str_probVersion = str_us + str_v



int main(){

// -------------------- input data starts here ---------------------------------

    //  // DG3 data
    // // TODO: take parameters form different file
    // std::string op_FnamePfx = "data/output/test_DG_nt60/"; //path for storing op npy data.

    // float nt = 60;
    // float is_stationary = 0;
    // float gsize = 100;
    // float num_actions = 8;
    // float nrzns = 5000;
    // float bDimx = nrzns;
    // float F = 20.2;
    // float r_outbound = -10;
    // float r_terminal = 10;
    // float i_term = 19;
    // float j_term = 40;
    // float nmodes = 5;
    // float x0 = 0.005;
    // float y0 = 0.005;
    // float dx = 0.01; float dy = 0.01;
    // float dt = 0.0004;
    // if (nrzns >= 1000)
    //     bDimx = 1000;

    // //TODO: define output file savepath

    // float z = -9999;
    // // TODO: 1. read paths form file
    // //       2. Make sure files are stored in np.float32 format
    // std::string data_path = "data/nT_60/";
    // std::string all_u_fname = data_path + "all_u_mat.npy";
    // std::string all_v_fname = data_path + "all_v_mat.npy";
    // std::string all_ui_fname = data_path + "all_ui_mat.npy";
    // std::string all_vi_fname = data_path + "all_vi_mat.npy";
    // std::string all_yi_fname = data_path + "all_Yi.npy";
    
   

// ---------------------------------------------------------

    // // simple grid
    // std::string op_FnamePfx = "data/output/test_DG_nt60/"; //path for storing op npy data.

    // float nt = 3;
    // float is_stationary = 0;
    // float gsize = 5;
    // float num_actions = 8;
    // float nrzns = 5;
    // float bDimx = nrzns;
    // float F = 1;
    // float r_outbound = -1;
    // float r_terminal = 1;
    // float i_term = 1;
    // float j_term = 3;
    // float nmodes = 1;
    // float x0 = 0.5;
    // float y0 = 0.5;
    // float dx = 1; float dy = 1;
    // float dt = 1;
    // if (nrzns >= 1000)
    //     bDimx = 1000;
 
    // float z = -9999;
    // // TODO: 1. read paths form file
    // //       2. Make sure files are stored in np.float32 format
    // std::string data_path = "data/jet_5x5/";
    // std::string all_u_fname = data_path + "all_u_mat.npy";
    // std::string all_v_fname = data_path + "all_v_mat.npy";
    // std::string all_ui_fname = data_path + "all_ui_mat.npy";
    // std::string all_vi_fname = data_path + "all_vi_mat.npy";
    // std::string all_yi_fname = data_path + "all_Yi.npy";

// ---------------------------------------------------------


 // // 10x10x10 grid. jet across grid varying between 1-2 units/sec.

    std::string prob_name = "all_jet_refg200x200x200_r10";
    std::string op_FnamePfx = "data/model_output/" + prob_name + "/"; //path for storing op npy data.

    int32_t nt = 200;
    float dt = 1;
    int32_t gsize = 200;
    float dx = 0.5; float dy = 0.5;
    float x0 = dx/2;
    float y0 = dy/2;
    int32_t num_actions = 16;
    int32_t nrzns = 10;
    int32_t bDimx = nrzns;
    float F = 1;
    float r_outbound = -100;
    float r_terminal = 10;
    // i_term and j_term are (i,j) coords for the TOP LEFT CORNER
    // of the square subgrid that constitutes the terminal states
    // int32_t i_term = 4;
    // int32_t j_term = 7;
    int32_t i_term = 100; //50
    int32_t j_term = 180; //90
    int term_subgrid_size = 2; //number of cells al
    
    float nmodes = 1;

    int32_t is_stationary = 0;
    if (nrzns >= 1000)
        bDimx = 1000;
 
    float z = -9999;
    // TODO: 1. read paths form file
    //       2. Make sure files are stored in np.float32 format
    std::string data_path = "data/" + prob_name + "/";
    std::string all_u_fname = data_path + "all_u_mat.npy";
    std::string all_v_fname = data_path + "all_v_mat.npy";
    std::string all_ui_fname = data_path + "all_ui_mat.npy";
    std::string all_vi_fname = data_path + "all_vi_mat.npy";
    std::string all_yi_fname = data_path + "all_Yi.npy";

// -------------------- input data ends here ---------------------------------

    int mkdir_status;
    std::string comm_mkdir = "mkdir ";
    std::string str = comm_mkdir + op_FnamePfx;
    const char * full_command = str.c_str();
    mkdir_status = system(full_command);
    std::cout << "mkdir_status = " << mkdir_status << std::endl;


    int all_u_n_elms;
    int all_v_n_elms;
    int all_ui_n_elms;
    int all_vi_n_elms;
    int all_yi_n_elms;

    cnpy::NpyArray all_u_cnpy = read_velocity_field_data(all_u_fname, &all_u_n_elms);
    cnpy::NpyArray all_v_cnpy = read_velocity_field_data(all_v_fname, &all_v_n_elms);
    cnpy::NpyArray all_ui_cnpy = read_velocity_field_data(all_ui_fname, &all_ui_n_elms);
    cnpy::NpyArray all_vi_cnpy = read_velocity_field_data(all_vi_fname, &all_vi_n_elms);
    cnpy::NpyArray all_yi_cnpy = read_velocity_field_data(all_yi_fname, &all_yi_n_elms);

    float* all_u_mat = all_u_cnpy.data<float>();
    float* all_v_mat = all_v_cnpy.data<float>();
    float* all_ui_mat = all_ui_cnpy.data<float>();
    float* all_vi_mat = all_vi_cnpy.data<float>();
    float* all_yi_mat = all_yi_cnpy.data<float>();

    // print_array<float>(all_u_mat, all_u_n_elms, "all_u_mat", " ");
    // print_array<float>(all_ui_mat, all_ui_n_elms,"all_ui_mat", " ");
    // print_array<float>(all_yi_mat, all_yi_n_elms,"all_yi_mat", " ");

    std::cout << "Finished reading Velocity Field Data !" << std::endl;


    //TODO: fill params in a function
    // Contains implicit casting from int32_t to float
    thrust::host_vector<float> H_params(32);
    H_params[0] = gsize;
    H_params[1] = num_actions; 
    H_params[2] = nrzns;
    H_params[3] = F;
    H_params[4] = dt;
    H_params[5] = r_outbound;
    H_params[6] = r_terminal;
    H_params[7] = nmodes;
    H_params[8] = i_term;
    H_params[9] = j_term;
    H_params[10] = nt;
    H_params[11] = is_stationary;
    H_params[12] = term_subgrid_size;
    for( int i =13; i<32; i++)
        H_params[i] = z;

    // Define grid ticks in host
    thrust::host_vector<float> H_xs(gsize, -1);
    thrust::host_vector<float> H_ys(gsize, -1);
    float* xs = thrust::raw_pointer_cast(&H_xs[0]);
    float* ys = thrust::raw_pointer_cast(&H_ys[0]);
    //TODO:  2. move the fucntion to a separate file
    define_xs_or_ys(xs, dx, x0, gsize);
    define_xs_or_ys(ys, dy, y0, gsize);

    // define angles in host
    thrust::host_vector<float> H_ac_angles(num_actions);
    float* ac_angles = thrust::raw_pointer_cast(&H_ac_angles[0]);
    //TODO: move to custom functions
    populate_ac_angles(ac_angles, num_actions);
    print_array<float>(ac_angles, num_actions, "ac_angles", "\n");

    //----- start copying data to device --------

    // Copy vel field data to device memory using thrust device_vector
    thrust::device_vector<float> D_all_u_vec (all_u_mat, all_u_mat + all_u_n_elms);
    thrust::device_vector<float> D_all_v_vec (all_v_mat, all_v_mat + all_v_n_elms);
    thrust::device_vector<float> D_all_ui_vec (all_ui_mat, all_ui_mat + all_ui_n_elms);
    thrust::device_vector<float> D_all_vi_vec (all_vi_mat, all_vi_mat + all_vi_n_elms);
    thrust::device_vector<float> D_all_yi_vec (all_yi_mat, all_yi_mat + all_yi_n_elms);

    float* D_all_u_arr = thrust::raw_pointer_cast(&D_all_u_vec[0]);
    float* D_all_v_arr = thrust::raw_pointer_cast(&D_all_v_vec[0]);
    float* D_all_ui_arr = thrust::raw_pointer_cast(&D_all_ui_vec[0]);
    float* D_all_vi_arr = thrust::raw_pointer_cast(&D_all_vi_vec[0]);
    float* D_all_yi_arr = thrust::raw_pointer_cast(&D_all_yi_vec[0]);


    std::cout << "Copied to Device : Velocity Field Data !" << std::endl;

    thrust::device_vector<float> D_tdummy(2,0);
    // initialise empty device vectors. These contain time-invariant data
    thrust::device_vector<float> D_params(32);
    thrust::device_vector<float> D_xs(gsize);
    thrust::device_vector<float> D_ys(gsize);

    // initialise reuseable host vectors
    thrust::host_vector<int32_t> H_coo_len_per_ac(num_actions);
    thrust::host_vector<long long int> H_Aarr_of_cooS1[(int)num_actions];
    thrust::host_vector<long long int> H_Aarr_of_cooS2[(int)num_actions];
    thrust::host_vector<float> H_Aarr_of_cooProb[(int)num_actions];
    thrust::host_vector<float> H_Aarr_of_Rs[(int)num_actions];
    //initialised with 0 size. later data from device is inserted/appended to the end of vector
    for (int i =0; i < num_actions; i++){
        H_Aarr_of_cooS1[i] = thrust::host_vector<long long int> (0);
    }
    for (int i =0; i < num_actions; i++){
        H_Aarr_of_cooS2[i] = thrust::host_vector<long long int> (0);
    }
    for (int i =0; i < num_actions; i++){
        H_Aarr_of_cooProb[i] = thrust::host_vector<float> (0);
    }
    for (int i =0; i < num_actions; i++){
        H_Aarr_of_Rs[i] = thrust::host_vector<float> (0);
    }

    ncells = gsize*gsize;                 // assign value to global variable
    chunk_size = ncells;                  // assgin default value to global varibale
    last_chunk_size = ncells;
    get_cell_chunk_partition(gsize, ncells, nrzns, num_actions,
        nmodes, nt, thrust_fraction, avg_nnz_per_row, &nchunks, &chunk_size, &last_chunk_size);
 
    CszNr = chunk_size * nrzns;
    CszNa = chunk_size * num_actions;
    CszNrNa = chunk_size * nrzns * num_actions * 1LL;
    std::cout << "CszNrNa = " << CszNrNa << "\n";

    
    //initialise master_value_vector for sort_by_key
    thrust::host_vector<long long int> H_master_vals(CszNrNa); //TODO: Chage name to H_mastera_sortVals
    // thrust::host_vector<int32_t> temp_test(CszNrNa);


    for (int i = 0; i < CszNrNa; i++)
        H_master_vals[i] = (i/nrzns);
    thrust::device_vector<long long int> D_master_vals(CszNrNa);
    D_master_vals = H_master_vals;

    // temp_test = D_master_vals;
    // std::cout << "temp_test\n" ;
    // for (int i = (NcNrNa) - 10; i < NcNrNa; i++){
    //     std::cout << i << "---\n";
    //     std::cout << temp_test[i] << "\n" ;
    //     std::cout << H_master_vals[i] << "\n" ;
    //     std::cout << D_master_vals[i] << "\n " ;
    // }
    // std::cout << "arr_size = " << arr_size << "\n";
    // int temp_var = (NcNrNa) - 1;
    // std::cout << "\n temp_test last term = " << temp_test[temp_var] << "\n";
    // std::cout << "\n H_master_vals last term = " << H_master_vals[temp_var] << "\n";
    // std::cout << "\n D_master_vals last term = " << D_master_vals[temp_var] << "\n";
    // print_array(&temp_test[0], 8000, "temp_test", " ");
    // print_array<float>(&H_master_vals[0], H_master_vals.size(), "H_master_vals", " ");


    // copy data from host to device
    D_params = H_params;
    D_xs = H_xs;
    D_ys = H_ys;


    // run time loop and compute transition data for each time step
    auto start = high_resolution_clock::now(); 
    auto end = high_resolution_clock::now(); 
    auto duration_t = duration_cast<microseconds>(end - start);
    //IMP: Run time loop till nt-1. There ar no S2s to S1s in the last timestep
    for(int t = 0; t < nt-1; t++){
        std::cout << "*** Computing data for timestep, T = " << t << std::endl;
        D_tdummy[0] = t;

        start = high_resolution_clock::now(); 
            // this function also concats coos across time.
            build_sparse_transition_model_at_T(t, bDimx, D_tdummy, D_all_u_arr, D_all_v_arr 
                                                ,D_all_ui_arr, D_all_vi_arr, D_all_yi_arr,
                                                D_params, D_xs, D_ys, H_ac_angles, D_master_vals,
                                                H_coo_len_per_ac,
                                                H_Aarr_of_cooS1, H_Aarr_of_cooS2, H_Aarr_of_cooProb,
                                                H_Aarr_of_Rs);
                                                //  output_data )
        end = high_resolution_clock::now(); 
        std::cout << std::endl ;
        duration_t = duration_cast<microseconds>(end - start);
        std::cout << "duration@t = "<< duration_t.count()/1e6 << std::endl;
        std::cout << std::endl << std::endl;
    }


    //fill R vectors of each action for the last time step with high negative values. 
    // this has to be done seaprately because the above loop runs till nt-1.
    thrust::host_vector<float> H_rewards_at_end_t(ncells, 0);
    for (int i =0; i < num_actions; i++){
        H_Aarr_of_Rs[i].insert(H_Aarr_of_Rs[i].end(), H_rewards_at_end_t.begin(), H_rewards_at_end_t.end());
    }
    for (int i =0; i < num_actions; i++)
        std::cout << H_Aarr_of_Rs[i].size() << " ";
    

    // find nnz per action
    thrust::host_vector<long long int> H_master_PrSum_nnz_per_ac(num_actions);
    long long int DP_relv_params[2] = {ncells*nt, num_actions*1LL};
    
    long long int master_nnz = 0;
    for(int i = 0; i < num_actions; i++){
        master_nnz += H_Aarr_of_cooS1[i].size();
        H_master_PrSum_nnz_per_ac[i] = master_nnz;
    }
       
    print_array<long long int>(DP_relv_params, 2, "DP_relv_params", " ");
    unsigned long int num_DP_params = sizeof(DP_relv_params) / sizeof(DP_relv_params[0]);
    std::cout << "chek num = " << sizeof(DP_relv_params) << std::endl;
    std::cout << "chek denom = " << sizeof(DP_relv_params[0]) << std::endl;

    //checks
    std::cout << "master_nnz = " << master_nnz << std::endl;
    std::cout << "H_Aarr_of_cooS1[i].size()" << std::endl;
    for(int i = 0; i < num_actions; i++)
        std::cout << H_Aarr_of_cooS1[i].size() << std::endl;
    print_array<long long int>(&H_Aarr_of_cooS2[0][0], 10,  "H_Aarr_of_cooS2[0]", " ");


    // save final coo data
    thrust::host_vector<long long int> H_master_cooS1(master_nnz);
    thrust::host_vector<long long int> H_master_cooS2(master_nnz);
    thrust::host_vector<float> H_master_cooVal(master_nnz);
    thrust::host_vector<float> H_master_R(ncells*nt*num_actions, -99999); 
    save_master_Coos_to_file(op_FnamePfx, num_actions,
                                H_master_cooS1, 
                                H_master_cooS2, 
                                H_master_cooVal,
                                H_master_R,
                                H_Aarr_of_cooS1,
                                H_Aarr_of_cooS2,
                                H_Aarr_of_cooProb,
                                H_Aarr_of_Rs,
                                H_params,
                                DP_relv_params,
                                num_DP_params);


    return 0;
}

//------------ main ends here ------------------------------------------

void save_master_Coos_to_file(std::string op_FnamePfx, int num_actions,
    thrust::host_vector<long long int> &H_master_cooS1, 
    thrust::host_vector<long long int> &H_master_cooS2, 
    thrust::host_vector<float> &H_master_cooVal,
    thrust::host_vector<float> &H_master_R,
    thrust::host_vector<long long int>* H_Aarr_of_cooS1,
    thrust::host_vector<long long int>* H_Aarr_of_cooS2,
    thrust::host_vector<float>* H_Aarr_of_cooProb,
    thrust::host_vector<float>* H_Aarr_of_Rs,
    thrust::host_vector<float> &prob_params,
    long long int* DP_relv_params,
    unsigned long int num_DP_params
    ){
    //  Convertes floats to int32 for COO row and col idxs
    //  copies from each action vector to a master vector
    //  master_coo vectors is concatation first across time, then across action
    //  ALSO, MODIFIES S1(t,i,j) to S1(t,i,j,a)

    unsigned long long int master_nnz = H_master_cooS1.size();
    unsigned long long int prob_params_size = prob_params.size();
    int m_idx = 0;
    int n_states = DP_relv_params[0];

    std::cout << "in save \n" ;

    for(int i = 0; i < num_actions; i++){
        for(int j = 0; j< H_Aarr_of_cooS1[i].size(); j++){
            // TODO: modify to include actions
            H_master_cooS1[m_idx] = H_Aarr_of_cooS1[i][j] + i*n_states;
            m_idx++;
        }
    }

    std::cout << "in save \n" ;
    m_idx = 0;
    for(int i = 0; i < num_actions; i++){
        for(int j = 0; j< H_Aarr_of_cooS2[i].size(); j++){
            H_master_cooS2[m_idx] = H_Aarr_of_cooS2[i][j];
            m_idx++;
        }
    }

    std::cout << "in save \n" ;
    m_idx = 0;
    for(int i = 0; i < num_actions; i++){
        for(int j = 0; j< H_Aarr_of_cooProb[i].size(); j++){
            H_master_cooVal[m_idx] = H_Aarr_of_cooProb[i][j];
            m_idx++;
        }
    }

    std::cout << "in save \n" ;
    m_idx = 0;
    for(int i = 0; i < num_actions; i++){
        for(int j = 0; j< H_Aarr_of_Rs[i].size(); j++){
            H_master_R[m_idx] = H_Aarr_of_Rs[i][j];
            m_idx++;
        }
    }

    
    std::cout << "check num_DP_params = " << num_DP_params << std::endl;

    cnpy::npy_save(op_FnamePfx + "master_cooS1.npy", &H_master_cooS1[0], {master_nnz,1},"w");
    cnpy::npy_save(op_FnamePfx + "master_cooS2.npy", &H_master_cooS2[0], {master_nnz,1},"w");
    cnpy::npy_save(op_FnamePfx + "master_cooVal.npy", &H_master_cooVal[0], {master_nnz,1},"w");
    cnpy::npy_save(op_FnamePfx + "master_R.npy", &H_master_R[0], {H_master_R.size(),1},"w");
    cnpy::npy_save(op_FnamePfx + "DP_relv_params.npy", &DP_relv_params[0], {num_DP_params,1},"w");
    cnpy::npy_save(op_FnamePfx + "prob_params.npy", &prob_params[0], {prob_params_size,1},"w");

}



cnpy::NpyArray read_velocity_field_data( std::string file_path_name, int* n_elements){
    // reads numpy file from input and 
    // returns cnpy::NpyArray stucture  and also fills in num_elements in the passed reference n_elements
    // extraction in main: float* vel_data = arr.data<float>();
    // TODO: make it general. currently hard-coded for float arrays.

    //print filename
    std::cout << "file path and name:   " << file_path_name << std::endl;
    cnpy::NpyArray arr = cnpy::npy_load(file_path_name);

    //prints for checks 
    int dim = arr.shape.size();
    int num_elements = 1;
    std::cout << "shape: " ;
    for (int i = 0; i < dim; i++){
        std::cout << arr.shape[i] << " , " ;
        num_elements = num_elements*arr.shape[i];
    }
    *n_elements = num_elements;
    std::cout << std::endl << "num_elements: " << num_elements << std::endl;

    float* vel_data = arr.data<float>();
    // print check first 10 elements
    std::cout << "First 10 elements of loaded array are: " << std::endl;
    for (int i = 0; i < 10; i++)
         std::cout << vel_data[i] << "  " ;
    
    std::cout << std::endl << std::endl;

    return arr;

}

void define_xs_or_ys(float* xs, float dx, float x0, int gsize){

    for(int i = 0; i < gsize;  i++)
        xs[i] = x0 + i*dx;
}


void populate_ac_angles(float* ac_angles, int num_actions){
    //fills array with equally spaced angles in radians
    for (int i = 0; i < num_actions; i++)
        ac_angles[i] = i*(2*M_PI)/num_actions;

}

// -L/usr/local/ -lcnpy -lz --std=c++11 