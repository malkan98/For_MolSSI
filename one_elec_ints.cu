/**
 * @file one_elec_ints.cu
 * @author Melisa Alkan
 * @brief Contains one electron integral implementations
 * @version 0.1
 * @date 2021-01-12
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#ifndef BLOCKDIM
#define BLOCKDIM 128
#endif

// These constants will fit in constant memory
extern __constant__ double inv_scal_fact[25];
extern __constant__ double factm[25];
extern __constant__ double boys_asymp_val[25];

// The following constants will not fit within constant memory
// therefore we use static global memory
extern __device__ double exp_cheby_coefs[30001 * 4];
extern __device__ double cheby_coefs[30001 * 25 * 4];

__forceinline__ __device__ double
boysf_interpolation_nuc_attr(const unsigned int M, const double T,
                             const double *__restrict__ chebyshev_coefs,
                             const double *__restrict__ inv_scaling_factor) {

  const unsigned int inv_inter = 1000;

  const unsigned int grid_position_T0 = (unsigned int)(T * inv_inter);

  // std::cout << "grid_position_T0 = " << grid_position_T0-1 << std::endl;
  // Coefficients for interpolation
  const double a0 = chebyshev_coefs[0 + 4 * M + 25 * 4 * grid_position_T0];
  const double a1 = chebyshev_coefs[1 + 4 * M + 25 * 4 * grid_position_T0];
  const double a2 = chebyshev_coefs[2 + 4 * M + 25 * 4 * grid_position_T0];
  const double a3 = chebyshev_coefs[3 + 4 * M + 25 * 4 * grid_position_T0];

  // Inverse of scaling factor
  const double deltam1 = inv_scaling_factor[M];
  const double Tp = ((double)grid_position_T0) / 1000;
  const double t = (T - Tp) * deltam1;

  // const double t = fmod(T,inter)*deltam1;

  // Chebyshev interpolation
  double pow_t = t * t;
  const double f1 = a1 * t;
  const double f2 = a2 * (2.0 * pow_t - 1.0);
  pow_t *= t;
  const double f3 = a3 * (4.0 * pow_t - 3.0 * t);

  return a0 + f1 + f2 + f3;
}

__forceinline__ __device__ double
boysf_asymptotic_nuc_attr(const unsigned int M, const double T,
                          const double *__restrict__ boys_asymp_value) {

  const double asympt = boys_asymp_value[M];
  double pow_t = 1 / T;
  double pow_t_tmp = pow_t;
  const double sqrt_t = asympt * sqrt(pow_t);
  for (unsigned int m = 2; m <= M; ++m) {
    pow_t *= pow_t_tmp;
  }
  const double abf_value = (M == 0) ? sqrt_t : (pow_t * sqrt_t);

  return abf_value;
}

__forceinline__ __device__ double
boysf_nuc_attr(const unsigned int M, const double T,
               const double *__restrict__ chebyshev_coefs,
               const double *__restrict__ inv_scaling_factor,
               const double *__restrict__ boys_asymp_value) {

  if (T <= 30.0) {
    return boysf_interpolation_nuc_attr(M, T, chebyshev_coefs,
                                        inv_scaling_factor);
  } else {
    return boysf_asymptotic_nuc_attr(M, T, boys_asymp_value);
  }
}

__forceinline__ __device__ double
expmt_nuc_attr(const double T, const double *__restrict__ exp_chebyshev_coefs) {

  if (T > 30.0) {
    return 0.0;
  } else {
    const unsigned int inv_inter = 1000;
    const double inter = 0.001;

    const unsigned int grid_position_T0 = (unsigned int)(T * inv_inter);

    // Coefficients for interpolation
    const double a0 = exp_chebyshev_coefs[0 + 4 * grid_position_T0];
    const double a1 = exp_chebyshev_coefs[1 + 4 * grid_position_T0];
    const double a2 = exp_chebyshev_coefs[2 + 4 * grid_position_T0];
    const double a3 = exp_chebyshev_coefs[3 + 4 * grid_position_T0];
    // Inverse of scaling factor
    const double deltam1 = 268.64248295588547989;

    const double t = fmod(T, inter) * deltam1;

    // Chebyshev interpolation
    double pow_t = t * t;
    const double f1 = a1 * t;
    const double f2 = a2 * (2.0 * pow_t - 1.0);
    pow_t *= t;
    const double f3 = a3 * (4.0 * pow_t - 3.0 * t);

    return a0 + f1 + f2 + f3;
  }
}


__global__ void nuc_attr_GPU_kernel_2_1(
    const unsigned int *__restrict__ Kab_gl, const double *__restrict__ Ax_gl,
    const double *__restrict__ Ay_gl, const double *__restrict__ Az_gl,
    const double *__restrict__ ABx_gl, const double *__restrict__ ABy_gl,
    const double *__restrict__ ABz_gl, const double *__restrict__ P_gl,
    const double *__restrict__ R_gl, const double *__restrict__ zeta_gl,
    const unsigned int n_atoms, const double *__restrict__ Z_gl,
    const double *__restrict__ UP_gl, const double *__restrict__ fz_gl,
    const unsigned int *offsets_sha, const unsigned int *offsets_shb,
    const unsigned int n_ab, const unsigned int nbas,
    double *__restrict__ Hcore_gl) {

  //---Execute within correct boundaries---//
  const unsigned int gThrIdx = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ double Hcore_block_sh[BLOCKDIM * 18];
  //__syncthreads();
  Hcore_block_sh[threadIdx.x + 0] = 0;
  Hcore_block_sh[threadIdx.x + 128] = 0;
  Hcore_block_sh[threadIdx.x + 256] = 0;
  Hcore_block_sh[threadIdx.x + 384] = 0;
  Hcore_block_sh[threadIdx.x + 512] = 0;
  Hcore_block_sh[threadIdx.x + 640] = 0;
  Hcore_block_sh[threadIdx.x + 768] = 0;
  Hcore_block_sh[threadIdx.x + 896] = 0;
  Hcore_block_sh[threadIdx.x + 1024] = 0;
  Hcore_block_sh[threadIdx.x + 1152] = 0;
  Hcore_block_sh[threadIdx.x + 1280] = 0;
  Hcore_block_sh[threadIdx.x + 1408] = 0;
  Hcore_block_sh[threadIdx.x + 1536] = 0;
  Hcore_block_sh[threadIdx.x + 1664] = 0;
  Hcore_block_sh[threadIdx.x + 1792] = 0;
  Hcore_block_sh[threadIdx.x + 1920] = 0;
  Hcore_block_sh[threadIdx.x + 2048] = 0;
  Hcore_block_sh[threadIdx.x + 2176] = 0;

  if (gThrIdx < n_ab) {

    //---Initialize useful arrays and constants---//
    const double fm[4] = {factm[0], factm[1], factm[2], factm[3]};
    double Zero_m[4] = {0};
    const unsigned int Kab = Kab_gl[gThrIdx];
    const double A[3] = {Ax_gl[gThrIdx], Ay_gl[gThrIdx], Az_gl[gThrIdx]};
    const double AB[3] = {ABx_gl[gThrIdx], ABy_gl[gThrIdx], ABz_gl[gThrIdx]};
    const unsigned int mu_0 = offsets_sha[gThrIdx];
    const unsigned int nu_0 = offsets_shb[gThrIdx];
    //---Initialization of VRR targets---//
    double ssf_sss_0_t = 0;
    double spd_sss_0_t = 0;
    double sdp_sss_0_t = 0;
    double sfs_sss_0_t = 0;
    double psd_sss_0_t = 0;
    double ppp_sss_0_t = 0;
    double pds_sss_0_t = 0;
    double dsp_sss_0_t = 0;
    double dps_sss_0_t = 0;
    double fss_sss_0_t = 0;
    double ssd_sss_0_t = 0;
    double spp_sss_0_t = 0;
    double sds_sss_0_t = 0;
    double psp_sss_0_t = 0;
    double pps_sss_0_t = 0;
    double dss_sss_0_t = 0;

    //---VRRs and contraction---//
    for (unsigned int kab = 0; kab < Kab; ++kab) {
      const double UP = UP_gl[gThrIdx + kab * n_ab];
      const double zeta = zeta_gl[gThrIdx + kab * n_ab];
      const double fz = fz_gl[gThrIdx + kab * n_ab];
      const double P[3] = {P_gl[gThrIdx + kab * n_ab],
                           P_gl[gThrIdx + kab * n_ab + Kab * n_ab],
                           P_gl[gThrIdx + kab * n_ab + 2 * Kab * n_ab]};
      const double PA[3] = {P[0] - A[0], P[1] - A[1], P[2] - A[2]};
      const double theta2 = 0.5 / (fz);
      // const double theta =
      // sqrt(theta2)*UP*pow((M_PI/zeta),(1.5));//*(1/(pow(sqrt(2*fz),3)*5.9149671727956128778));
      const double theta = sqrt(theta2) * UP * 1.0622519320271969144771;
      for (unsigned int M = 0; M < n_atoms; ++M) {
        double RP[3] = {R_gl[3 * M] - P[0], R_gl[3 * M + 1] - P[1],
                        R_gl[3 * M + 2] - P[2]};
        const double R2 = (RP[0] * RP[0]) + (RP[1] * RP[1]) + (RP[2] * RP[2]);
        const double T = zeta * R2;
        const double expt = expmt_nuc_attr(T, exp_cheby_coefs);
        Zero_m[3] =
            boysf_nuc_attr(3, T, cheby_coefs, inv_scal_fact, boys_asymp_val);
        Zero_m[2] = fm[2] * (expt + 2 * T * Zero_m[3]);
        Zero_m[1] = fm[1] * (expt + 2 * T * Zero_m[2]);
        Zero_m[0] = fm[0] * (expt + 2 * T * Zero_m[1]);
        Zero_m[0] *= theta;
        Zero_m[1] *= theta;
        Zero_m[2] *= theta;
        Zero_m[3] *= theta;
        const double pss_sss_2 = Zero_m[2] * PA[0] + Zero_m[3] * RP[0];
        const double pss_sss_1 = Zero_m[1] * PA[0] + Zero_m[2] * RP[0];
        const double pss_sss_0 = Zero_m[0] * PA[0] + Zero_m[1] * RP[0];
        const double sps_sss_2 = Zero_m[2] * PA[1] + Zero_m[3] * RP[1];
        const double sps_sss_1 = Zero_m[1] * PA[1] + Zero_m[2] * RP[1];
        const double sps_sss_0 = Zero_m[0] * PA[1] + Zero_m[1] * RP[1];
        const double ssp_sss_2 = Zero_m[2] * PA[2] + Zero_m[3] * RP[2];
        const double ssp_sss_1 = Zero_m[1] * PA[2] + Zero_m[2] * RP[2];
        const double ssp_sss_0 = Zero_m[0] * PA[2] + Zero_m[1] * RP[2];
        const double dss_sss_1 = (Zero_m[1] - Zero_m[2]) * fz +
                                 pss_sss_1 * PA[0] + pss_sss_2 * RP[0];
        const double dss_sss_0 = (Zero_m[0] - Zero_m[1]) * fz +
                                 pss_sss_0 * PA[0] + pss_sss_1 * RP[0];
        dss_sss_0_t += Z_gl[M] * dss_sss_0;
        const double pps_sss_1 = sps_sss_1 * PA[0] + sps_sss_2 * RP[0];
        const double pps_sss_0 = sps_sss_0 * PA[0] + sps_sss_1 * RP[0];
        pps_sss_0_t += Z_gl[M] * pps_sss_0;
        const double psp_sss_1 = ssp_sss_1 * PA[0] + ssp_sss_2 * RP[0];
        const double psp_sss_0 = ssp_sss_0 * PA[0] + ssp_sss_1 * RP[0];
        psp_sss_0_t += Z_gl[M] * psp_sss_0;
        const double sds_sss_1 = (Zero_m[1] - Zero_m[2]) * fz +
                                 sps_sss_1 * PA[1] + sps_sss_2 * RP[1];
        const double sds_sss_0 = (Zero_m[0] - Zero_m[1]) * fz +
                                 sps_sss_0 * PA[1] + sps_sss_1 * RP[1];
        sds_sss_0_t += Z_gl[M] * sds_sss_0;
        const double spp_sss_1 = ssp_sss_1 * PA[1] + ssp_sss_2 * RP[1];
        const double spp_sss_0 = ssp_sss_0 * PA[1] + ssp_sss_1 * RP[1];
        spp_sss_0_t += Z_gl[M] * spp_sss_0;
        const double ssd_sss_1 = (Zero_m[1] - Zero_m[2]) * fz +
                                 ssp_sss_1 * PA[2] + ssp_sss_2 * RP[2];
        const double ssd_sss_0 = (Zero_m[0] - Zero_m[1]) * fz +
                                 ssp_sss_0 * PA[2] + ssp_sss_1 * RP[2];
        ssd_sss_0_t += Z_gl[M] * ssd_sss_0;
        const double fss_sss_0 = 2 * (pss_sss_0 - pss_sss_1) * fz +
                                 dss_sss_0 * PA[0] + dss_sss_1 * RP[0];
        fss_sss_0_t += Z_gl[M] * fss_sss_0;
        const double dps_sss_0 = (sps_sss_0 - sps_sss_1) * fz +
                                 pps_sss_0 * PA[0] + pps_sss_1 * RP[0];
        dps_sss_0_t += Z_gl[M] * dps_sss_0;
        const double dsp_sss_0 = (ssp_sss_0 - ssp_sss_1) * fz +
                                 psp_sss_0 * PA[0] + psp_sss_1 * RP[0];
        dsp_sss_0_t += Z_gl[M] * dsp_sss_0;
        const double pds_sss_0 = sds_sss_0 * PA[0] + sds_sss_1 * RP[0];
        pds_sss_0_t += Z_gl[M] * pds_sss_0;
        const double ppp_sss_0 = spp_sss_0 * PA[0] + spp_sss_1 * RP[0];
        ppp_sss_0_t += Z_gl[M] * ppp_sss_0;
        const double psd_sss_0 = ssd_sss_0 * PA[0] + ssd_sss_1 * RP[0];
        psd_sss_0_t += Z_gl[M] * psd_sss_0;
        const double sfs_sss_0 = 2 * (sps_sss_0 - sps_sss_1) * fz +
                                 sds_sss_0 * PA[1] + sds_sss_1 * RP[1];
        sfs_sss_0_t += Z_gl[M] * sfs_sss_0;
        const double sdp_sss_0 = (ssp_sss_0 - ssp_sss_1) * fz +
                                 spp_sss_0 * PA[1] + spp_sss_1 * RP[1];
        sdp_sss_0_t += Z_gl[M] * sdp_sss_0;
        const double spd_sss_0 = ssd_sss_0 * PA[1] + ssd_sss_1 * RP[1];
        spd_sss_0_t += Z_gl[M] * spd_sss_0;
        const double ssf_sss_0 = 2 * (ssp_sss_0 - ssp_sss_1) * fz +
                                 ssd_sss_0 * PA[2] + ssd_sss_1 * RP[2];
        ssf_sss_0_t += Z_gl[M] * ssf_sss_0;
      } // close nuc loop
    }   // kab loop
    //---HRRs---//
    Hcore_block_sh[threadIdx.x + 0] = fss_sss_0_t + dss_sss_0_t * AB[0];
    Hcore_block_sh[threadIdx.x + 128] = dps_sss_0_t + dss_sss_0_t * AB[1];
    Hcore_block_sh[threadIdx.x + 256] = dsp_sss_0_t + dss_sss_0_t * AB[2];
    Hcore_block_sh[threadIdx.x + 384] = dps_sss_0_t + pps_sss_0_t * AB[0];
    Hcore_block_sh[threadIdx.x + 512] = pds_sss_0_t + pps_sss_0_t * AB[1];
    Hcore_block_sh[threadIdx.x + 640] = ppp_sss_0_t + pps_sss_0_t * AB[2];
    Hcore_block_sh[threadIdx.x + 768] = dsp_sss_0_t + psp_sss_0_t * AB[0];
    Hcore_block_sh[threadIdx.x + 896] = ppp_sss_0_t + psp_sss_0_t * AB[1];
    Hcore_block_sh[threadIdx.x + 1024] = psd_sss_0_t + psp_sss_0_t * AB[2];
    Hcore_block_sh[threadIdx.x + 1152] = pds_sss_0_t + sds_sss_0_t * AB[0];
    Hcore_block_sh[threadIdx.x + 1280] = sfs_sss_0_t + sds_sss_0_t * AB[1];
    Hcore_block_sh[threadIdx.x + 1408] = sdp_sss_0_t + sds_sss_0_t * AB[2];
    Hcore_block_sh[threadIdx.x + 1536] = ppp_sss_0_t + spp_sss_0_t * AB[0];
    Hcore_block_sh[threadIdx.x + 1664] = sdp_sss_0_t + spp_sss_0_t * AB[1];
    Hcore_block_sh[threadIdx.x + 1792] = spd_sss_0_t + spp_sss_0_t * AB[2];
    Hcore_block_sh[threadIdx.x + 1920] = psd_sss_0_t + ssd_sss_0_t * AB[0];
    Hcore_block_sh[threadIdx.x + 2048] = spd_sss_0_t + ssd_sss_0_t * AB[1];
    Hcore_block_sh[threadIdx.x + 2176] = ssf_sss_0_t + ssd_sss_0_t * AB[2];
#pragma unroll
    for (unsigned dmu = 0; dmu < 6; ++dmu) {
      unsigned int mu = mu_0 + dmu;
#pragma unroll
      for (unsigned dnu = 0; dnu < 3; ++dnu) {
        unsigned int nu = nu_0 + dnu;
        // const double correct_fact = (nu == mu) ? 0.5 : 1.0;
        Hcore_gl[mu + nbas * nu] -=
            Hcore_block_sh[threadIdx.x + BLOCKDIM * (dnu + 3 * dmu)];
        Hcore_gl[nu + nbas * mu] -=
            Hcore_block_sh[threadIdx.x + BLOCKDIM * (dnu + 3 * dmu)];
      }
    }

  } // if block
}

extern "C" void nuc_attr_GPU_class_2_1(
    const unsigned int *__restrict__ Kab_gl, const double *__restrict__ Ax_gl,
    const double *__restrict__ Ay_gl, const double *__restrict__ Az_gl,
    const double *__restrict__ ABx_gl, const double *__restrict__ ABy_gl,
    const double *__restrict__ ABz_gl, const double *__restrict__ P_gl,
    const double *__restrict__ R_gl, const double *__restrict__ zeta_gl,
    const unsigned int n_atoms, const double *__restrict__ Z_gl,
    const double *__restrict__ UP_gl, const double *__restrict__ fz_gl,
    const unsigned int *offsets_sha, const unsigned int *offsets_shb,
    const unsigned int n_ab, const unsigned int nbas,
    double *__restrict__ Hcore_gl, const unsigned int block_dim,
    cudaStream_t istream) {

  const float t_nab_d = ceil(((double)(n_ab) / block_dim));
  const unsigned int n_blocks = (unsigned int)t_nab_d;
  dim3 block_d(block_dim);
  dim3 grid_d(n_blocks);
  nuc_attr_GPU_kernel_2_1<<<grid_d, block_d, 0, istream>>>(
      Kab_gl, Ax_gl, Ay_gl, Az_gl, ABx_gl, ABy_gl, ABz_gl, P_gl, R_gl, zeta_gl,
      n_atoms, Z_gl, UP_gl, fz_gl, offsets_sha, offsets_shb, n_ab, nbas,
      Hcore_gl);
  return;
}

__global__ void kin_ov_GPU_kernel_2_2(
    const unsigned int *__restrict__ Kab_gl, const double *__restrict__ Ax_gl,
    const double *__restrict__ Ay_gl, const double *__restrict__ Az_gl,
    const double *__restrict__ ABx_gl, const double *__restrict__ ABy_gl,
    const double *__restrict__ ABz_gl, const double *__restrict__ P_gl,
    const double *__restrict__ zeta_gl, const double *__restrict__ beta_gl,
    const double *__restrict__ UP_gl, const double *__restrict__ fz_gl,
    const unsigned int *offsets_sha, const unsigned int *offsets_shb,
    const unsigned int n_ab, const unsigned int nbas,
    double *__restrict__ Hcore_gl, double *__restrict__ S_gl) {

  //---Execute within correct boundaries---//
  const unsigned int gThrIdx = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ double Hcore_block_sh[BLOCKDIM * 36];
  Hcore_block_sh[threadIdx.x + 0] = 0;
  Hcore_block_sh[threadIdx.x + 128] = 0;
  Hcore_block_sh[threadIdx.x + 256] = 0;
  Hcore_block_sh[threadIdx.x + 384] = 0;
  Hcore_block_sh[threadIdx.x + 512] = 0;
  Hcore_block_sh[threadIdx.x + 640] = 0;
  Hcore_block_sh[threadIdx.x + 768] = 0;
  Hcore_block_sh[threadIdx.x + 896] = 0;
  Hcore_block_sh[threadIdx.x + 1024] = 0;
  Hcore_block_sh[threadIdx.x + 1152] = 0;
  Hcore_block_sh[threadIdx.x + 1280] = 0;
  Hcore_block_sh[threadIdx.x + 1408] = 0;
  Hcore_block_sh[threadIdx.x + 1536] = 0;
  Hcore_block_sh[threadIdx.x + 1664] = 0;
  Hcore_block_sh[threadIdx.x + 1792] = 0;
  Hcore_block_sh[threadIdx.x + 1920] = 0;
  Hcore_block_sh[threadIdx.x + 2048] = 0;
  Hcore_block_sh[threadIdx.x + 2176] = 0;
  Hcore_block_sh[threadIdx.x + 2304] = 0;
  Hcore_block_sh[threadIdx.x + 2432] = 0;
  Hcore_block_sh[threadIdx.x + 2560] = 0;
  Hcore_block_sh[threadIdx.x + 2688] = 0;
  Hcore_block_sh[threadIdx.x + 2816] = 0;
  Hcore_block_sh[threadIdx.x + 2944] = 0;
  Hcore_block_sh[threadIdx.x + 3072] = 0;
  Hcore_block_sh[threadIdx.x + 3200] = 0;
  Hcore_block_sh[threadIdx.x + 3328] = 0;
  Hcore_block_sh[threadIdx.x + 3456] = 0;
  Hcore_block_sh[threadIdx.x + 3584] = 0;
  Hcore_block_sh[threadIdx.x + 3712] = 0;
  Hcore_block_sh[threadIdx.x + 3840] = 0;
  Hcore_block_sh[threadIdx.x + 3968] = 0;
  Hcore_block_sh[threadIdx.x + 4096] = 0;
  Hcore_block_sh[threadIdx.x + 4224] = 0;
  Hcore_block_sh[threadIdx.x + 4352] = 0;
  Hcore_block_sh[threadIdx.x + 4480] = 0;
  //__syncthreads();
  if (gThrIdx < n_ab) {

    //---Initialize useful arrays and constants---//
    double Zero_m[1] = {0};
    const unsigned int Kab = Kab_gl[gThrIdx];
    const double A[3] = {Ax_gl[gThrIdx], Ay_gl[gThrIdx], Az_gl[gThrIdx]};
    const double AB[3] = {ABx_gl[gThrIdx], ABy_gl[gThrIdx], ABz_gl[gThrIdx]};
    const unsigned int mu_0 = offsets_sha[gThrIdx];
    const unsigned int nu_0 = offsets_shb[gThrIdx];

    //---VRRs and contraction---//
    for (unsigned int kab = 0; kab < Kab; ++kab) {
      const double UP = UP_gl[gThrIdx + kab * n_ab];
      const double beta = beta_gl[gThrIdx + kab * n_ab];
      const double fz = fz_gl[gThrIdx + kab * n_ab];
      const double P[3] = {P_gl[gThrIdx + kab * n_ab],
                           P_gl[gThrIdx + kab * n_ab + Kab * n_ab],
                           P_gl[gThrIdx + kab * n_ab + 2 * Kab * n_ab]};
      const double PA[3] = {P[0] - A[0], P[1] - A[1], P[2] - A[2]};
      const double PB[3] = {P[0] - (A[0] - AB[0]), P[1] - (A[1] - AB[1]),
                            P[2] - (A[2] - AB[2])};
      // Zero_m[0] =
      // UP*pow((M_PI/zeta),(1.5))*(1/(pow(sqrt(2*fz),3)*5.9149671727956128778));
      Zero_m[0] = UP * 0.9413962637767148126260;
      const double pss_sss_0 = (Zero_m[0] * PA[0]);
      const double sps_sss_0 = (Zero_m[0] * PA[1]);
      const double ssp_sss_0 = (Zero_m[0] * PA[2]);
      const double sss_pss_0 = (Zero_m[0] * PB[0]);
      const double sss_sps_0 = (Zero_m[0] * PB[1]);
      const double sss_ssp_0 = (Zero_m[0] * PB[2]);
      const double dss_sss_0 = (Zero_m[0] * fz + pss_sss_0 * PA[0]);
      const double pps_sss_0 = (sps_sss_0 * PA[0]);
      const double psp_sss_0 = (ssp_sss_0 * PA[0]);
      const double pss_pss_0 = (Zero_m[0] * fz + pss_sss_0 * PB[0]);
      const double pss_sps_0 = (pss_sss_0 * PB[1]);
      const double pss_ssp_0 = (pss_sss_0 * PB[2]);
      const double sds_sss_0 = (Zero_m[0] * fz + sps_sss_0 * PA[1]);
      const double spp_sss_0 = (ssp_sss_0 * PA[1]);
      const double sps_pss_0 = (sps_sss_0 * PB[0]);
      const double sps_sps_0 = (Zero_m[0] * fz + sps_sss_0 * PB[1]);
      const double sps_ssp_0 = (sps_sss_0 * PB[2]);
      const double ssd_sss_0 = (Zero_m[0] * fz + ssp_sss_0 * PA[2]);
      const double ssp_pss_0 = (ssp_sss_0 * PB[0]);
      const double ssp_sps_0 = (ssp_sss_0 * PB[1]);
      const double ssp_ssp_0 = (Zero_m[0] * fz + ssp_sss_0 * PB[2]);
      const double sss_dss_0 = (Zero_m[0] * fz + sss_pss_0 * PB[0]);
      const double sss_pps_0 = (sss_sps_0 * PB[0]);
      const double sss_psp_0 = (sss_ssp_0 * PB[0]);
      const double sss_sds_0 = (Zero_m[0] * fz + sss_sps_0 * PB[1]);
      const double sss_spp_0 = (sss_ssp_0 * PB[1]);
      const double sss_ssd_0 = (Zero_m[0] * fz + sss_ssp_0 * PB[2]);
      const double dss_pss_0 = (2 * pss_sss_0 * fz + dss_sss_0 * PB[0]);
      const double dss_sps_0 = (dss_sss_0 * PB[1]);
      const double dss_ssp_0 = (dss_sss_0 * PB[2]);
      const double pps_pss_0 = (sps_sss_0 * fz + pps_sss_0 * PB[0]);
      const double pps_sps_0 = (pss_sss_0 * fz + pps_sss_0 * PB[1]);
      const double pps_ssp_0 = (pps_sss_0 * PB[2]);
      const double psp_pss_0 = (ssp_sss_0 * fz + psp_sss_0 * PB[0]);
      const double psp_sps_0 = (psp_sss_0 * PB[1]);
      const double psp_ssp_0 = (pss_sss_0 * fz + psp_sss_0 * PB[2]);
      const double pss_dss_0 =
          (pss_sss_0 * fz + sss_pss_0 * fz + pss_pss_0 * PB[0]);
      const double pss_pps_0 = (sss_sps_0 * fz + pss_sps_0 * PB[0]);
      const double pss_psp_0 = (sss_ssp_0 * fz + pss_ssp_0 * PB[0]);
      const double pss_sds_0 = (pss_sss_0 * fz + pss_sps_0 * PB[1]);
      const double pss_spp_0 = (pss_ssp_0 * PB[1]);
      const double pss_ssd_0 = (pss_sss_0 * fz + pss_ssp_0 * PB[2]);
      const double sds_pss_0 = (sds_sss_0 * PB[0]);
      const double sds_sps_0 = (2 * sps_sss_0 * fz + sds_sss_0 * PB[1]);
      const double sds_ssp_0 = (sds_sss_0 * PB[2]);
      const double spp_pss_0 = (spp_sss_0 * PB[0]);
      const double spp_sps_0 = (ssp_sss_0 * fz + spp_sss_0 * PB[1]);
      const double spp_ssp_0 = (sps_sss_0 * fz + spp_sss_0 * PB[2]);
      const double sps_dss_0 = (sps_sss_0 * fz + sps_pss_0 * PB[0]);
      const double sps_pps_0 = (sps_sps_0 * PB[0]);
      const double sps_psp_0 = (sps_ssp_0 * PB[0]);
      const double sps_sds_0 =
          (sps_sss_0 * fz + sss_sps_0 * fz + sps_sps_0 * PB[1]);
      const double sps_spp_0 = (sss_ssp_0 * fz + sps_ssp_0 * PB[1]);
      const double sps_ssd_0 = (sps_sss_0 * fz + sps_ssp_0 * PB[2]);
      const double ssd_pss_0 = (ssd_sss_0 * PB[0]);
      const double ssd_sps_0 = (ssd_sss_0 * PB[1]);
      const double ssd_ssp_0 = (2 * ssp_sss_0 * fz + ssd_sss_0 * PB[2]);
      const double ssp_dss_0 = (ssp_sss_0 * fz + ssp_pss_0 * PB[0]);
      const double ssp_pps_0 = (ssp_sps_0 * PB[0]);
      const double ssp_psp_0 = (ssp_ssp_0 * PB[0]);
      const double ssp_sds_0 = (ssp_sss_0 * fz + ssp_sps_0 * PB[1]);
      const double ssp_spp_0 = (ssp_ssp_0 * PB[1]);
      const double ssp_ssd_0 =
          (ssp_sss_0 * fz + sss_ssp_0 * fz + ssp_ssp_0 * PB[2]);
      const double dss_dss_0 =
          (dss_sss_0 * fz + 2 * pss_pss_0 * fz + dss_pss_0 * PB[0]);
      S_gl[mu_0 + 0 + nbas * (nu_0 + 0)] += (dss_dss_0);
      // S_gl[nu_0 + 0 + nbas *(mu_0 + 0)] += (dss_dss_0);
      const double dss_pps_0 = (2 * pss_sps_0 * fz + dss_sps_0 * PB[0]);
      S_gl[mu_0 + 0 + nbas * (nu_0 + 1)] += (dss_pps_0);
      // S_gl[nu_0 + 1 + nbas *(mu_0 + 0)] += (dss_pps_0);
      const double dss_psp_0 = (2 * pss_ssp_0 * fz + dss_ssp_0 * PB[0]);
      S_gl[mu_0 + 0 + nbas * (nu_0 + 2)] += (dss_psp_0);
      // S_gl[nu_0 + 2 + nbas *(mu_0 + 0)] += (dss_psp_0);
      const double dss_sds_0 = (dss_sss_0 * fz + dss_sps_0 * PB[1]);
      S_gl[mu_0 + 0 + nbas * (nu_0 + 3)] += (dss_sds_0);
      // S_gl[nu_0 + 3 + nbas *(mu_0 + 0)] += (dss_sds_0);
      const double dss_spp_0 = (dss_ssp_0 * PB[1]);
      S_gl[mu_0 + 0 + nbas * (nu_0 + 4)] += (dss_spp_0);
      // S_gl[nu_0 + 4 + nbas *(mu_0 + 0)] += (dss_spp_0);
      const double dss_ssd_0 = (dss_sss_0 * fz + dss_ssp_0 * PB[2]);
      S_gl[mu_0 + 0 + nbas * (nu_0 + 5)] += (dss_ssd_0);
      // S_gl[nu_0 + 5 + nbas *(mu_0 + 0)] += (dss_ssd_0);
      const double pps_dss_0 =
          (pps_sss_0 * fz + sps_pss_0 * fz + pps_pss_0 * PB[0]);
      S_gl[mu_0 + 1 + nbas * (nu_0 + 0)] += (pps_dss_0);
      // S_gl[nu_0 + 0 + nbas *(mu_0 + 1)] += (pps_dss_0);
      const double pps_pps_0 = (sps_sps_0 * fz + pps_sps_0 * PB[0]);
      S_gl[mu_0 + 1 + nbas * (nu_0 + 1)] += (pps_pps_0);
      // S_gl[nu_0 + 1 + nbas *(mu_0 + 1)] += (pps_pps_0);
      const double pps_psp_0 = (sps_ssp_0 * fz + pps_ssp_0 * PB[0]);
      S_gl[mu_0 + 1 + nbas * (nu_0 + 2)] += (pps_psp_0);
      // S_gl[nu_0 + 2 + nbas *(mu_0 + 1)] += (pps_psp_0);
      const double pps_sds_0 =
          (pps_sss_0 * fz + pss_sps_0 * fz + pps_sps_0 * PB[1]);
      S_gl[mu_0 + 1 + nbas * (nu_0 + 3)] += (pps_sds_0);
      // S_gl[nu_0 + 3 + nbas *(mu_0 + 1)] += (pps_sds_0);
      const double pps_spp_0 = (pss_ssp_0 * fz + pps_ssp_0 * PB[1]);
      S_gl[mu_0 + 1 + nbas * (nu_0 + 4)] += (pps_spp_0);
      // S_gl[nu_0 + 4 + nbas *(mu_0 + 1)] += (pps_spp_0);
      const double pps_ssd_0 = (pps_sss_0 * fz + pps_ssp_0 * PB[2]);
      S_gl[mu_0 + 1 + nbas * (nu_0 + 5)] += (pps_ssd_0);
      // S_gl[nu_0 + 5 + nbas *(mu_0 + 1)] += (pps_ssd_0);
      const double psp_dss_0 =
          (psp_sss_0 * fz + ssp_pss_0 * fz + psp_pss_0 * PB[0]);
      S_gl[mu_0 + 2 + nbas * (nu_0 + 0)] += (psp_dss_0);
      // S_gl[nu_0 + 0 + nbas *(mu_0 + 2)] += (psp_dss_0);
      const double psp_pps_0 = (ssp_sps_0 * fz + psp_sps_0 * PB[0]);
      S_gl[mu_0 + 2 + nbas * (nu_0 + 1)] += (psp_pps_0);
      // S_gl[nu_0 + 1 + nbas *(mu_0 + 2)] += (psp_pps_0);
      const double psp_psp_0 = (ssp_ssp_0 * fz + psp_ssp_0 * PB[0]);
      S_gl[mu_0 + 2 + nbas * (nu_0 + 2)] += (psp_psp_0);
      // S_gl[nu_0 + 2 + nbas *(mu_0 + 2)] += (psp_psp_0);
      const double psp_sds_0 = (psp_sss_0 * fz + psp_sps_0 * PB[1]);
      S_gl[mu_0 + 2 + nbas * (nu_0 + 3)] += (psp_sds_0);
      // S_gl[nu_0 + 3 + nbas *(mu_0 + 2)] += (psp_sds_0);
      const double psp_spp_0 = (psp_ssp_0 * PB[1]);
      S_gl[mu_0 + 2 + nbas * (nu_0 + 4)] += (psp_spp_0);
      // S_gl[nu_0 + 4 + nbas *(mu_0 + 2)] += (psp_spp_0);
      const double psp_ssd_0 =
          (psp_sss_0 * fz + pss_ssp_0 * fz + psp_ssp_0 * PB[2]);
      S_gl[mu_0 + 2 + nbas * (nu_0 + 5)] += (psp_ssd_0);
      // S_gl[nu_0 + 5 + nbas *(mu_0 + 2)] += (psp_ssd_0);
      const double sds_dss_0 = (sds_sss_0 * fz + sds_pss_0 * PB[0]);
      S_gl[mu_0 + 3 + nbas * (nu_0 + 0)] += (sds_dss_0);
      // S_gl[nu_0 + 0 + nbas *(mu_0 + 3)] += (sds_dss_0);
      const double sds_pps_0 = (sds_sps_0 * PB[0]);
      S_gl[mu_0 + 3 + nbas * (nu_0 + 1)] += (sds_pps_0);
      // S_gl[nu_0 + 1 + nbas *(mu_0 + 3)] += (sds_pps_0);
      const double sds_psp_0 = (sds_ssp_0 * PB[0]);
      S_gl[mu_0 + 3 + nbas * (nu_0 + 2)] += (sds_psp_0);
      // S_gl[nu_0 + 2 + nbas *(mu_0 + 3)] += (sds_psp_0);
      const double sds_sds_0 =
          (sds_sss_0 * fz + 2 * sps_sps_0 * fz + sds_sps_0 * PB[1]);
      S_gl[mu_0 + 3 + nbas * (nu_0 + 3)] += (sds_sds_0);
      // S_gl[nu_0 + 3 + nbas *(mu_0 + 3)] += (sds_sds_0);
      const double sds_spp_0 = (2 * sps_ssp_0 * fz + sds_ssp_0 * PB[1]);
      S_gl[mu_0 + 3 + nbas * (nu_0 + 4)] += (sds_spp_0);
      // S_gl[nu_0 + 4 + nbas *(mu_0 + 3)] += (sds_spp_0);
      const double sds_ssd_0 = (sds_sss_0 * fz + sds_ssp_0 * PB[2]);
      S_gl[mu_0 + 3 + nbas * (nu_0 + 5)] += (sds_ssd_0);
      // S_gl[nu_0 + 5 + nbas *(mu_0 + 3)] += (sds_ssd_0);
      const double spp_dss_0 = (spp_sss_0 * fz + spp_pss_0 * PB[0]);
      S_gl[mu_0 + 4 + nbas * (nu_0 + 0)] += (spp_dss_0);
      // S_gl[nu_0 + 0 + nbas *(mu_0 + 4)] += (spp_dss_0);
      const double spp_pps_0 = (spp_sps_0 * PB[0]);
      S_gl[mu_0 + 4 + nbas * (nu_0 + 1)] += (spp_pps_0);
      // S_gl[nu_0 + 1 + nbas *(mu_0 + 4)] += (spp_pps_0);
      const double spp_psp_0 = (spp_ssp_0 * PB[0]);
      S_gl[mu_0 + 4 + nbas * (nu_0 + 2)] += (spp_psp_0);
      // S_gl[nu_0 + 2 + nbas *(mu_0 + 4)] += (spp_psp_0);
      const double spp_sds_0 =
          (spp_sss_0 * fz + ssp_sps_0 * fz + spp_sps_0 * PB[1]);
      S_gl[mu_0 + 4 + nbas * (nu_0 + 3)] += (spp_sds_0);
      // S_gl[nu_0 + 3 + nbas *(mu_0 + 4)] += (spp_sds_0);
      const double spp_spp_0 = (ssp_ssp_0 * fz + spp_ssp_0 * PB[1]);
      S_gl[mu_0 + 4 + nbas * (nu_0 + 4)] += (spp_spp_0);
      // S_gl[nu_0 + 4 + nbas *(mu_0 + 4)] += (spp_spp_0);
      const double spp_ssd_0 =
          (spp_sss_0 * fz + sps_ssp_0 * fz + spp_ssp_0 * PB[2]);
      S_gl[mu_0 + 4 + nbas * (nu_0 + 5)] += (spp_ssd_0);
      // S_gl[nu_0 + 5 + nbas *(mu_0 + 4)] += (spp_ssd_0);
      const double ssd_dss_0 = (ssd_sss_0 * fz + ssd_pss_0 * PB[0]);
      S_gl[mu_0 + 5 + nbas * (nu_0 + 0)] += (ssd_dss_0);
      // S_gl[nu_0 + 0 + nbas *(mu_0 + 5)] += (ssd_dss_0);
      const double ssd_pps_0 = (ssd_sps_0 * PB[0]);
      S_gl[mu_0 + 5 + nbas * (nu_0 + 1)] += (ssd_pps_0);
      // S_gl[nu_0 + 1 + nbas *(mu_0 + 5)] += (ssd_pps_0);
      const double ssd_psp_0 = (ssd_ssp_0 * PB[0]);
      S_gl[mu_0 + 5 + nbas * (nu_0 + 2)] += (ssd_psp_0);
      // S_gl[nu_0 + 2 + nbas *(mu_0 + 5)] += (ssd_psp_0);
      const double ssd_sds_0 = (ssd_sss_0 * fz + ssd_sps_0 * PB[1]);
      S_gl[mu_0 + 5 + nbas * (nu_0 + 3)] += (ssd_sds_0);
      // S_gl[nu_0 + 3 + nbas *(mu_0 + 5)] += (ssd_sds_0);
      const double ssd_spp_0 = (ssd_ssp_0 * PB[1]);
      S_gl[mu_0 + 5 + nbas * (nu_0 + 4)] += (ssd_spp_0);
      // S_gl[nu_0 + 4 + nbas *(mu_0 + 5)] += (ssd_spp_0);
      const double ssd_ssd_0 =
          (ssd_sss_0 * fz + 2 * ssp_ssp_0 * fz + ssd_ssp_0 * PB[2]);
      S_gl[mu_0 + 5 + nbas * (nu_0 + 5)] += (ssd_ssd_0);
      // S_gl[nu_0 + 5 + nbas *(mu_0 + 5)] += (ssd_ssd_0);
      Hcore_block_sh[threadIdx.x + 0] +=
          -dss_sss_0 + 7 * dss_dss_0 * beta -
          4 * (2 * pss_pss_0 + sss_dss_0) * (beta * beta) * (fz * fz) -
          2 * (beta * beta) * fz *
              (5 * dss_dss_0 + 2 * (dss_pss_0 + 2 * pss_dss_0) * PB[0]) -
          2 * dss_dss_0 * (beta * beta) * (PB[0] * PB[0]) -
          2 * dss_dss_0 * (beta * beta) * (PB[1] * PB[1]) -
          2 * dss_dss_0 * (beta * beta) * (PB[2] * PB[2]);
      Hcore_block_sh[threadIdx.x + 128] +=
          -(beta * (4 * (pss_sps_0 + sss_pps_0) * beta * (fz * fz) +
                    2 * beta * fz *
                        (5 * dss_pps_0 + (dss_sps_0 + 4 * pss_pps_0) * PB[0] +
                         dss_pss_0 * PB[1]) +
                    dss_pps_0 * (-7 + 2 * beta * (PB[0] * PB[0]) +
                                 2 * beta * (PB[1] * PB[1]) +
                                 2 * beta * (PB[2] * PB[2]))));
      Hcore_block_sh[threadIdx.x + 256] +=
          -(beta * (4 * (pss_ssp_0 + sss_psp_0) * beta * (fz * fz) +
                    2 * beta * fz *
                        (5 * dss_psp_0 + (dss_ssp_0 + 4 * pss_psp_0) * PB[0] +
                         dss_pss_0 * PB[2]) +
                    dss_psp_0 * (-7 + 2 * beta * (PB[0] * PB[0]) +
                                 2 * beta * (PB[1] * PB[1]) +
                                 2 * beta * (PB[2] * PB[2]))));
      Hcore_block_sh[threadIdx.x + 384] +=
          -dss_sss_0 + 7 * dss_sds_0 * beta -
          4 * sss_sds_0 * (beta * beta) * (fz * fz) -
          2 * dss_sds_0 * (beta * beta) * (PB[0] * PB[0]) -
          2 * (beta * beta) * fz *
              (5 * dss_sds_0 + 4 * pss_sds_0 * PB[0] + 2 * dss_sps_0 * PB[1]) -
          2 * dss_sds_0 * (beta * beta) * (PB[1] * PB[1]) -
          2 * dss_sds_0 * (beta * beta) * (PB[2] * PB[2]);
      Hcore_block_sh[threadIdx.x + 512] +=
          -(beta * (4 * sss_spp_0 * beta * (fz * fz) +
                    2 * beta * fz *
                        (5 * dss_spp_0 + 4 * pss_spp_0 * PB[0] +
                         dss_ssp_0 * PB[1] + dss_sps_0 * PB[2]) +
                    dss_spp_0 * (-7 + 2 * beta * (PB[0] * PB[0]) +
                                 2 * beta * (PB[1] * PB[1]) +
                                 2 * beta * (PB[2] * PB[2]))));
      Hcore_block_sh[threadIdx.x + 640] +=
          -dss_sss_0 + 7 * dss_ssd_0 * beta -
          4 * sss_ssd_0 * (beta * beta) * (fz * fz) -
          2 * dss_ssd_0 * (beta * beta) * (PB[0] * PB[0]) -
          2 * dss_ssd_0 * (beta * beta) * (PB[1] * PB[1]) -
          2 * (beta * beta) * fz *
              (5 * dss_ssd_0 + 4 * pss_ssd_0 * PB[0] + 2 * dss_ssp_0 * PB[2]) -
          2 * dss_ssd_0 * (beta * beta) * (PB[2] * PB[2]);
      Hcore_block_sh[threadIdx.x + 768] +=
          -pps_sss_0 + 7 * pps_dss_0 * beta -
          4 * sps_pss_0 * (beta * beta) * (fz * fz) -
          2 * pps_dss_0 * (beta * beta) * (PB[0] * PB[0]) -
          2 * (beta * beta) * fz *
              (5 * pps_dss_0 + 2 * (pps_pss_0 + sps_dss_0) * PB[0] +
               2 * pss_dss_0 * PB[1]) -
          2 * pps_dss_0 * (beta * beta) * (PB[1] * PB[1]) -
          2 * pps_dss_0 * (beta * beta) * (PB[2] * PB[2]);
      Hcore_block_sh[threadIdx.x + 896] +=
          -(beta * (2 * (pss_pss_0 + sps_sps_0) * beta * (fz * fz) +
                    2 * beta * fz *
                        (5 * pps_pps_0 + (pps_sps_0 + 2 * sps_pps_0) * PB[0] +
                         (pps_pss_0 + 2 * pss_pps_0) * PB[1]) +
                    pps_pps_0 * (-7 + 2 * beta * (PB[0] * PB[0]) +
                                 2 * beta * (PB[1] * PB[1]) +
                                 2 * beta * (PB[2] * PB[2]))));
      Hcore_block_sh[threadIdx.x + 1024] +=
          -(beta * (2 * sps_ssp_0 * beta * (fz * fz) +
                    2 * beta * fz *
                        (5 * pps_psp_0 + (pps_ssp_0 + 2 * sps_psp_0) * PB[0] +
                         2 * pss_psp_0 * PB[1] + pps_pss_0 * PB[2]) +
                    pps_psp_0 * (-7 + 2 * beta * (PB[0] * PB[0]) +
                                 2 * beta * (PB[1] * PB[1]) +
                                 2 * beta * (PB[2] * PB[2]))));
      Hcore_block_sh[threadIdx.x + 1152] +=
          -pps_sss_0 + 7 * pps_sds_0 * beta -
          4 * pss_sps_0 * (beta * beta) * (fz * fz) -
          2 * pps_sds_0 * (beta * beta) * (PB[0] * PB[0]) -
          2 * (beta * beta) * fz *
              (5 * pps_sds_0 + 2 * sps_sds_0 * PB[0] +
               2 * (pps_sps_0 + pss_sds_0) * PB[1]) -
          2 * pps_sds_0 * (beta * beta) * (PB[1] * PB[1]) -
          2 * pps_sds_0 * (beta * beta) * (PB[2] * PB[2]);
      Hcore_block_sh[threadIdx.x + 1280] +=
          -(beta *
            (2 * pss_ssp_0 * beta * (fz * fz) +
             2 * beta * fz *
                 (5 * pps_spp_0 + 2 * sps_spp_0 * PB[0] +
                  (pps_ssp_0 + 2 * pss_spp_0) * PB[1] + pps_sps_0 * PB[2]) +
             pps_spp_0 *
                 (-7 + 2 * beta * (PB[0] * PB[0]) + 2 * beta * (PB[1] * PB[1]) +
                  2 * beta * (PB[2] * PB[2]))));
      Hcore_block_sh[threadIdx.x + 1408] +=
          -pps_sss_0 + 7 * pps_ssd_0 * beta -
          2 * pps_ssd_0 * (beta * beta) * (PB[0] * PB[0]) -
          2 * pps_ssd_0 * (beta * beta) * (PB[1] * PB[1]) -
          2 * (beta * beta) * fz *
              (5 * pps_ssd_0 + 2 * sps_ssd_0 * PB[0] + 2 * pss_ssd_0 * PB[1] +
               2 * pps_ssp_0 * PB[2]) -
          2 * pps_ssd_0 * (beta * beta) * (PB[2] * PB[2]);
      Hcore_block_sh[threadIdx.x + 1536] +=
          -psp_sss_0 + 7 * psp_dss_0 * beta -
          4 * ssp_pss_0 * (beta * beta) * (fz * fz) -
          2 * psp_dss_0 * (beta * beta) * (PB[0] * PB[0]) -
          2 * psp_dss_0 * (beta * beta) * (PB[1] * PB[1]) -
          2 * (beta * beta) * fz *
              (5 * psp_dss_0 + 2 * (psp_pss_0 + ssp_dss_0) * PB[0] +
               2 * pss_dss_0 * PB[2]) -
          2 * psp_dss_0 * (beta * beta) * (PB[2] * PB[2]);
      Hcore_block_sh[threadIdx.x + 1664] +=
          -(beta * (2 * ssp_sps_0 * beta * (fz * fz) +
                    2 * beta * fz *
                        (5 * psp_pps_0 + (psp_sps_0 + 2 * ssp_pps_0) * PB[0] +
                         psp_pss_0 * PB[1] + 2 * pss_pps_0 * PB[2]) +
                    psp_pps_0 * (-7 + 2 * beta * (PB[0] * PB[0]) +
                                 2 * beta * (PB[1] * PB[1]) +
                                 2 * beta * (PB[2] * PB[2]))));
      Hcore_block_sh[threadIdx.x + 1792] +=
          -(beta * (2 * (pss_pss_0 + ssp_ssp_0) * beta * (fz * fz) +
                    2 * beta * fz *
                        (5 * psp_psp_0 + (psp_ssp_0 + 2 * ssp_psp_0) * PB[0] +
                         (psp_pss_0 + 2 * pss_psp_0) * PB[2]) +
                    psp_psp_0 * (-7 + 2 * beta * (PB[0] * PB[0]) +
                                 2 * beta * (PB[1] * PB[1]) +
                                 2 * beta * (PB[2] * PB[2]))));
      Hcore_block_sh[threadIdx.x + 1920] +=
          -psp_sss_0 + 7 * psp_sds_0 * beta -
          2 * psp_sds_0 * (beta * beta) * (PB[0] * PB[0]) -
          2 * psp_sds_0 * (beta * beta) * (PB[1] * PB[1]) -
          2 * (beta * beta) * fz *
              (5 * psp_sds_0 + 2 * ssp_sds_0 * PB[0] + 2 * psp_sps_0 * PB[1] +
               2 * pss_sds_0 * PB[2]) -
          2 * psp_sds_0 * (beta * beta) * (PB[2] * PB[2]);
      Hcore_block_sh[threadIdx.x + 2048] +=
          -(beta *
            (2 * pss_sps_0 * beta * (fz * fz) +
             2 * beta * fz *
                 (5 * psp_spp_0 + 2 * ssp_spp_0 * PB[0] + psp_ssp_0 * PB[1] +
                  psp_sps_0 * PB[2] + 2 * pss_spp_0 * PB[2]) +
             psp_spp_0 *
                 (-7 + 2 * beta * (PB[0] * PB[0]) + 2 * beta * (PB[1] * PB[1]) +
                  2 * beta * (PB[2] * PB[2]))));
      Hcore_block_sh[threadIdx.x + 2176] +=
          -psp_sss_0 + 7 * psp_ssd_0 * beta -
          4 * pss_ssp_0 * (beta * beta) * (fz * fz) -
          2 * psp_ssd_0 * (beta * beta) * (PB[0] * PB[0]) -
          2 * psp_ssd_0 * (beta * beta) * (PB[1] * PB[1]) -
          2 * (beta * beta) * fz *
              (5 * psp_ssd_0 + 2 * ssp_ssd_0 * PB[0] +
               2 * (psp_ssp_0 + pss_ssd_0) * PB[2]) -
          2 * psp_ssd_0 * (beta * beta) * (PB[2] * PB[2]);
      Hcore_block_sh[threadIdx.x + 2304] +=
          -sds_sss_0 + 7 * sds_dss_0 * beta -
          4 * sss_dss_0 * (beta * beta) * (fz * fz) -
          2 * sds_dss_0 * (beta * beta) * (PB[0] * PB[0]) -
          2 * (beta * beta) * fz *
              (5 * sds_dss_0 + 2 * sds_pss_0 * PB[0] + 4 * sps_dss_0 * PB[1]) -
          2 * sds_dss_0 * (beta * beta) * (PB[1] * PB[1]) -
          2 * sds_dss_0 * (beta * beta) * (PB[2] * PB[2]);
      Hcore_block_sh[threadIdx.x + 2432] +=
          -(beta * (4 * (sps_pss_0 + sss_pps_0) * beta * (fz * fz) +
                    2 * beta * fz *
                        (5 * sds_pps_0 + sds_sps_0 * PB[0] +
                         (sds_pss_0 + 4 * sps_pps_0) * PB[1]) +
                    sds_pps_0 * (-7 + 2 * beta * (PB[0] * PB[0]) +
                                 2 * beta * (PB[1] * PB[1]) +
                                 2 * beta * (PB[2] * PB[2]))));
      Hcore_block_sh[threadIdx.x + 2560] +=
          -(beta * (4 * sss_psp_0 * beta * (fz * fz) +
                    2 * beta * fz *
                        (5 * sds_psp_0 + sds_ssp_0 * PB[0] +
                         4 * sps_psp_0 * PB[1] + sds_pss_0 * PB[2]) +
                    sds_psp_0 * (-7 + 2 * beta * (PB[0] * PB[0]) +
                                 2 * beta * (PB[1] * PB[1]) +
                                 2 * beta * (PB[2] * PB[2]))));
      Hcore_block_sh[threadIdx.x + 2688] +=
          -sds_sss_0 + 7 * sds_sds_0 * beta -
          4 * (2 * sps_sps_0 + sss_sds_0) * (beta * beta) * (fz * fz) -
          2 * sds_sds_0 * (beta * beta) * (PB[0] * PB[0]) -
          2 * (beta * beta) * fz *
              (5 * sds_sds_0 + 2 * (sds_sps_0 + 2 * sps_sds_0) * PB[1]) -
          2 * sds_sds_0 * (beta * beta) * (PB[1] * PB[1]) -
          2 * sds_sds_0 * (beta * beta) * (PB[2] * PB[2]);
      Hcore_block_sh[threadIdx.x + 2816] +=
          -(beta * (4 * (sps_ssp_0 + sss_spp_0) * beta * (fz * fz) +
                    2 * beta * fz *
                        (5 * sds_spp_0 + (sds_ssp_0 + 4 * sps_spp_0) * PB[1] +
                         sds_sps_0 * PB[2]) +
                    sds_spp_0 * (-7 + 2 * beta * (PB[0] * PB[0]) +
                                 2 * beta * (PB[1] * PB[1]) +
                                 2 * beta * (PB[2] * PB[2]))));
      Hcore_block_sh[threadIdx.x + 2944] +=
          -sds_sss_0 + 7 * sds_ssd_0 * beta -
          4 * sss_ssd_0 * (beta * beta) * (fz * fz) -
          2 * sds_ssd_0 * (beta * beta) * (PB[0] * PB[0]) -
          2 * sds_ssd_0 * (beta * beta) * (PB[1] * PB[1]) -
          2 * (beta * beta) * fz *
              (5 * sds_ssd_0 + 4 * sps_ssd_0 * PB[1] + 2 * sds_ssp_0 * PB[2]) -
          2 * sds_ssd_0 * (beta * beta) * (PB[2] * PB[2]);
      Hcore_block_sh[threadIdx.x + 3072] +=
          -spp_sss_0 + 7 * spp_dss_0 * beta -
          2 * spp_dss_0 * (beta * beta) * (PB[0] * PB[0]) -
          2 * spp_dss_0 * (beta * beta) * (PB[1] * PB[1]) -
          2 * (beta * beta) * fz *
              (5 * spp_dss_0 + 2 * spp_pss_0 * PB[0] + 2 * ssp_dss_0 * PB[1] +
               2 * sps_dss_0 * PB[2]) -
          2 * spp_dss_0 * (beta * beta) * (PB[2] * PB[2]);
      Hcore_block_sh[threadIdx.x + 3200] +=
          -(beta *
            (2 * ssp_pss_0 * beta * (fz * fz) +
             2 * beta * fz *
                 (5 * spp_pps_0 + spp_sps_0 * PB[0] +
                  (spp_pss_0 + 2 * ssp_pps_0) * PB[1] + 2 * sps_pps_0 * PB[2]) +
             spp_pps_0 *
                 (-7 + 2 * beta * (PB[0] * PB[0]) + 2 * beta * (PB[1] * PB[1]) +
                  2 * beta * (PB[2] * PB[2]))));
      Hcore_block_sh[threadIdx.x + 3328] +=
          -(beta *
            (2 * sps_pss_0 * beta * (fz * fz) +
             2 * beta * fz *
                 (5 * spp_psp_0 + spp_ssp_0 * PB[0] + 2 * ssp_psp_0 * PB[1] +
                  spp_pss_0 * PB[2] + 2 * sps_psp_0 * PB[2]) +
             spp_psp_0 *
                 (-7 + 2 * beta * (PB[0] * PB[0]) + 2 * beta * (PB[1] * PB[1]) +
                  2 * beta * (PB[2] * PB[2]))));
      Hcore_block_sh[threadIdx.x + 3456] +=
          -spp_sss_0 + 7 * spp_sds_0 * beta -
          4 * ssp_sps_0 * (beta * beta) * (fz * fz) -
          2 * spp_sds_0 * (beta * beta) * (PB[0] * PB[0]) -
          2 * spp_sds_0 * (beta * beta) * (PB[1] * PB[1]) -
          2 * (beta * beta) * fz *
              (5 * spp_sds_0 + 2 * (spp_sps_0 + ssp_sds_0) * PB[1] +
               2 * sps_sds_0 * PB[2]) -
          2 * spp_sds_0 * (beta * beta) * (PB[2] * PB[2]);
      Hcore_block_sh[threadIdx.x + 3584] +=
          -(beta * (2 * (sps_sps_0 + ssp_ssp_0) * beta * (fz * fz) +
                    2 * beta * fz *
                        (5 * spp_spp_0 + (spp_ssp_0 + 2 * ssp_spp_0) * PB[1] +
                         (spp_sps_0 + 2 * sps_spp_0) * PB[2]) +
                    spp_spp_0 * (-7 + 2 * beta * (PB[0] * PB[0]) +
                                 2 * beta * (PB[1] * PB[1]) +
                                 2 * beta * (PB[2] * PB[2]))));
      Hcore_block_sh[threadIdx.x + 3712] +=
          -spp_sss_0 + 7 * spp_ssd_0 * beta -
          4 * sps_ssp_0 * (beta * beta) * (fz * fz) -
          2 * spp_ssd_0 * (beta * beta) * (PB[0] * PB[0]) -
          2 * spp_ssd_0 * (beta * beta) * (PB[1] * PB[1]) -
          2 * (beta * beta) * fz *
              (5 * spp_ssd_0 + 2 * ssp_ssd_0 * PB[1] +
               2 * (spp_ssp_0 + sps_ssd_0) * PB[2]) -
          2 * spp_ssd_0 * (beta * beta) * (PB[2] * PB[2]);
      Hcore_block_sh[threadIdx.x + 3840] +=
          -ssd_sss_0 + 7 * ssd_dss_0 * beta -
          4 * sss_dss_0 * (beta * beta) * (fz * fz) -
          2 * ssd_dss_0 * (beta * beta) * (PB[0] * PB[0]) -
          2 * ssd_dss_0 * (beta * beta) * (PB[1] * PB[1]) -
          2 * (beta * beta) * fz *
              (5 * ssd_dss_0 + 2 * ssd_pss_0 * PB[0] + 4 * ssp_dss_0 * PB[2]) -
          2 * ssd_dss_0 * (beta * beta) * (PB[2] * PB[2]);
      Hcore_block_sh[threadIdx.x + 3968] +=
          -(beta * (4 * sss_pps_0 * beta * (fz * fz) +
                    2 * beta * fz *
                        (5 * ssd_pps_0 + ssd_sps_0 * PB[0] + ssd_pss_0 * PB[1] +
                         4 * ssp_pps_0 * PB[2]) +
                    ssd_pps_0 * (-7 + 2 * beta * (PB[0] * PB[0]) +
                                 2 * beta * (PB[1] * PB[1]) +
                                 2 * beta * (PB[2] * PB[2]))));
      Hcore_block_sh[threadIdx.x + 4096] +=
          -(beta * (4 * (ssp_pss_0 + sss_psp_0) * beta * (fz * fz) +
                    2 * beta * fz *
                        (5 * ssd_psp_0 + ssd_ssp_0 * PB[0] +
                         (ssd_pss_0 + 4 * ssp_psp_0) * PB[2]) +
                    ssd_psp_0 * (-7 + 2 * beta * (PB[0] * PB[0]) +
                                 2 * beta * (PB[1] * PB[1]) +
                                 2 * beta * (PB[2] * PB[2]))));
      Hcore_block_sh[threadIdx.x + 4224] +=
          -ssd_sss_0 + 7 * ssd_sds_0 * beta -
          4 * sss_sds_0 * (beta * beta) * (fz * fz) -
          2 * ssd_sds_0 * (beta * beta) * (PB[0] * PB[0]) -
          2 * ssd_sds_0 * (beta * beta) * (PB[1] * PB[1]) -
          2 * (beta * beta) * fz *
              (5 * ssd_sds_0 + 2 * ssd_sps_0 * PB[1] + 4 * ssp_sds_0 * PB[2]) -
          2 * ssd_sds_0 * (beta * beta) * (PB[2] * PB[2]);
      Hcore_block_sh[threadIdx.x + 4352] +=
          -(beta * (4 * (ssp_sps_0 + sss_spp_0) * beta * (fz * fz) +
                    2 * beta * fz *
                        (5 * ssd_spp_0 + ssd_ssp_0 * PB[1] +
                         (ssd_sps_0 + 4 * ssp_spp_0) * PB[2]) +
                    ssd_spp_0 * (-7 + 2 * beta * (PB[0] * PB[0]) +
                                 2 * beta * (PB[1] * PB[1]) +
                                 2 * beta * (PB[2] * PB[2]))));
      Hcore_block_sh[threadIdx.x + 4480] +=
          -ssd_sss_0 + 7 * ssd_ssd_0 * beta -
          4 * (2 * ssp_ssp_0 + sss_ssd_0) * (beta * beta) * (fz * fz) -
          2 * ssd_ssd_0 * (beta * beta) * (PB[0] * PB[0]) -
          2 * ssd_ssd_0 * (beta * beta) * (PB[1] * PB[1]) -
          2 * (beta * beta) * fz *
              (5 * ssd_ssd_0 + 2 * (ssd_ssp_0 + 2 * ssp_ssd_0) * PB[2]) -
          2 * ssd_ssd_0 * (beta * beta) * (PB[2] * PB[2]);
    } // kab loop

    const double correct_fact = (nu_0 == mu_0) ? 0.5 : 1.0;
#pragma unroll
    for (unsigned dmu = 0; dmu < 6; ++dmu) {
      unsigned int mu = mu_0 + dmu;
#pragma unroll
      for (unsigned int dnu = 0; dnu < 6; ++dnu) {
        unsigned int nu = nu_0 + dnu;
        //    const double correct_fact = (nu == mu) ? 0.5 : 1.0;
        S_gl[mu + nbas * nu] = S_gl[mu + nbas * nu];
        S_gl[nu + nbas * mu] = S_gl[mu + nbas * nu];
        Hcore_gl[mu + nbas * nu] +=
            correct_fact *
            Hcore_block_sh[threadIdx.x + BLOCKDIM * (dnu + 6 * dmu)];
        Hcore_gl[nu + nbas * mu] +=
            correct_fact *
            Hcore_block_sh[threadIdx.x + BLOCKDIM * (dnu + 6 * dmu)];
      }
    }
  } // If block
}

extern "C" void kin_ov_GPU_class_2_2(
    const unsigned int *__restrict__ Kab_gl, const double *__restrict__ Ax_gl,
    const double *__restrict__ Ay_gl, const double *__restrict__ Az_gl,
    const double *__restrict__ ABx_gl, const double *__restrict__ ABy_gl,
    const double *__restrict__ ABz_gl, const double *__restrict__ P_gl,
    const double *__restrict__ zeta_gl, const double *__restrict__ beta_gl,
    const double *__restrict__ UP_gl, const double *__restrict__ fz_gl,
    const unsigned int *offsets_sha, const unsigned int *offsets_shb,
    const unsigned int n_ab, const unsigned int nbas,
    double *__restrict__ Hcore_gl, double *__restrict__ S_gl,
    const unsigned int block_dim, cudaStream_t istream) {

  const float t_nab_d = ceil(((double)(n_ab) / block_dim));
  const unsigned int n_blocks = (unsigned int)t_nab_d;
  dim3 block_d(block_dim);
  dim3 grid_d(n_blocks);
  kin_ov_GPU_kernel_2_2<<<grid_d, block_d, 0, istream>>>(
      Kab_gl, Ax_gl, Ay_gl, Az_gl, ABx_gl, ABy_gl, ABz_gl, P_gl, zeta_gl,
      beta_gl, UP_gl, fz_gl, offsets_sha, offsets_shb, n_ab, nbas, Hcore_gl,
      S_gl);
  return;
}


