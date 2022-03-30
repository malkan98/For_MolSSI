/**
 * @file hgp_kernel_scheduler.h
 * @author Melisa Alkan (you@domain.com)
 * @brief Contains the scheduler declaration for invoking the cuda kernels
 * @version 0.1
 * @date 2021-01-12
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#ifndef HGP_KERNEL_SCHEDULER_H
#define HGP_KERNEL_SCHEDULER_H

#include "one_elec_integrals/one_elec_ints.h"

#include <cuda_runtime.h>

namespace genfock {
  namespace CUDA_GPU {

  /**
   * @brief Head Gordon Pople scheduler for equal kernels (ab == cd)
   * @author Melisa Alkan
   * @date December 2020
   * @param Kab_gl Ab contraction degree
   * @param Kcd_gl Cd contraction degree
   * @param offset_a Basis function offset in a
   * @param offset_b Basis function offset in b
   * @param offset_c Basis function offset in c
   * @param offset_d Basis function offset in d
   * @param Ax_gl    X coordinate of A center
   * @param Ay_gl    Y coordinate of A center
   * @param Az_gl    Z coordinate of A center
   * @param Cx_gl    X coorindate of C center
   * @param Cy_gl    Y coordinate of C center
   * @param Cz_gl    Z coordinate of C center
   * @param ABx_gl   X coordinte of AB center
   * @param ABy_gl   Y coordinate of AB center
   * @param ABz_gl   Z coordinate of AB center
   * @param CDx_gl   X coordinate of CD center
   * @param CDy_gl   Y coordinate of CD center
   * @param CDz_gl   Z coordinate of CD center
   * @param P_gl     P vector 
   * @param Q_gl     Q vector
   * @param zeta_gl  zeta exponent
   * @param eta_gl   eta exponent
   * @param UP_gl    UP quantity
   * @param UQ_gl    UQ quantity
   * @param fz_gl    fz quantity
   * @param fe_gl    fe quantity
   * @param nbasfunct Basis function number 
   * @param Density_matrix Density matrix pointer
   * @param Fock_matrix    Fock matrix pointer
   * @param n_ab           Batch size
   * @param n_cd           Ket size
   * @param block_dim      Block dimension
   * @param istream        Cuda stream
   * @param am_a           Angular momentum in a
   * @param am_b           Angular momentum in b
   * @param am_c           Angular momentum in c
   * @param am_d           Angular momentum in d
   * @param stride         Stride
   */
  void hgp_kernel_scheduler_equal(const unsigned int* Kab_gl, const unsigned int* Kcd_gl,
                            const unsigned int* offset_a, const unsigned int* offset_b,
                            const unsigned int* offset_c, const unsigned int* offset_d,
                            const double* Ax_gl, const double* Ay_gl, const double* Az_gl,
                            const double* Cx_gl, const double* Cy_gl, const double* Cz_gl,
                            const double* ABx_gl, const double* ABy_gl, const double* ABz_gl,
                            const double* CDx_gl, const double* CDy_gl, const double* CDz_gl,
                            const double* P_gl, const double* Q_gl, 
                            const double* zeta_gl, const double* eta_gl,
                            const double* UP_gl, const double* UQ_gl,
                            const double* fz_gl, const double* fe_gl,
                            unsigned nbasfunct, const double* __restrict__ Density_matrix, double* __restrict__ Fock_matrix,
                            const unsigned int n_ab, const unsigned int n_cd,
                            const unsigned int block_dim, cudaStream_t istream,
                            unsigned int am_a, unsigned int am_b,
                            unsigned int am_c, unsigned int am_d,
			                      unsigned int stride);

  /**
   * @brief Nuclear attraction integral kernel scheduler
   * @author Melisa Alkan
   * @date 2020
   * @param Kab_gl   Ab contraction degree
   * @param Ax_gl    X coordinate of A center
   * @param Ay_gl    Y coordinate of A center
   * @param Az_gl    Z coordinate of A center
   * @param ABx_gl   X coordinte of AB center
   * @param ABy_gl   Y coordinate of AB center
   * @param ABz_gl   Z coordinate of AB center
   * @param P_gl     P vector
   * @param R_gl     Nuclear coordinates
   * @param zeta_gl  Zeta exponent
   * @param n_atoms  Number of atoms
   * @param Z_gl     Atomic charge
   * @param UP_gl    UP quantity
   * @param fz_gl    fz
   * @param offsets_sha offset of shell a
   * @param offsets_shb offset of shell b
   * @param n_ab batch size
   * @param nbas number of basis functions
   * @param Hcore_gl hcore matrix pointer
   * @param block_dim block dimension for cuda
   * @param istream  stream
   * @param am_a  Angular momentum in a
   * @param am_b  Angular momentum in b
   */
  void nuc_attr_scheduler(const unsigned int* __restrict__ Kab_gl, const double* __restrict__ Ax_gl,
                          const double* __restrict__ Ay_gl, const double* __restrict__ Az_gl,
                          const double* __restrict__ ABx_gl, const double* __restrict__ ABy_gl, 
                          const double* __restrict__ ABz_gl, const double* __restrict__ P_gl, 
                          const double* __restrict__ R_gl, const double* __restrict__ zeta_gl,
                          const unsigned int n_atoms, const double* __restrict__ Z_gl,
                          const double* __restrict__ UP_gl, const double* __restrict__ fz_gl,
                          const unsigned int* offsets_sha, const unsigned int* offsets_shb,
                          const unsigned int n_ab, const unsigned int nbas, double* __restrict__ Hcore_gl,
                          const unsigned int block_dim, cudaStream_t istream,
                          unsigned int am_a, unsigned int am_b); 


  /**
   * @brief Kinetic energy and overlap integral kernel scheduler
   * @author Melisa Alkan
   * @date 2020
   * @param Kab_gl   Ab contraction degree
   * @param Ax_gl    X coordinate of A center
   * @param Ay_gl    Y coordinate of A center
   * @param Az_gl    Z coordinate of A center
   * @param ABx_gl   X coordinte of AB center
   * @param ABy_gl   Y coordinate of AB center
   * @param ABz_gl   Z coordinate of AB center
   * @param P_gl     P vector
   * @param zeta_gl  Zeta exponent
   * @param beta_gl  Beta exponent
   * @param UP_gl    UP quantity
   * @param fz_gl    fz
   * @param offsets_sha offset of shell a
   * @param offsets_shb offset of shell b
   * @param n_ab batch size
   * @param nbas number of basis functions
   * @param Hcore_gl hcore matrix pointer
   * @param S_gl overlap matrix pointer
   * @param block_dim block dimension for cuda
   * @param istream  stream
   * @param am_a  Angular momentum in a
   * @param am_b  Angular momentum in b 
   */
  void Kinetic_Overlap_scheduler(const unsigned int* __restrict__ Kab_gl, const double* __restrict__ Ax_gl,
                               const double* __restrict__ Ay_gl, const double* __restrict__ Az_gl,
                               const double* __restrict__ ABx_gl, const double* __restrict__ ABy_gl,
                               const double* __restrict__ ABz_gl, const double* __restrict__ P_gl,
                               const double* __restrict__ zeta_gl, const double* __restrict__ beta_gl,
                               const double* __restrict__ UP_gl, const double* __restrict__ fz_gl,
                               const unsigned int* offsets_sha, const unsigned int* offsets_shb,
                               const unsigned int n_ab, const unsigned int nbas,
                               double* __restrict__ Hcore_gl, double* __restrict__ S_gl,
                               const unsigned int block_dim, cudaStream_t istream,
                               unsigned int am_a, unsigned int am_b);

  

  }
}

#endif