/**
 * @file dscheduler.cpp
 * @author Melisa Alkan
 * @brief Implementation of scheduler up to d functions
 * @version 0.1
 * @date 2021-01-12
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include "hgp_kernel_scheduler.h"
#include <iostream>
namespace genfock {

namespace CUDA_GPU { 

  void hgp_kernel_scheduler_equal( const unsigned int* Kab_gl, const unsigned int* Kcd_gl,
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
                            const unsigned int block_dim,cudaStream_t istream,
                            unsigned int am_a, unsigned int am_b,
                            unsigned int am_c, unsigned int am_d,
			    unsigned int stride){

  unsigned int key = am_a*11*11*11 + am_b*11*11 + am_c*11 + am_d;
  switch(key){

      case 0 :
          compute_GPU_class_0_0_0_0_equal( Kab_gl, Kcd_gl, offset_a, offset_b, offset_c, offset_d,
                                   Ax_gl, Ay_gl, Az_gl, Cx_gl, Cy_gl, Cz_gl,
                                   P_gl, Q_gl, zeta_gl, eta_gl, UP_gl, UQ_gl,
                                   fz_gl, fe_gl, nbasfunct, Density_matrix, Fock_matrix,
                                   n_ab, n_cd, block_dim, istream,
                                   am_a, am_b, am_c, am_d, stride);
          break;

      case 1342 :
          compute_GPU_class_1_0_1_0_equal( Kab_gl, Kcd_gl, offset_a, offset_b, offset_c, offset_d,
                                   Ax_gl, Ay_gl, Az_gl, Cx_gl, Cy_gl, Cz_gl,
                                   P_gl, Q_gl, zeta_gl, eta_gl, UP_gl, UQ_gl,
                                   fz_gl, fe_gl, nbasfunct, Density_matrix, Fock_matrix,
                                   n_ab, n_cd, block_dim, istream,
                                   am_a, am_b, am_c, am_d, stride);
          break;

      case 1464 :
          compute_GPU_class_1_1_1_1_equal( Kab_gl, Kcd_gl, offset_a, offset_b, offset_c, offset_d,
                                   Ax_gl, Ay_gl, Az_gl, Cx_gl, Cy_gl, Cz_gl,
                                   ABx_gl, ABy_gl, ABz_gl, CDx_gl, CDy_gl, CDz_gl,
                                   P_gl, Q_gl, zeta_gl, eta_gl, UP_gl, UQ_gl,
                                   fz_gl, fe_gl, nbasfunct, Density_matrix, Fock_matrix,
                                   n_ab, n_cd, block_dim, istream,
                                   am_a, am_b, am_c, am_d, stride);
          break;
      case 2684 :
        compute_GPU_class_2_0_2_0_equal( Kab_gl, Kcd_gl, offset_a, offset_b, offset_c, offset_d,
                             Ax_gl, Ay_gl, Az_gl, Cx_gl, Cy_gl, Cz_gl,
                             P_gl, Q_gl, zeta_gl, eta_gl, UP_gl, UQ_gl,
                             fz_gl, fe_gl, nbasfunct, Density_matrix, Fock_matrix,
                             n_ab, n_cd, block_dim,istream,
                             am_a, am_b, am_c, am_d, stride);
                break;

      case 2806 :
          compute_GPU_class_2_1_2_1_equal( Kab_gl, Kcd_gl, offset_a, offset_b, offset_c, offset_d,
                                   Ax_gl, Ay_gl, Az_gl, Cx_gl, Cy_gl, Cz_gl,
                                   ABx_gl, ABy_gl, ABz_gl, CDx_gl, CDy_gl, CDz_gl,
                                   P_gl, Q_gl, zeta_gl, eta_gl, UP_gl, UQ_gl,
                                   fz_gl, fe_gl, nbasfunct, Density_matrix, Fock_matrix,
                                   n_ab, n_cd, block_dim,istream,
                                   am_a, am_b, am_c, am_d, stride);
                    break;

            case 2928 :
          compute_GPU_class_2_2_2_2_equal( Kab_gl, Kcd_gl, offset_a, offset_b, offset_c, offset_d,
                                   Ax_gl, Ay_gl, Az_gl, Cx_gl, Cy_gl, Cz_gl,
                                   ABx_gl, ABy_gl, ABz_gl, CDx_gl, CDy_gl, CDz_gl,
                                   P_gl, Q_gl, zeta_gl, eta_gl, UP_gl, UQ_gl,
                                   fz_gl, fe_gl, nbasfunct, Density_matrix, Fock_matrix,
                                   n_ab, n_cd, block_dim,istream,
                                   am_a, am_b, am_c, am_d, stride);
				   break;

      default :
          std::cerr << "Class not supported!" << std::endl;
          exit(1);
    }


    return;

  } // equal scheduler


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
                          unsigned int am_a, unsigned int am_b){

    unsigned int key = am_a*11 + am_b;

    switch(key){
        case 0 :
            nuc_attr_GPU_class_0_0(Kab_gl, Ax_gl, Ay_gl, Az_gl,
                                    P_gl, R_gl, zeta_gl,
                                    n_atoms, Z_gl, UP_gl, fz_gl,
                                    offsets_sha, offsets_shb,
                                    n_ab, nbas, Hcore_gl, block_dim, istream);
            break;

        case 11 :
            nuc_attr_GPU_class_1_0(Kab_gl, Ax_gl, Ay_gl, Az_gl,
                                    P_gl, R_gl, zeta_gl,
                                    n_atoms, Z_gl, UP_gl, fz_gl,
                                    offsets_sha, offsets_shb,
                                    n_ab, nbas, Hcore_gl, block_dim, istream);
            break;

        case 1 :
            nuc_attr_GPU_class_0_1(Kab_gl, Ax_gl, Ay_gl, Az_gl,
                                    ABx_gl, ABy_gl, ABz_gl,
                                    P_gl, R_gl, zeta_gl, n_atoms,
                                    Z_gl, UP_gl, fz_gl,
                                    offsets_sha, offsets_shb,
                                    n_ab, nbas, Hcore_gl, block_dim, istream);
            break;


        case 12 :
            nuc_attr_GPU_class_1_1(Kab_gl, Ax_gl, Ay_gl, Az_gl,
                                    ABx_gl, ABy_gl, ABz_gl,
                                    P_gl, R_gl, zeta_gl, n_atoms,
                                    Z_gl, UP_gl, fz_gl,
                                    offsets_sha, offsets_shb,
                                    n_ab, nbas, Hcore_gl, block_dim, istream);
            break;   

        case 22 :
            nuc_attr_GPU_class_2_0(Kab_gl, Ax_gl, Ay_gl, Az_gl,
                                    P_gl, R_gl, zeta_gl,
                                    n_atoms, Z_gl, UP_gl, fz_gl,
                                    offsets_sha, offsets_shb,
                                    n_ab, nbas, Hcore_gl, block_dim, istream);
            break;

        case 2 :
            nuc_attr_GPU_class_0_2(Kab_gl, Ax_gl, Ay_gl, Az_gl,
                                    ABx_gl, ABy_gl, ABz_gl,
                                    P_gl, R_gl, zeta_gl, n_atoms,
                                    Z_gl, UP_gl, fz_gl,
                                    offsets_sha, offsets_shb,
                                    n_ab, nbas, Hcore_gl, block_dim, istream);
            break;

        case 23 :
            nuc_attr_GPU_class_2_1(Kab_gl, Ax_gl, Ay_gl, Az_gl,
                                    ABx_gl, ABy_gl, ABz_gl,
                                    P_gl, R_gl, zeta_gl, n_atoms,
                                    Z_gl, UP_gl, fz_gl,
                                    offsets_sha, offsets_shb,
                                    n_ab, nbas, Hcore_gl, block_dim, istream);
            break;

        case 13 :
            nuc_attr_GPU_class_1_2(Kab_gl, Ax_gl, Ay_gl, Az_gl,
                                    ABx_gl, ABy_gl, ABz_gl,
                                    P_gl, R_gl, zeta_gl, n_atoms,
                                    Z_gl, UP_gl, fz_gl,
                                    offsets_sha, offsets_shb,
                                    n_ab, nbas, Hcore_gl, block_dim, istream);
            break;

        case 24 :
            nuc_attr_GPU_class_2_2(Kab_gl, Ax_gl, Ay_gl, Az_gl,
                                    ABx_gl, ABy_gl, ABz_gl,
                                    P_gl, R_gl, zeta_gl, n_atoms,
                                    Z_gl, UP_gl, fz_gl,
                                    offsets_sha, offsets_shb,
                                    n_ab, nbas, Hcore_gl, block_dim, istream);
            break;

      	default :
          std::cerr << "Class not supported!" << std::endl;
          exit(1);
          }

	return;
        }


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
                               unsigned int am_a, unsigned int am_b){

    unsigned int key = am_a*11 + am_b;
        switch(key){
        case 0 :
            kin_ov_GPU_class_0_0(Kab_gl, Ax_gl, Ay_gl, Az_gl,
                                 ABx_gl, ABy_gl, ABz_gl,
                                 P_gl, zeta_gl, beta_gl,
                                 UP_gl, fz_gl, offsets_sha, offsets_shb,
                                 n_ab, nbas, Hcore_gl, S_gl, block_dim, istream);
            break;

        case 11 :
            kin_ov_GPU_class_1_0(Kab_gl, Ax_gl, Ay_gl, Az_gl,
                                 ABx_gl, ABy_gl, ABz_gl,
                                 P_gl, zeta_gl, beta_gl,
                                 UP_gl, fz_gl, offsets_sha, offsets_shb,
                                 n_ab, nbas, Hcore_gl, S_gl, block_dim, istream);
            break;

                case 1 :
            kin_ov_GPU_class_0_1(Kab_gl, Ax_gl, Ay_gl, Az_gl,
                                 ABx_gl, ABy_gl, ABz_gl,
                                 P_gl, zeta_gl, beta_gl,
                                 UP_gl, fz_gl, offsets_sha, offsets_shb,
                                 n_ab, nbas, Hcore_gl, S_gl, block_dim, istream);
            break;


        case 12 :
            kin_ov_GPU_class_1_1(Kab_gl, Ax_gl, Ay_gl, Az_gl,
                                 ABx_gl, ABy_gl, ABz_gl,
                                 P_gl, zeta_gl, beta_gl,
                                 UP_gl, fz_gl, offsets_sha, offsets_shb,
                                 n_ab, nbas, Hcore_gl, S_gl, block_dim, istream);
            break;   

        case 22 :
            kin_ov_GPU_class_2_0(Kab_gl, Ax_gl, Ay_gl, Az_gl,
                                 ABx_gl, ABy_gl, ABz_gl,
                                 P_gl, zeta_gl, beta_gl,
                                 UP_gl, fz_gl, offsets_sha, offsets_shb,
                                 n_ab, nbas, Hcore_gl, S_gl, block_dim, istream);
            break;

        case 2 :
            kin_ov_GPU_class_0_2(Kab_gl, Ax_gl, Ay_gl, Az_gl,
                                 ABx_gl, ABy_gl, ABz_gl,
                                 P_gl, zeta_gl, beta_gl,
                                 UP_gl, fz_gl, offsets_sha, offsets_shb,
                                 n_ab, nbas, Hcore_gl, S_gl, block_dim, istream);
            break;

        case 23 :
            kin_ov_GPU_class_2_1(Kab_gl, Ax_gl, Ay_gl, Az_gl,
                                 ABx_gl, ABy_gl, ABz_gl,
                                 P_gl, zeta_gl, beta_gl,
                                 UP_gl, fz_gl, offsets_sha, offsets_shb,
                                 n_ab, nbas, Hcore_gl, S_gl, block_dim, istream);
            break;

        case 13 :
            kin_ov_GPU_class_1_2(Kab_gl, Ax_gl, Ay_gl, Az_gl,
                                 ABx_gl, ABy_gl, ABz_gl,
                                 P_gl, zeta_gl, beta_gl,
                                 UP_gl, fz_gl, offsets_sha, offsets_shb,
                                 n_ab, nbas, Hcore_gl, S_gl, block_dim, istream);
            break;

        case 24 :
            kin_ov_GPU_class_2_2(Kab_gl, Ax_gl, Ay_gl, Az_gl,
                                 ABx_gl, ABy_gl, ABz_gl,
                                 P_gl, zeta_gl, beta_gl,
                                 UP_gl, fz_gl, offsets_sha, offsets_shb,
                                 n_ab, nbas, Hcore_gl, S_gl, block_dim, istream);
            break;    

        default :
          std::cerr << "Class not supported!" << std::endl;
          exit(1);
        }

        return;
       
       
        }

return;
} //kinetic overlap scheduler
}
