#define __timing 0
!---------------------------------------------------------------------
!=====================================================================
!---------------------------------------------------------------------

!*module omp_gpu_ints
!>  @brief   in this module gpu-enabled openmp versions of the two-electron integral 
!>           (twoei) packages are called, namely sp,spd,eric,rys. this is a modified
!>           version of the threaded openmp code. currently, only sp and rys have gpu
!>           capabilities. the code is only called with -qopenmp and -qoffload flags
!>           enabled. 
!>  @details the underlying twoei routines are in int2a.src, int2b.src, and int2c.src.
!>           a lot of work has been done to inline the routines, since most of gpu
!>           openmp compilers don't support nested subroutines for offloading. 
!>  @author  melisa alkan
!>  @date    2020-2022

!---------------------------------------------------------------------
!=====================================================================
!---------------------------------------------------------------------

module omp_gpu_ints

    use omp_lib
    use prec, only: fp

    use mx_limits, only: &
        mxsh, mxgtot, mxatm, mxang, mxang2, mxgsh, mxg2

    implicit none

    private

    public &
        ompmod_twoei_gpu

    real(kind=fp), parameter :: &
        zero=0.0_fp, one=1.0_fp

    interface
        real(kind=fp) function schwdn(dsh,ish,jsh,ksh,lsh,ia)
            use prec
            integer, intent(in) :: ish, jsh, ksh, lsh, ia(*)
            real(kind=fp), intent(in) :: dsh(*)
        end function
    end interface

contains


!---------------------------------------------------------------------
!=====================================================================
!---------------------------------------------------------------------


!---------------------------------------------------------------------
!=====================================================================
!---------------------------------------------------------------------

!*module ompmod_gpu_ints  ompmod_twoei_gpu
!>    @brief   Driver for all ERI packages
!>
!>    @details Calls TWOEI routines based on the input molecule;
!>             chooses "best" package; Schwartz screening is done
!>             on CPU. Supports s,p,d shells for now.
!>
!>    @author  Melisa Alkan
!
!>    @date _may, 2021
!
!     parameters:
!
!>    @param[in]     schwrz    for Schwartz screening
!>    @param[in]     typscf    rhf,uhf,rohf -> currently gpu support is for rhf only
!>    @param[inout]  da,db     alpha and beta densities
!>    @param[out]    fa,fb     alpha and beta fock matrix
  subroutine ompmod_twoei_gpu(typscf,schwrz,nint,nschwz, &
                             l1,l2a,l2b,xints, &
                             nsh2,maxg, &
                             ia,da,fa,db,fb,dsh,nflmat, &
                             cutoff, oflag)

    logical, intent(in) :: schwrz
    integer, intent(out) ::  nint, nschwz
    integer, intent(in) :: l1, l2a, l2b, nsh2, nflmat, maxg, ia(l1)
    real(kind=8), intent(in) :: typscf
    real(kind=fp), intent(in) :: cutoff, xints(nsh2), dsh(nsh2)
    real(kind=fp), intent(inout) :: da(l2a), db(l2b)
    real(kind=fp), intent(out) :: fa(l2a*nflmat), fb(l2b*nflmat)
    logical, intent(in) :: oflag

    common /par   / me,master,nproc,ibtyp,iptim,goparr,dskwrk,maswrk
        integer :: me,master,nproc,ibtyp,iptim
        logical :: dskwrk, maswrk, goparr
    common /nshel / ex(mxgtot),cs(mxgtot),cp(mxgtot),cd(mxgtot),     &
                    cf(mxgtot),cg(mxgtot),ch(mxgtot),ci(mxgtot),     &
                    kstart(mxsh),katom(mxsh),ktype(mxsh),kng(mxsh),  &
                    kloc(mxsh),kmin(mxsh),kmax(mxsh),nshell
        integer :: kstart,katom,ktype,kng,kloc,kmin,kmax,nshell
        real(kind=fp) :: ex,cs,cp,cd,cf,cg,ch,ci
    common /shlnos / qq4,lit,ljt,lkt,llt,loci,locj,lock,locl, &
                    mini,minj,mink,minl,maxi,maxj,maxk,maxl, &
                    nij,ij,kl,ijkl
        integer :: lit,ljt,lkt,llt,loci,locj,lock,locl, &
                   mini,minj,mink,minl,maxi,maxj,maxk,maxl, &
                   nij,ij,kl,ijkl
        real(kind=fp) :: qq4
    common /maxc  / cmax(mxgtot),cmaxa(mxgsh),cmaxb(mxgsh), &
                    cmaxc(mxgsh),cmaxd(mxgsh),ismlp(mxg2),ismlq
        real(kind=fp) :: cmax,cmaxa,cmaxb,cmaxc,cmaxd
        integer :: ismlp,ismlq
    common /infoa / nat,ich,mul,num,nqmt,ne,na,nb, &
                  zan(mxatm),c(3,mxatm),ian(mxatm)
        integer :: nat,ich,mul,num,nqmt,ne,na,nb
        real(kind=fp) :: zan,ian,c
    common /b     / co(mxsh,3)
        real(kind=fp) :: co
    common /intac2 / ei1,ei2,cux
        real(kind=fp) :: ei1,ei2,cux
    common /fmttbl / fgrid(0:ntx,0:npf,0:ngrd),xgrid(0:ntx,0:npx), &
                    tmax,rfinc(0:ngrd),rxinc, &
                    rmr(mxqt),tlgm(0:mxqt),nord
        real(kind=fp) :: fgrid,xgrid,tmax,rfinc,rxinc, &
                         rmr,tlgm
        integer :: nord

integer,parameter :: ntx=4
integer,parameter :: npf=450
integer,parameter :: ngrd=7
integer,parameter :: npx=1000
integer,parameter :: mxqt=16

!   common blocks /maxc  /, /shlnos/ contain
!   both global (cmax(:),qq4,norgp) and thread-local
!   data (the rest).

!$omp threadprivate(/shlnos/,/maxc  /,/shlinf/)

!
!   --- initialization of variables ---
!
  integer :: ii, jj, kk, ll, ijij, klkl, jork
  real(kind=fp) :: w1,w2
  logical :: do_raxintsp,do_raxintspd,do_eric,do_rysint
  integer :: ncount_raxintsp,ncount_raxintspd,ncount_eric,ncount_rysint
  integer,allocatable,dimension(:,:) :: index_raxintsp,index_raxintspd,index_eric_ts,index_rysint

  !====for now integrals up to d function only====
  write(*,*) "max angular momentum", maxval(ktype)
  if(maxval(ktype).gt.4) then
     write(*,'(a15,f15.3)') "no f functions on gpus yet"
     call abrt()
  endif

      !calculate num of quartets of certain int types 
      call rhf_ncount_gpu &
          (ncount_raxintsp,ncount_raxintspd,ncount_eric,ncount_rysint, &
           dirscf,schwrz,cutoff,nschwz,nsh2,l1,typscf,xints,dsh,nint,ia)

  !=== decide which integral packages to use based on the molecule input ===
  do_raxintsp=.false.
  do_raxintspd=.false.
  do_eric=.false.
  do_rysint=.false.

  if(ncount_raxintsp.gt.0) do_raxintsp=.true.
  if(ncount_raxintspd.gt.0) do_raxintspd=.true.
  if(ncount_eric.gt.0) do_eric=.true.
  if(ncount_rysint.gt.0) do_rysint=.true.

  ! allocate shell index arrays
  if(do_raxintsp) allocate(index_raxintsp(4,ncount_raxintsp))
  if(do_raxintspd) allocate(index_raxintspd(4,ncount_raxintspd))
  if(do_eric) allocate(index_eric(4,ncount_eric))
  if(do_rysint) allocate(index_rysint(4,ncount_rysint))

      ! add shell information to index arrays
      ! for their respective packages; 
      ! perform schwrz screening here 
      call rhf_index_gpu &
          (index_raxintsp,index_raxintspd,index_eric,index_rysint, &
           ncount_raxintsp,ncount_raxintspd,ncount_eric,ncount_rysint, &
           schwrz,cutoff,nschwz,nsh2,l1,typscf,xints,dsh,nint,ia)

      !====do sp integrals on gpus via openmp target====
      if(do_raxintsp) then
      w1 = omp_get_wtime()
      call sp_gpu &
          (typscf,schwrz,nint,nschwz, &
           l1,l2a,l2b,xints, &
           nsh2,maxg,index_raxintsp, &
           ia,da,fa,db,fb,dsh,nflmat, &
           cutoff, oflag, &
           cmax(1:mxex),ex(1:mxex),cs(1:mxex),cp(1:mxex),cd(1:mxex),     &
           cf(1:mxex),cg(1:mxex),ch(1:mxex),ci(1:mxex),     &
           kstart(1:nshell),katom(1:nshell),ktype(1:nshell),kng(1:nshell),  &
           kloc(1:nshell),kmin(1:nshell),kmax(1:nshell),nshell,mxex, &
           fgrid,xgrid,tmax,rfinc,rxinc,rmr,tlgm,nord, &
           co(1:nshell,1:3))
         enddo
      w2 = omp_get_wtime()
      write(*,* "time in sp gpu", w2-w1
      endif

      !====do spd integrals====
      if(do_raxintspd) then
      w1 = omp_get_wtime()
         do icount_raxintspd=1,ncount_raxintspd
            ish=index_raxintspd(1,icount_raxintspd)
            jsh=index_raxintspd(2,icount_raxintspd)
            ksh=index_raxintspd(3,icount_raxintspd)
            lsh=index_raxintspd(4,icount_raxintspd)
            call ompmod_raxintspd(ish, jsh, ksh, lsh, ghondo)
            call dirfck(typscf,ia,da,fa,db,fb,ghondo, &
                        l2a,nint,nflmat)
         enddo
      w2 = omp_get_wtime()
      write(*,*) "time in spd cpu ", w2-w1
      endif

      !====do eric integrals====
      if(do_eric) then
      w1 = omp_get_wtime()
         do icount_raxintspd=1,ncount_raxintspd
            ish=index_raxintspd(1,icount_raxintspd)
            jsh=index_raxintspd(2,icount_raxintspd)
            ksh=index_raxintspd(3,icount_raxintspd)
            lsh=index_raxintspd(4,icount_raxintspd)
            call ompmod_eric(ish, jsh, ksh, lsh, ghondo)
            call dirfck(typscf,ia,da,fa,db,fb,ghondo, &
                        l2a,nint,nflmat)
         enddo
      w2 = omp_get_wtime()
      write(*,*) "time in eric cpu ", w2-w1
      endif

      !====do rys integrals on gpus via openmp target====
      if(do_rysint) then
      w1 = omp_get_wtime()
      call rys_gpu &
          (typscf,schwrz,nint,nschwz, &
           l1,l2a,l2b,xints, &
           nsh2,maxg,,index_rysint, &
           ia,da,fa,db,fb,dsh,nflmat, &
           cutoff, oflag, &
           cmax(1:mxex),ex(1:mxex),cs(1:mxex),cp(1:mxex),cd(1:mxex),     &
           cf(1:mxex),cg(1:mxex),ch(1:mxex),ci(1:mxex),     &
           kstart(1:nshell),katom(1:nshell),ktype(1:nshell),kng(1:nshell),  &
           kloc(1:nshell),kmin(1:nshell),kmax(1:nshell),nshell,mxex, &
           fgrid,xgrid,tmax,rfinc,rxinc,rmr,tlgm,nord, &
           co(1:nshell,1:3))
        w2 = omp_get_wtime()
        write(*,*) "time in rys gpu ", w2-w1
      endif

  ! deallocate shell index arrays
  if(do_raxintsp) deallocate(index_raxintsp)
  if(do_raxintspd) deallocate(index_raxintspd)
  if(do_eric) deallocate(index_eric)
  if(do_rysint) deallocate(index_rysint)

  end subroutine ompmod_twoei_gpu
