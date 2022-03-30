#define __TIMING 0
!---------------------------------------------------------------------
!=====================================================================
!---------------------------------------------------------------------

!*MODULE SPD_ints
!>  @brief   In this module contracted integrals of s,p,d functions
!>           via a combined Pople-Hehre and McMurchie-Davidson method are
!>           offloaded. Paper by Ishimura&Nagase (Theor.Chem.Acc,2008).
!>           This code is only called with -qopenmp and -qoffload flags
!>           enabled. 
!>
!>  @details The underlying TWOEI routines are in int2c.src. Multi-GPU
!>           execution is enabled through DDI. 
!>
!>  @author  Melisa Alkan
!>  @date    2022
subroutine spd_0002 &
          (n_s_shl,i_s_shl, &
          n_d_shl,i_d_shl, &            
          l1,l2a,l2b,xints, &
          nsh2,maxg, &
          ia,da,fa,db,fb,dsh,nflmat, &
          cutoff, &
          cmax,lrint, &
          nshell,kng,kstart,kmin,kmax,kloc, &
          ex,cs,cp,cd,co, &
          fgrid,xgrid,tmax,rfinc,rxinc, &
          rmr,tlgm, &
          ei1,ei2,cux, &
          ssxints,sdxints,xintsss,xintssd)

    use omp_lib
    use prec, only: fp
    use mx_limits, only: &
        mxsh, mxgtot, mxatm, mxang, mxang2, mxgsh, mxg2
    use camdft, only: camflag
    use lrcdft, only: lcflag, emu, emu2

    implicit none

    integer, intent(in) :: l1,l2a,l2b, nsh2, nflmat, maxg, ia(l1)
    real(kind=fp), intent(in) ::  cutoff, xints(nsh2), dsh(nsh2)
    real(kind=fp), intent(inout) :: da(l2a), db(l2b)
    real(kind=fp) :: fa(l2a*nflmat), fb(l2b*nflmat) 

    integer :: ish, jsh, ksh, lsh
    integer, intent(in) ::  n_s_shl,n_d_shl
    integer, intent(in) ::  i_s_shl(n_s_shl),i_d_shl(n_d_shl)
    integer :: i1,j1,k1,ll1
    integer :: ii,i,ijk,ip,ijp,ijkp,ijklp,j,k,l
    integer :: lpopi,lpopj,lpopk,lpopl
    integer :: ik,il,jk,jl
    integer :: is1,js1,ks1,ls1
    integer :: maxj2,maxl2,ii1,jj1,kk1,itmp,nkl
    integer :: i2,j2,k2,l2,jj,kk,ll,ii2,jj2,kk2

    !basis set info and important arrays
    integer :: nga,ngb,ngc,ngd,ngangb !shell contraction
    integer :: ni,nj,nk,nl,n,ji
    integer :: ismlp(mxg2),ismlq,isml
    integer :: lit,ljt,lkt,llt,loci,locj,lock,locl, & !angular mom, shell location
                mini,minj,mink,minl,maxi,maxj,maxk,maxl, &
                nij,ij,kl,ijkl !min and max of shells 
    integer :: nshell,kng(nshell),kstart(nshell),kmin(nshell), &
               kmax(nshell),kloc(nshell) !k-arrays from nshell common block
    real(kind=fp) :: r00(2),r01(3),r03(6) 
    real(kind=fp) :: cmax(mxgtot)
    real(kind=fp) :: r12,rab,x34,x43,aqz,qpr,qps, & !several below for distances
                     tx12(mxg2),tx21(mxg2),ty01(mxg2),ty02(mxg2), & !for atom coordinates
                     d00p(mxg2) !density
    real(kind=fp) :: p12(3,3),p34(3,3),p(3,3),t(3),sq(3)
    real(kind=fp) :: rcd,r34,cosg,sing,tmp
    real(kind=fp) :: acx,acz,acy,acy2,aqx,aqx2,aqxy,y03,y04   
    real(kind=fp) :: x01,x02,x12,x21,y01,y02,y12,r12y12
    real(kind=fp) :: e12,tst,x03,x04,y34,r34y34,e34,cq,cqx,cqz
    real(kind=fp) :: gpople(6),gpople_final(6) !main integral arrays 
    real(kind=fp) :: fqd(0:2),fq0(1),fq1(2),fq2(3),fqd2(3),fq(0:1) !boys function
    real(kind=fp) :: ex(mxgtot),cs(mxgtot),cp(mxgtot),cd(mxgtot) !exponents,coefficients
    real(kind=fp) :: co(mxsh,3)
    real(kind=fp) :: fgrid(0:ntx,0:npf,0:ngrd),xgrid(0:ntx,0:npx), &
                     tmax,rfinc(0:ngrd),rxinc, &
                     rmr(mxqt),tlgm(0:mxqt)
    real(kind=fp) :: c1(2),c2(3),c3(6), q(6,6) !int buffers
    real(kind=fp) :: ei1,ei2,cux,t2,et
    real(kind=fp) :: buff(10) !this buffer is to reuse intermediates
    real(kind=fp) :: fqz,pqr,pqs,rho,efr,xva
    real(kind=fp) :: tv,fx,fqf,x41,xin,rox,xmd2,xmd1,xmdt
! -------------------------------
!necessary constants
    real(kind=fp),parameter :: pito52=34.986836655249726d+00
    real(kind=fp),parameter :: zer=0.0d+00
    real(kind=fp),parameter :: pt5=0.5d+00
    real(kind=fp),parameter :: one=1.0d+00
    real(kind=fp),parameter :: pt7=0.7d+00
    real(kind=fp),parameter :: pt9=0.9d+00
!acycut changed from a value of 1e-4 in 2004,
!for accuracy when using diffuse exponents.
    real(kind=fp),parameter :: acycut=1.0d-10
    real(kind=fp),parameter :: tenm12=1.0d-122)
    real(kind=fp),parameter :: half=0.5d00,four=4.0d00, two=2.0d00
    real(kind=fp),parameter :: sqrt3=1.73205080756888d+00
    real(kind=fp),parameter :: pi4=0.78539816339744831d+00
    real(kind=fp),parameter :: xval1=1.0d+00,xval4=4.0d+00
    integer,parameter :: ntx=4
    integer,parameter :: npf=450
    integer,parameter :: ngrd=7
    integer,parameter :: npx=1000
    integer,parameter :: mxqt=16
! --------------------------
!for omp variables and chunking/distributing work
!across many GPUs
integer :: istart,iend,iijj,iijjkkll,kkll
integer :: nthreads,nquarts,iquart,nteams
integer :: nchunksize_tmp,nchunksize_mod,istart_tmp,iend_tmp
integer :: ijsh,niijj,nkkll,ikk,iijj_tmp,kkll_tmp
integer :: ish_tmp,jsh_tmp,ksh_tmp,lsh_tmp

!for screening
integer :: ssxints(n_s_shl*(n_s_shl+1)/2)
integer :: sdxints(n_s_shl*n_d_shl)
real(kind=fp) :: xintsss(n_s_shl*(n_s_shl+1)/2)
real(kind=fp) :: xintssd(n_s_shl*n_d_shl)
real(kind=fp) :: scut=1.0d-15,scutij,scutkl

real(kind=fp),allocatable :: xintss0020(:),xintds0020(:)
integer,allocatable :: nss0020(:),nds0020(:)
integer,allocatable :: idxid(:,:),idx(:)

lpopi =  0  !0
lpopj =  0  !0
lpopk =  0  !0
lpopl =  1  !2

mini = 1
minj = 1
mink = 1
maxi = 1
maxj = 1
maxk = 1
maxl = 6
!screening and gathering the quartet array
allocate(nss0020(n_s_shl*(n_s_shl+1)/2))
allocate(xintss0020(n_s_shl*(n_s_shl+1)/2))
allocate(nds0020(n_d_shl*n_s_shl))
allocate(xintds0020(n_d_shl*n_s_shl))

scutij = cutoff/maxval(xintssd)
niijj=0
do iijj=1,n_s_shl*(n_s_shl+1)/2
   if(xintsss(iijj).ge.scutij) then
      niijj=niijj+1
      xintss0020(niijj) = xintsss(iijj)
      nss0020(niijj) = iijj
   endif
enddo
scutkl = cutoff/maxval(xintsss)
nkkll=0
do iijj=1,n_d_shl*n_s_shl
   if(xintssd(iijj).ge.scutkl) then
      nkkll=nkkll+1
      xintds0020(nkkll) = xintssd(iijj)
      nds0020(nkkll) = iijj
   endif
enddo

allocate(idx(niijj*nkkll))
nquarts=0
do iijj=1,niijj
   do kkll=1,nkkll
      if(xintss0020(iijj)*xintds0020(kkll) .ge. cutoff) then
         nquarts=nquarts+1
         idx(nquarts)=kkll+(iijj-1)*nkkll
      endif
   enddo
enddo

nteams=80
nthreads=32
! ---------------------------------------
!$omp target teams distribute parallel do default(none) &
!$omp num_teams(nteams) thread_limit(nthreads) &
!$omp map(to:idx(1:nquarts),nquarts) &
!$omp map(to:nss0020,nds0020) &
!$omp map(to:da,xints,dsh,ia,cutoff,rmr) &
!$omp map(to:i_s_shl,n_s_shl,n_d_shl,i_d_shl) &
!$omp map(to:kstart,kmin,kloc,kng,co,ex,cmax,cs,cp,cd) &
!$omp map(to:rfinc,rxinc,fgrid,xgrid,tmax) &
!$omp map(to:emu2,ei1,ei2,cux) &
!$omp shared(idx,nquarts,nkkll) &
!$omp shared(fa,da,xints,dsh,ia,cutoff) &
!$omp shared(i_s_shl,n_s_shl,n_d_shl,i_d_shl) &
!$omp shared(kstart,kmin,kloc,kng,co,ex,cmax,cs,cp,cd) &
!$omp shared(rfinc,fgrid,tmax) &
!$omp shared(emu2,ei1,ei2,cux) &
!$omp shared(maxl,xval4,xval1) &
!
!$omp private(t2,xmd2,et,fqd,c1,q,gpople_final,rox,xmdt) &
!$omp private(kngtot,k1,j1,i1,r03,xmd1,c2,c3,fqd2) &
!$omp private(denmax,test) &
!$omp private(d01p,d00p,sq,fq0,fq1) &
!$omp private(isml,ismlq,ismlp) &
!$omp private(nga,ngb,ngc,ngd,ngangb,minl,locl) &
!$omp private(ii,jj,kk,ll) &
!$omp private(ii2,jj2,kk2) &
!$omp private(i2,j2,k2,l2) &
!$omp private(ii1,jj1,kk1,ll1) &
!$omp private(kl,jk,jl,il,ij,ik,ip,ji) &
!$omp private(ish,jsh,ksh,lsh) &
!$omp private(lsh_tmp,ish_tmp,jsh_tmp,ksh_tmp) &
!$omp private(is1,js1,ks1,ls1,r12) &
!$omp private(p,p12,p34) &
!$omp private(buff,r00,r01,gpople,fq,fqf) &
!$omp private(y34,x43,y03,y04,x34,x03,x04) &
!$omp private(acz,acy2,acy,acx) &
!$omp private(aqx2,aqxy,aqz,aqx) &
!$omp private(cq,cqx,cqz,qps,cosg,sing) &
!$omp private(ty01,tx12,tx21,ty02) &
!$omp private(rab,rcd,r34y34,r34,e34,tst) &
!$omp private(iijj,kkll,iijjkkll) &
!$omp private(iijj_tmp,kkll_tmp)
do iquart = 1,nquarts
    iijjkkll = idx(iquart)

    iijj_tmp = (iijjkkll-1)/nkkll + 1
    kkll_tmp = mod(iijjkkll-1,nkkll) + 1

    iijj=nss0020(iijj_tmp)
    kkll=nds0020(kkll_tmp)

    ish_tmp = (1 + sqrt(1.0+8.0*(iijj-1)))/2
    jsh_tmp = iijj - ish_tmp*(ish_tmp-1)/2

    ksh_tmp = (kkll-1)/n_d_shl + 1
    lsh_tmp = mod(kkll-1,n_d_shl) + 1

    ii = i_s_shl(ish_tmp) !s
    jj = i_s_shl(jsh_tmp) !s
    kk = i_s_shl(ksh_tmp) !s
    ll = i_d_shl(lsh_tmp) !d

    ish=ii
    jsh=jj
    ksh=kk
    lsh=ll

! zero out necessary arrays 
    r00(1:2)= zer
    r01(1:3)= zer
    r03(1:6)= zer
    gpople=zer

! obtain info about shells: check k-arrays in the common
! block nshell; 
! gaussian exponents go into ex, coefficients go into cs,cp,cd 

    nga= kng(ish)
    ngb= kng(jsh)
    ngc= kng(ksh)
    ngd= kng(lsh)
    ngangb=nga*ngb      

! starting locations of shells is1,js1,ks1,ls1 

    is1= kstart(ish)
    js1= kstart(jsh)
    ks1= kstart(ksh)
    ls1= kstart(lsh)

! get coordinates of atoms associated with 
! ab and cd r12&r34
! below is Pople-Hehre algorithm

      r12= zer
      r34= zer
      do 150 n=1,3
         p12(n,1)= co(ish,n)
         p12(n,2)= co(jsh,n)
         p12(n,3)= p12(n,2)-p12(n,1)
      r12= r12+p12(n,3)*p12(n,3)
         p34(n,1)= co(ksh,n)
         p34(n,2)= co(lsh,n)
         p34(n,3)= p34(n,2)-p34(n,1)
  150 r34= r34+p34(n,3)*p34(n,3)
!
! find direction cosines of penultimate axes from coordinates of ab
! p(1,1),p(1,2),... are direction cosines of axes at p.  z-axis along ab
! buff(1),buff(2),buff(3)... are direction cosines of axes at q.  z-axis along cd
!
! find direction cosines of ab and cd. these are local z-axes.
! if indeterminate take along space z-axis
!
      p(1:2,3)= zer
      p(3,3)= one
      rab= zer
      if(r12.ne.zer) then
         rab= sqrt(r12)
         buff(4)= one/rab
         p(1,3)= p12(1,3)*buff(4)
         p(2,3)= p12(2,3)*buff(4)
         p(3,3)= p12(3,3)*buff(4)
      endif

      buff(1:2)= zer
      buff(3)= one
      rcd= zer
      if(r34.ne.zer) then
         rcd= sqrt(r34)
         buff(4)= one/rcd
         buff(1)= p34(1,3)*buff(4)
         buff(2)= p34(2,3)*buff(4)
         buff(3)= p34(3,3)*buff(4)
      endif
!
! find local y-axis as common perpendicular to ab and cd
! if indeterminate take perpendicular to ab and space z-axis
! if still indeterminate take perpendicular to ab and space x-axis
!
      cosg= buff(1)*p(1,3)+buff(2)*p(2,3)+buff(3)*p(3,3)
      cosg= min( one,cosg)
      cosg= max(-one,cosg)
!
! modified rotation testing.
! this fix cures the small angle problem.
!
      p(1,2)= buff(3)*p(2,3)-buff(2)*p(3,3)
      p(2,2)= buff(1)*p(3,3)-buff(3)*p(1,3)
      p(3,2)= buff(2)*p(1,3)-buff(1)*p(2,3)
      if( abs(cosg).gt.pt9) then
         sing= sqrt(p(1,2)*p(1,2)+p(2,2)*p(2,2)+p(3,2)*p(3,2))
      else
         sing= sqrt(one-cosg*cosg)
      endif
      if( abs(cosg).le.pt9 .or. sing.ge.tenm12) then
         buff(4)= one/sing
         p(1,2)= p(1,2)*buff(4)
         p(2,2)= p(2,2)*buff(4)
         p(3,2)= p(3,2)*buff(4)
      else
         i=3
         if( abs(p(1,3)).le.pt7) i=1
         buff(4) = p(i,3)*p(i,3)
         buff(4) = min( one,buff(4))
         buff(4) = sqrt(one-buff(4))
         if(buff(4).ne.zer) buff(4)= one/buff(4)
         if( abs(p(1,3)).le.pt7) then
            p(1,2)= zer
            p(2,2)= p(3,3)*buff(4)
            p(3,2)=-p(2,3)*buff(4)
         else
            p(1,2)= p(2,3)*buff(4)
            p(2,2)=-p(1,3)*buff(4)
            p(3,2)= zer
         endif
      endif
!
! find direction cosines of local x-axes
!
      p(1,1)= p(2,2)*p(3,3)-p(3,2)*p(2,3)
      p(2,1)= p(3,2)*p(1,3)-p(1,2)*p(3,3)
      p(3,1)= p(1,2)*p(2,3)-p(2,2)*p(1,3)
!
! find coordinates of c relative to local axes at a
!
      buff(1:3)= p34(1:3,1)-p12(1:3,1)
      acx = buff(1)*p(1,1)+buff(2)*p(2,1)+buff(3)*p(3,1)
      acy = buff(1)*p(1,2)+buff(2)*p(2,2)+buff(3)*p(3,2)
      acz = buff(1)*p(1,3)+buff(2)*p(2,3)+buff(3)*p(3,3)
!
! set acy= 0  if close
!
      if( abs(acy).le.acycut) then
         acy = zer
         acy2= zer
      else
         acy2= acy*acy
      endif
!
! get bra density
!      
   ji= 1
   do i=1,nga
      buff(1)= ex(is1+i-1)
      do j=1,ngb
         buff(2)= ex(js1+j-1)
         buff(3)= buff(1)+buff(2)
         buff(4)= one/buff(3)
         buff(5)= buff(1)*buff(4)
         buff(6)= one-buff(5)
         buff(7)= buff(5)*buff(2)
         tx12(ji)= buff(3)
         tx21(ji)= buff(4)*pt5
         ty02(ji)= buff(6)*rab
         ty01(ji)= ty02(ji)-rab
         buff(8)= r12*buff(7)
         if(buff(8).gt.cux) then
            ismlp(ji)=2
            ji=ji+1
            cycle
         endif
         buff(9)= buff(4)* exp(-buff(8))
         buff(10)= buff(9)*cmax(is1+i-1)*cmax(js1+j-1)
         ismlp(ji)=0
         if(buff(10).le.ei1) ismlp(ji)=1
         if(buff(10).le.ei2) ismlp(ji)=2
         buff(9)= pito52*buff(9)   

         d00p(ji)= buff(9)*cs(is1+i-1)*cs(js1+j-1)
         ji=ji+1
      enddo
   enddo

!
! begin McMurchie-Davidson ERIs
!
    do k=1,ngc
      x03= ex(ks1+k-1)
        do l=1,ngd
          x04= ex(ls1+l-1)            
          x34= x03+x04
          x43= one/x34
          y03= x03*x43
          y04= one-y03
          y34= y03*x04
          r34y34= r34*y34
          if(r34y34.gt.cux) cycle
          e34= x43* exp(-r34y34)
          tst= e34*cmax(ks1+k-1)*cmax(ls1+l-1)
          if(tst.le.ei2) cycle
          ismlq= 0
          if(tst.le.ei1) ismlq= 1
          cq = rcd*y04
          cqx= cq*sing
          cqz= cq*cosg
          aqx= acx+cqx
          aqx2=aqx*aqx
          aqxy=aqx*acy
          aqz= acz+cqz
          qps= aqx2+acy2
          sq(1)= e34*cs(ks1+k-1)*cd(ls1+l-1)
          sq(2)= e34*cp(ks1+k-1)*cd(ls1+l-1)
          sq(3)= e34*cd(ks1+k-1)*cd(ls1+l-1) 

          fq0(1)= zer
          fq1(1:2)= zer
          fqd2(1:3)= zer

          do i=1,ngangb
            isml= ismlq+ismlp(i)
            if(isml.ge.2) cycle
            buff(1)= tx12(i)
            buff(2)= ty02(i)
            buff(3)= d00p(i)
            buff(4)= one/(buff(1)+x34)
            buff(5)= buff(2)-aqz
            buff(6)= buff(5)*buff(5)
            buff(7)= buff(1)*x34*buff(4)
            if(lrint) then
              buff(8)= emu2/(emu2+buff(7))
              buff(7)= buff(7)*buff(8)
              buff(3)= buff(3)*sqrt(buff(8))
            endif
            buff(9)=(buff(6)+qps)*buff(7)
            buff(7)= buff(7)+buff(7)
            n=2
  !
  !     fm(t) evaluation
  !
            if(buff(9).le.tmax) then
            buff(10)= buff(9)*rfinc(n)
            ip= nint(buff(10))
            buff(1)=    fgrid(4,ip,n) *buff(10)
            buff(1)=(buff(1)+fgrid(3,ip,n))*buff(10)
            buff(1)=(buff(1)+fgrid(2,ip,n))*buff(10)
            buff(1)=(buff(1)+fgrid(1,ip,n))*buff(10)
            buff(1)= buff(1)+fgrid(0,ip,n)
            buff(10)= buff(9)*rxinc
            ip= nint(buff(10))
            et=    xgrid(4,ip) *buff(10)
            et=(et+xgrid(3,ip))*buff(10)
            et=(et+xgrid(2,ip))*buff(10)
            et=(et+xgrid(1,ip))*buff(10)
            et= et+xgrid(0,ip)

            fqd(n)= buff(1)
            t2= buff(9)+buff(9)
            fqd(1)=(t2*fqd(2)+et)*rmr(2)
            fqd(0)=(t2*fqd(1)+et)*rmr(1)

            fqf= buff(3)*sqrt(buff(4))
              fqd(0)= fqd(0)*fqf
            fqf= fqf*buff(7)
              fqd(1)= fqd(1)*fqf
            fqf= fqf*buff(7)
              fqd(2)= fqd(2)*fqf

           else
            buff(2)= one/buff(9)
            fqd(0)= buff(3)*sqrt(pi4*buff(2)*buff(4))
            rox= buff(7)*buff(2)
            fqf= pt5*rox
              fqd(1)= fqd(0)*fqf
            fqf= fqf+rox
              fqd(2)= fqd(1)*fqf
           endif

           fq0(1)= fq0(1)+fqd(0)
           fq1(1)= fq1(1)+fqd(1)
           fq1(2)= fq1(2)+fqd(1)*buff(5)
           fqd2(1)= fqd2(1)+fqd(2)
           fqd2(2)= fqd2(2)+fqd(2)*buff(5)
           fqd2(3)= fqd2(3)+fqd(2)*buff(6)
          enddo

          xmd2= x43 *0.5d+00
          xmd1= xmd2*sq(1)
          xmdt= xmd2*xmd1
          xmd2=-xmd1*y03

          buff(1)= +fq0(1)*xmd1
          buff(2)= +fq0(1)*y03*y03*sq(1)
          r00(1:2)= r00(1:2)+buff(1:2)

          buff(2)= -fq1(1)*aqx
          buff(3)= -fq1(1)*acy
          buff(4)= +fq1(2)
          r01(1:3)= r01(1:3)+ buff(2:4) *xmd2

          !d functions
          buff(1)= +(fqd2(1)*aqx2-fq1(1))
          buff(2)= +(fqd2(1)*acy2-fq1(1))
          buff(3)= +(fqd2(3)     -fq1(1))
          buff(4)= + fqd2(1)*aqxy
          buff(5)= - fqd2(2)*aqx
          buff(6)= - fqd2(2)*acy
          r03(1:6)=r03(1:6)+ buff(1:6) * xmdt

        enddo
    enddo 

    buff(1)= rcd*sing
    buff(2)= rcd*cosg

    c3(1)=+r03(1)
    c3(2)=+r03(4)
    c3(3)=+r03(5)
    c3(4)=+r03(2)
    c3(5)=+r03(6)
    c3(6)=+r03(3)

    c2(1)=+r01(1)
    c2(2)=+r01(2)
    c2(3)=+r01(3)   

    c1(1)=+r00(1)
    c1(2)=+r00(2)

    gpople( 1) =  +c3(1) + c1(1) + (+c2(1)+c2(1)+c1(2)*buff(1) ) * buff(1)
    gpople( 2) =  +c3(4)+c1(1)
    gpople( 3) =  +c3(6)+c1(1)+(+c2(3)+c2(3)+c1(2)*buff(2))*buff(2)
    gpople( 4) =  +c3(2)+c2(2)*buff(1)
    gpople( 5) =  +c3(3)+c2(3)*buff(1)+(+c2(1)+c1(2)*buff(1))*buff(2)
    gpople( 6) =  +c3(5)+c2(2)*buff(2)

!
!jms  now, the transpose of p to be used for computational efficiency
!
      do 195 j=1,2
         do 195 i=j+1,3
            buff(1)= p(i,j)
            p(i,j)= p(j,i)
            p(j,i)= buff(1)
  195 continue

      do 110 i=1,3
         q(1,i)= p(1,i)*p(1,i)
         q(2,i)= p(2,i)*p(2,i)
         q(3,i)= p(3,i)*p(3,i)
         q(4,i)= p(1,i)*p(2,i)*two
         q(5,i)= p(1,i)*p(3,i)*two
         q(6,i)= p(2,i)*p(3,i)*two
  110 continue
      do 120 i=4,5
         j=i-2
         q(1,i)= p(1,1)*p(1,j)
         q(2,i)= p(2,1)*p(2,j)
         q(3,i)= p(3,1)*p(3,j)
         q(4,i)= p(1,1)*p(2,j)+p(2,1)*p(1,j)
         q(5,i)= p(1,1)*p(3,j)+p(3,1)*p(1,j)
         q(6,i)= p(2,1)*p(3,j)+p(3,1)*p(2,j)
  120 continue
         q(1,6)= p(1,2)*p(1,3)
         q(2,6)= p(2,2)*p(2,3)
         q(3,6)= p(3,2)*p(3,3)
         q(4,6)= p(1,2)*p(2,3)+p(2,2)*p(1,3)
         q(5,6)= p(1,2)*p(3,3)+p(3,2)*p(1,3)
         q(6,6)= p(2,2)*p(3,3)+p(3,2)*p(2,3)

      do 130 i=4,6
         do 130 j=1,6
            q(j,i)= q(j,i)*sqrt3
  130 continue

         gpople_final(1) = gpople(1)*q(1,1)+gpople(2)*q(2,1)+gpople(3)*q(3,1)+gpople(4)*q(4,1)+gpople(5)*q(5,1)+gpople(6)*q(6,1)
         gpople_final(2) = gpople(1)*q(1,2)+gpople(2)*q(2,2)+gpople(3)*q(3,2)+gpople(4)*q(4,2)+gpople(5)*q(5,2)+gpople(6)*q(6,2)
         gpople_final(3) = gpople(1)*q(1,3)+gpople(2)*q(2,3)+gpople(3)*q(3,3)+gpople(4)*q(4,3)+gpople(5)*q(5,3)+gpople(6)*q(6,3)
         gpople_final(4) = gpople(1)*q(1,4)+gpople(2)*q(2,4)+gpople(3)*q(3,4)+gpople(4)*q(4,4)+gpople(5)*q(5,4)+gpople(6)*q(6,4)
         gpople_final(5) = gpople(1)*q(1,5)+gpople(2)*q(2,5)+gpople(3)*q(3,5)+gpople(4)*q(4,5)+gpople(5)*q(5,5)+gpople(6)*q(6,5)
         gpople_final(6) = gpople(1)*q(1,6)+gpople(2)*q(2,6)+gpople(3)*q(3,6)+gpople(4)*q(4,6)+gpople(5)*q(5,6)+gpople(6)*q(6,6)

! starting digestion of the TWOEIs into the Fock build; this routine is for sssd integrals only 
! loop through l only, everything else is set 1
! apply permutational symmetry

    minl=1
    maxl=6

    ii1 = kloc(ish)
    jj1 = kloc(jsh)
    kk1 = kloc(ksh)
    locl = kloc(lsh)-minl

    i2 = ii1
    j2 = jj1
    if (ii1.lt.jj1) then ! sort <ij|
       i2 = jj1
       j2 = ii1
    endif

    do l=minl,maxl !1,6 because d functions

       buff(1) = gpople_final(l)

       if(abs(buff(1)).lt.cutoff) cycle ! goto 300

       ll1 = l + locl
       k2 = kk1
       l2 = ll1
       if (k2.lt.l2) then ! sort |kl>
          k2 = ll1
          l2 = kk1
       endif
       ii = i2
       jj = j2
       kk = k2
       ll = l2
       if (ii.lt.kk) then ! sort <ij|kl>
          ii = k2
          jj = l2
          kk = i2
          ll = j2
       else if (ii.eq.kk.and.jj.lt.ll) then ! sort <ij|il>
          jj = l2
          ll = j2
       endif
       ii2 = ia(ii)
       jj2 = ia(jj)
       kk2 = ia(kk)
       ij = ii2 + jj
       ik = ii2 + kk
       il = ii2 + ll
       jk = jj2 + kk
       jl = jj2 + ll
       kl = kk2 + ll
       if (jj.lt.kk) jk = kk2 + jj
       if (jj.lt.ll) jl = ia(ll) + jj

! account for identical permutations
       if(ii.eq.jj) buff(1) = buff(1)*half
       if(kk.eq.ll) buff(1) = buff(1)*half
       if(ii.eq.kk.and.jj.eq.ll) buff(1) = buff(1)*half
       buff(2) = buff(1)*xval1
       buff(3) = buff(1)*xval4

       buff(4) =  buff(3)*da(kl)
       buff(5) =  buff(3)*da(ij)
       buff(6) = -buff(2)*da(jl)
       buff(7) = -buff(2)*da(ik)
       buff(8) = -buff(2)*da(jk)
       buff(9) = -buff(2)*da(il)
!$omp atomic
       fa(ij) = fa(ij) + buff(4)
!$omp atomic
       fa(kl) = fa(kl) + buff(5)
!$omp atomic
       fa(ik) = fa(ik) + buff(6)
!$omp atomic
       fa(jl) = fa(jl) + buff(7)
!$omp atomic
       fa(il) = fa(il) + buff(8)
!$omp atomic
       fa(jk) = fa(jk) + buff(9)

    enddo
enddo
!$omp end target teams distribute parallel do
end
