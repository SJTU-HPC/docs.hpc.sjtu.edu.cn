      PROGRAM main
      USE OMP_LIB
      IMPLICIT NONE

      INCLUDE 'mpif.h'

      INTEGER :: myPE, totPEs, ierr, tid, nthreads

      CALL MPI_INIT(ierr)

      CALL MPI_COMM_RANK(MPI_COMM_WORLD, myPE, ierr)
      CALL MPI_COMM_SIZE(MPI_COMM_WORLD, totPEs, ierr)

!$OMP PARALLEL PRIVATE(tid, nthreads)
      tid = OMP_GET_THREAD_NUM()
      nthreads = OMP_GET_NUM_THREADS()

!      write(*,100) "Hello from thread ",tid," of ",         &
!     &            nthreads," in rank ",myPE," of ",totPEs
!$OMP END PARALLEL

  100 FORMAT(a,i3,a,i3,a,i3,a,i3)

      CALL MPI_FINALIZE(ierr)

      END
