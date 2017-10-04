#include <stdio.h>
#include <mpi.h>

int main (int argc, char *argv[]) {
  int rank, size;
  char  hnbuf[64];
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank); /* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size); /* get number of processes */
  memset(hnbuf, 0, sizeof(hnbuf));
  (void)gethostname(hnbuf, sizeof(hnbuf));
  printf( "%d of %d on %s\n", rank, size, hnbuf );
  MPI_Finalize();
  return 0;
}
