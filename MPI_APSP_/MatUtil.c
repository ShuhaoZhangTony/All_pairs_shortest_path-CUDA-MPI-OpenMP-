#include "MatUtil.h"
#include <mpi.h>

void GenMatrix(int *mat, const size_t N)
{
	for(int i = 0; i < N*N; i ++)
		mat[i] = rand()%32 - 1;

}

inline int min(int a, int b) {
	return ( a!= -1 && a < b) ? a :b;
}

bool CmpArray(const int *l, const int *r, const size_t eleNum)
{
	for(int i = 0; i < eleNum; i ++)
		if(l[i] != r[i])
		{
			printf("ERROR: l[%d] = %d, r[%d] = %d\n", i, l[i], i, r[i]);
			return false;
		}
	return true;
}


/*
	Sequential (Single Thread) APSP on CPU.
*/
void ST_APSP(int *mat, const size_t N)
{
	for(int k = 0; k < N; k ++)
		for(int i = 0; i < N; i ++)
			for(int j = 0; j < N; j ++)
			{
				int i0 = i*N + j;
				int i1 = i*N + k;
				int i2 = k*N + j;
				if(mat[i1] != -1 && mat[i2] != -1)
					mat[i0] = (mat[i0] != -1 && mat[i0] < mat[i1] + mat[i2]) ?
					  mat[i0] : (mat[i1] + mat[i2]);
			}
}

/*
	Parallel (Multiple Thread) APSP on CPU.
*/



void MT_APSP(int *part, MPI_Comm comm, int myrank, const size_t N, int p) {

  int s = N/p;
  int root, offset;
  int *temp = (int*)malloc(sizeof(int)*N);
        //printf("Local: %d\n",myrank);
        //printMatrix(part,N,p);

  for (int k = 0; k < N; k++) {
    root = k/s;
    if (myrank == root) {
      offset = k - myrank*s;
      for (int j = 0; j < N; j++) 
                                temp[j] = part[offset*N + j];
    } 
                
    MPI_Bcast(temp, N, MPI_INT, root, comm);
    for(int i = 0; i < s; i ++)
                        for(int j = 0; j < N; j ++) {
                                int i0 = i*N + j;
                                int i1 = i*N + k;
                                if (part[i1] != -1 && temp[j] != -1)
                                        part[i0] = min(part[i0], part[i1] + temp[j]);
                        }
  }
}

