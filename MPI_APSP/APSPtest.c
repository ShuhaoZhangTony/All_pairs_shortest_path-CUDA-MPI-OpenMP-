#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include "MatUtil.h"


int main(int argc, char *argv[]) {
	if(argc != 2)
	{
		printf("Missing Argument\n");
		exit(-1);
	}
	
	int i, j;
	int *mat, *ref;
	int P, myrank;
	size_t N = atoi(argv[1]); //matrix size
	struct timeval tv1,tv2;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &P);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	
	
	if(myrank == 0) {
		// generate a random matrix.
		printf("input Size is %d; number of process is %d \n",N,P);
		mat = (int*)malloc(sizeof(int)*N*N);
		GenMatrix(mat, N);

		// compute the reference result.
		ref = (int*)malloc(sizeof(int)*N*N);
		memcpy(ref, mat, sizeof(int)*N*N);
		
		gettimeofday(&tv1, NULL);
		ST_APSP(ref, N);
		gettimeofday(&tv2, NULL);
		printf("Sequential time = %ld usecs\n", 
				(tv2.tv_sec-tv1.tv_sec)*1000000+tv2.tv_usec-tv1.tv_usec);  
		
		// compute your results
		
	}
		int strip = N/P;
		int *part = (int*)malloc(sizeof(int)*N*strip); /*array holding the part for this processor*/
	// Scatter data to all processors
	MPI_Barrier(MPI_COMM_WORLD);
	    if (myrank == 0) {
                gettimeofday(&tv1,NULL);
        }
	MPI_Scatter(mat, N*strip, MPI_INT, part, N*strip, MPI_INT, 0, MPI_COMM_WORLD);

	// Compute matrix in parallel
	MT_APSP (part, MPI_COMM_WORLD, myrank, N, P);
	
	//Gather the results
	MPI_Gather(part, N*strip, MPI_INT, mat, N*strip, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	if(myrank == 0) {
		gettimeofday(&tv2, NULL);
		printf("Parallel time = %ld usecs\n\n",
				(tv2.tv_sec-tv1.tv_sec)*1000000+tv2.tv_usec-tv1.tv_usec);
#ifdef test	
		// compare your result with reference result
		if(CmpArray(mat, ref, N*N))
			printf("Your result is correct.\n");
		else
			printf("Your result is wrong.\n");
#endif			
		free(mat);
		free(ref);
		
	}
	free(part);
	MPI_Finalize();
	return 0;
}
