#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include "MatUtil.h"
//#define Debug
int main(int argc, char **argv)
{
	if(argc != 3)
	{
		printf("Usage: test {N}\n");
		exit(-1);
	}
	size_t N = atoi(argv[1]);
	size_t npro = atoi(argv[2]);
	omp_set_num_threads(npro);
	 int iam = 0, np = 1;
	 #pragma omp parallel private(iam, np)
          {
                  np = omp_get_num_threads();
                  iam = omp_get_thread_num();
#ifdef Debug
                  printf("Hello from thread %d out of %d\n", iam, np);
#endif
          }
	struct timeval tv1,tv2;
	// generate a random matrix.
	printf("Size is %d,numP is %d\n",N,npro);
	int *mat = (int*)malloc(sizeof(int)*N*N);
	GenMatrix(mat, N);

	// compute the reference result.
	int *ref = (int*)malloc(sizeof(int)*N*N);
	memcpy(ref, mat, sizeof(int)*N*N);
	gettimeofday(&tv1,NULL);
	ST_APSP(ref, N);
	gettimeofday(&tv2,NULL);
	printf("Sequential time = %ld usecs\n",
                          (tv2.tv_sec-tv1.tv_sec)*1000000+tv2.tv_usec-tv1.tv_usec);


	// compute your results
	int *result = (int*)malloc(sizeof(int)*N*N);
	memcpy(result, mat, sizeof(int)*N*N);
//	ST_APSP(result, N);

	gettimeofday(&tv1,NULL);
        OMP_APSP(result,N);
        gettimeofday(&tv2,NULL);
        printf("OpenMp time = %ld usecs\n",
                          (tv2.tv_sec-tv1.tv_sec)*1000000+tv2.tv_usec-tv1.tv_usec);

	// compare your result with reference result
	if(CmpArray(result, ref, N*N))
		printf("Your result is correct.\n");
	else
		printf("Your result is wrong.\n");
}
