// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <iomanip>
// includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
#include <APSP_CPU.h>
#define MAX_THREADS_PER_BLOCK 1024
#define TILE_DIM 16 
#define BLOCK_ROWS 8 
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
using namespace std;
__global__ void
CUDA_APSP(int *d_mat,int k,int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int	j = blockIdx.y * blockDim.y + threadIdx.y;
	 if(i<N&&j<N){
		int i0 = i*N + j;
		int i1 = i*N + k;
		int i2 = k*N + j;
		if(d_mat[i1] != -1 && d_mat[i2] != -1)
			d_mat[i0] = 
				(d_mat[i0] != -1 && d_mat[i0] < d_mat[i1] + d_mat[i2]) ? d_mat[i0] : (d_mat[i1] + d_mat[i2]);
		//__syncthreads();
	}
}
void CUDA_APSP_base(dim3 grid, dim3 threads, int *d_mat,int N){
	int k;
	for(k=0;k<N;k++)
		CUDA_APSP<<<grid,threads>>>(d_mat,k,N);
}
__global__ void
CUDA_APSP_coalcesing(int *d_mat,int *d_mat_trans,int k,int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int	j = blockIdx.y * blockDim.y + threadIdx.y;
	 if(i<N&&j<N){
		int i0 = i*N + j;//all
		int i0_trans=j*N +i;
		//int i1 = i*N + k;//all k column, jump between rows
		int i1_trans=k*N + i;//all k row, jump between colums... used for trans matrix
		int i2 = k*N + j;//all k row, jump between columns
		
		if(d_mat_trans[i1_trans] != -1 && d_mat[i2] != -1){
			d_mat[i0] = 
				(d_mat[i0] != -1 && d_mat[i0] < d_mat_trans[i1_trans] + d_mat[i2]) ? d_mat[i0] : (d_mat_trans[i1_trans] + d_mat[i2]);
			d_mat_trans[i0_trans]=d_mat[i0];
		}
		//__syncthreads();
	}
}
__global__ void transposeDiagonal(int *odata, 
 int *idata, int width, int height, int nreps) 
{ 
	 __shared__ float tile[TILE_DIM][TILE_DIM+1]; 
 
	 int blockIdx_x, blockIdx_y; 
 
	 // diagonal reordering 
	 if (width == height) { 
		 blockIdx_y = blockIdx.x; 
		 blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x; 
	 } else { 
		 int bid = blockIdx.x + gridDim.x*blockIdx.y; 
		 blockIdx_y = bid%gridDim.y; 
		 blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x; 
	 } 
 
	 int xIndex = blockIdx_x*TILE_DIM + threadIdx.x; 
	 int yIndex = blockIdx_y*TILE_DIM + threadIdx.y; 
	 int index_in = xIndex + (yIndex)*width; 
 
	 xIndex = blockIdx_y*TILE_DIM + threadIdx.x; 
	 yIndex = blockIdx_x*TILE_DIM + threadIdx.y; 
	 int index_out = xIndex + (yIndex)*height; 
 
	 for (int r=0; r < nreps; r++) { 
		 for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) { 
			 tile[threadIdx.y+i][threadIdx.x] = 
			 idata[index_in+i*width]; 
		 } 
 
	 __syncthreads(); 
 
	 for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) { 
		 odata[index_out+i*height] = 
			 tile[threadIdx.x][threadIdx.y+i]; 
		 } 
 	}	 	
} 
void CUDA_APSP_coalcesing(dim3 grid, dim3 threads,dim3 grid_trans,dim3 threads_trans,int *d_mat,int N){
	int mem_size=sizeof(int)*N*N;
	int *d_mat_trans2;
    	checkCudaErrors(cudaMalloc((void **) &d_mat_trans2, mem_size));
	transposeDiagonal<<<grid_trans,threads_trans>>>(d_mat_trans2,d_mat,N,N,1);
	int k;
	for(k=0;k<N;k++)
		CUDA_APSP_coalcesing<<<grid,threads>>>(d_mat,d_mat_trans2,k,N);	
    	checkCudaErrors(cudaFree(d_mat_trans2));
}
__global__ void
CUDA_APSP_SharedMemory(int *d_mat,int k,int N)
{
    extern __shared__ int s_mem[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int	j = blockIdx.y * blockDim.y + threadIdx.y;
	 if(i<N&&j<N){
		int i0 = i*N + j;
		int i1 = i*N + k;
		int i2 = k*N + j;
		if(threadIdx.y==0)
			s_mem[threadIdx.x]=d_mat[i1];
		__syncthreads();
		
	/*	if(d_mat[i1] != -1 && d_mat[i2] != -1)
			d_mat[i0] = 
				(d_mat[i0] != -1 && d_mat[i0] < d_mat[i1] + d_mat[i2]) ? d_mat[i0] : (d_mat[i1] + d_mat[i2]);
	*/
		if(s_mem[threadIdx.x] != -1 && d_mat[i2] != -1)
			d_mat[i0] = 
				(d_mat[i0] != -1 && d_mat[i0] < s_mem[threadIdx.x] + d_mat[i2]) ? d_mat[i0] : (s_mem[threadIdx.x] + d_mat[i2]);
		__syncthreads();
	}
}
__global__ void
CUDA_APSP_SharedMemory_double(int *d_mat,int k,int N)
{
    extern __shared__ int s_mem[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
         if(i<N&&j<N){
                int i0 = i*N + j;
                int i1 = i*N + k;
                int i2 = k*N + j;
		if(threadIdx.y==0)
	                s_mem[threadIdx.x]=d_mat[i1];
		else if(threadIdx.x==0)
	                s_mem[blockDim.x+threadIdx.y]=d_mat[i2];
                __syncthreads();

                if(s_mem[threadIdx.x] != -1 && s_mem[blockDim.x+threadIdx.y] != -1)
                        d_mat[i0] =
                                (d_mat[i0] != -1 && d_mat[i0] < s_mem[threadIdx.x] + s_mem[blockDim.x+threadIdx.y]) ? d_mat[i0] : (s_mem[threadIdx.x] + s_mem[blockDim.x+threadIdx.y]);
                __syncthreads();
        }
}

void CUDA_APSP_SharedMemory(dim3 grid, dim3 threads, int *d_mat,int N){
	int SizeS=sizeof(int)*threads.x;
	int k;
	for(k=0;k<N;k++)
		CUDA_APSP_SharedMemory<<<grid,threads,SizeS>>>(d_mat,k,N);	
}
void CUDA_APSP_SharedMemory_double(dim3 grid, dim3 threads, int *d_mat,int N){
	int SizeS=sizeof(int)*threads.x;
	int k;
	for(k=0;k<N;k++)
		CUDA_APSP_SharedMemory_double<<<grid,threads,SizeS>>>(d_mat,k,N);	
}
__global__ void
CUDA_APSP_Advanced(int *d_mat,int *d_mat_trans,int k,int N)
{ 
    extern __shared__ int s_mem[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int	j = blockIdx.y * blockDim.y + threadIdx.y;
	 if(i<N&&j<N){
		int i0 = i*N + j;//all
		int i0_trans=j*N +i;
		//int i1 = i*N + k;//all k column, jump between rows
		int i1_trans=k*N + i;//all k row, jump between colums... used for trans matrix
		int i2 = k*N + j;//all k row, jump between columns
		if(threadIdx.x==0)
			s_mem[threadIdx.y]=d_mat[i2];
		__syncthreads();

		if(d_mat_trans[i1_trans] != -1 &&  s_mem[threadIdx.y]!= -1){
			d_mat[i0] = 
				(d_mat[i0] != -1 && d_mat[i0] < d_mat_trans[i1_trans] +s_mem[threadIdx.y]) ? d_mat[i0] : (d_mat_trans[i1_trans] + s_mem[threadIdx.y]);
			d_mat_trans[i0_trans]=d_mat[i0];
		}
		__syncthreads();
	}
}

void CUDA_APSP_Advanced(dim3 grid, dim3 threads,dim3 grid_trans,dim3 threads_trans, int *d_mat,int N){
	/*colased*/
	int mem_size=sizeof(int)*N*N;
	int *d_mat_trans2;
    	checkCudaErrors(cudaMalloc((void **) &d_mat_trans2, mem_size));
	transposeDiagonal<<<grid_trans,threads_trans>>>(d_mat_trans2,d_mat,N,N,1);

	int SizeS=sizeof(int)*threads.x;
	int k;
	for(k=0;k<N;k++)
		CUDA_APSP_Advanced<<<grid,threads,SizeS>>>(d_mat,d_mat_trans2,k,N);	
	 checkCudaErrors(cudaFree(d_mat_trans2));
}

__global__ void
CUDA_APSP_Advanced_double(int *d_mat,int *d_mat_trans,int k,int N)
{ 
    extern __shared__ int s_mem[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int	j = blockIdx.y * blockDim.y + threadIdx.y;
	 if(i<N&&j<N){
		int i0 = i*N + j;//all
		int i0_trans=j*N +i;
		//int i1 = i*N + k;//all k column, jump between rows
		int i1_trans=k*N + i;//all k row, jump between colums... used for trans matrix
		int i2 = k*N + j;//all k row, jump between columns
		if(threadIdx.x==0)
			s_mem[threadIdx.y]=d_mat[i2]; 
		else if(threadIdx.y==0)
			s_mem[blockDim.x+threadIdx.x]=d_mat_trans[i1_trans];
		__syncthreads();
		if(s_mem[blockDim.x+threadIdx.x]!= -1 &&  s_mem[threadIdx.y]!= -1){
			d_mat[i0] = 
				(d_mat[i0] != -1 && d_mat[i0] < s_mem[blockDim.x+threadIdx.x] +s_mem[threadIdx.y]) ? d_mat[i0] : (s_mem[blockDim.x+threadIdx.x] + s_mem[threadIdx.y]);
			d_mat_trans[i0_trans]=d_mat[i0];
		}
		__syncthreads();
	}
}
void CUDA_APSP_Advanced_double(dim3 grid, dim3 threads,dim3 grid_trans,dim3 threads_trans,int *d_mat,int N){
	/*colased*/
	int mem_size=sizeof(int)*N*N;
	int *d_mat_trans2;
    	checkCudaErrors(cudaMalloc((void **) &d_mat_trans2, mem_size));
	transposeDiagonal<<<grid_trans,threads_trans>>>(d_mat_trans2,d_mat,N,N,1);

	int SizeS=2*sizeof(int)*threads.x;
	int k;
	for(k=0;k<N;k++)
		CUDA_APSP_Advanced_double<<<grid,threads,SizeS>>>(d_mat,d_mat_trans2,k,N);	
	 checkCudaErrors(cudaFree(d_mat_trans2));
}
double Computation(int flag,size_t N,size_t P,int test,double cpu_time,const int *h_mat=NULL,int *o_h_mat=NULL,const int *d_mat=NULL){
	int mem_size=sizeof(int)*N*N;
	int *o_h_mat2 = (int*)malloc(mem_size);//use in GPU : as output
	int *o_d_mat;//use in GPU : as input
	// setup execution parameters
	int num_of_blocks = 1;
	int num_of_threads_per_block = P;
	if(P>MAX_THREADS_PER_BLOCK){
		cout<<"Number of process per block must less than MAX_T:"<<MAX_THREADS_PER_BLOCK<<endl;
		exit(-1);
	}
	num_of_blocks=(int)ceil(N/(double)num_of_threads_per_block); 
    	dim3  grid(num_of_blocks, num_of_blocks, 1);
    	dim3  threads(num_of_threads_per_block, num_of_threads_per_block, 1);
	
	num_of_blocks=N/TILE_DIM; 
    	dim3  grid_trans(num_of_blocks, num_of_blocks, 1);
    	dim3  threads_trans(TILE_DIM, BLOCK_ROWS, 1);
	if(flag!=0){
    		checkCudaErrors(cudaMalloc((void **) &o_d_mat, mem_size));
		checkCudaErrors(cudaMemcpy(o_d_mat, d_mat, mem_size,
                               cudaMemcpyDeviceToDevice));						   						   	
	}else{
		memcpy(o_h_mat,h_mat,mem_size);
	//	cout << "num of blocks are: "<< num_of_blocks<<" num of threads per block: "<< 
	//	num_of_threads_per_block<<endl;
	}
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsed_time_ms;
	cudaEventRecord(start,0);
	if(!flag){ //work on CPU
		ST_APSP(o_h_mat, N);	
	}else{
	 if(flag==1){	     //work on GPU
		CUDA_APSP_base(grid,threads,o_d_mat,N);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemcpy(o_h_mat2, o_d_mat, mem_size,
                               cudaMemcpyDeviceToHost));	
	}else if(flag==2){
		CUDA_APSP_coalcesing(grid,threads,grid_trans,threads_trans,o_d_mat,N);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemcpy(o_h_mat2, o_d_mat, mem_size,
                               cudaMemcpyDeviceToHost));	
	}else if(flag==3){
		CUDA_APSP_SharedMemory(grid,threads,o_d_mat,N);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemcpy(o_h_mat2, o_d_mat, mem_size,
                               cudaMemcpyDeviceToHost));	
	}else if(flag==4){
		CUDA_APSP_SharedMemory_double(grid,threads,o_d_mat,N);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemcpy(o_h_mat2, o_d_mat, mem_size,
                               cudaMemcpyDeviceToHost));	
	}else if(flag==5){
		CUDA_APSP_Advanced(grid,threads,grid_trans,threads_trans,o_d_mat,N);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemcpy(o_h_mat2, o_d_mat, mem_size,
                               cudaMemcpyDeviceToHost));	
	}else if(flag==6){
		CUDA_APSP_Advanced_double(grid,threads,grid_trans,threads_trans,o_d_mat,N);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemcpy(o_h_mat2, o_d_mat, mem_size,
                               cudaMemcpyDeviceToHost));	
		}
	}
	float speedUp=1;
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms,start,stop);
	cout.flags(ios::internal);
	cout.setf(ios::fixed);
	cout <<"\n Time to calculate results on ";
	if(!flag){
		cout <<"CPU:						";
		cpu_time = elapsed_time_ms;
	}else{
		speedUp = cpu_time/elapsed_time_ms;
		if(flag==1)
			cout <<"GPU(baseline):					";
		else if(flag==2)
			cout <<"GPU(colasing only):				";
		else if(flag==3)
			cout <<"GPU(shared mem only):				";
		else if(flag==4)
			cout <<"GPU(share mem double):				";
		else if(flag==5)
			cout <<"GPU(Advanced):					";
		else if(flag==6)
			cout <<"GPU(Advanced,double):				";
	}
	cout<<setprecision(3)<<elapsed_time_ms<<" ms, ";
	if(flag!=0)
		cout<<speedUp<<" speed up than on CPU";
	cout<<endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	if(test){
		if(CmpArray(o_h_mat, o_h_mat2, N*N))
			printf("Your result is correct.\n");
		else
			printf("Your result is wrong.\n");
	}
	free(o_h_mat2);
	if(flag!=0)
    	checkCudaErrors(cudaFree(o_d_mat));
	return elapsed_time_ms;
}
void usage(int argc, char **argv) {
	fprintf(stderr, "Usage: %s <size of matrix> <num of process> <start command> <end command> <test>\n", argv[0]);
	fprintf(stderr, "\t<size of matrix>		  	- side size of matrix n (positive integer)\n");
	fprintf(stderr, "\t<num of process>			- number of thread per block(1 - 32)\n");
	fprintf(stderr, "\t<start  command>			- differnt setting on GPU(1-6)\n");
	fprintf(stderr, "\t<end    command>			- differnt setting on GPU(larger than start command, 1-6)\n");
	fprintf(stderr, "\t<test >				- differnt setting on GPU(larger than start command, 1-6)\n");
	exit(1);
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
/*Initial settings*/
	size_t N;
	size_t P;	
    	printf("%s Starting...\n\n", argv[0]);
	int devID = findCudaDevice(argc, (const char **)argv);
/*Get input*/
	if(argc != 6)
	{
		usage(argc,argv);
	}
	N = atoi(argv[1]);
	P = atoi(argv[2]);
	int cmd_s=atoi(argv[3]);
	int cmd_e=atoi(argv[4]);
	int test=atoi(argv[5]);
	printf("Size is %d,numP is %d,cmd_s is %d, cmd_e is %d, test is %d \n",N,P,cmd_s,cmd_e,test);
/*Allocate Data*/
	unsigned int mem_size=sizeof(int)*N*N;
	// allocate host memory
	int *h_mat = (int*)malloc(mem_size);//use in CPU : as input 
	int *o_h_mat = (int*)malloc(mem_size);//use in CPU : as input 
	GenMatrix(h_mat, N);	

    	int *d_mat;//use in GPU : as input
    	checkCudaErrors(cudaMalloc((void **) &d_mat, mem_size));
	checkCudaErrors(cudaMemcpy(d_mat, h_mat, mem_size,
                               cudaMemcpyHostToDevice));						   						   	
	double cpu_time=0;
/*Computation on HOST*/
 	cpu_time=Computation(0,N,P,0,cpu_time,h_mat,o_h_mat); //h_mat compute -> h_mat
/*Computation on GPU*/	
	for(int i=cmd_s;i<=cmd_e;i++){
		Computation(i,N,P,test,cpu_time,h_mat,o_h_mat,d_mat);//d_mat compute->ref_mat
	}
	free(h_mat);
	free(o_h_mat);
    	checkCudaErrors(cudaFree(d_mat));
    	cudaDeviceReset();
}
