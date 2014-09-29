#include "kernel.h"
#include "main.h"
#include <cmath>
#include <time.h> 
#include <thrust/copy.h>

__global__ void copy(const int *a, int *b, int length)
{
	int i = threadIdx.x+ (blockIdx.x * blockDim.x);
	if(i<length){
		b[i] = a[i];
	}
}

__global__ void scan(int *b, int d)
{
    int k = threadIdx.x+ (blockIdx.x * blockDim.x);
	int d2 = pow(2.0f,d-1);

    if(k>=d2)
		b[k] = b[k-d2] + b[k];
}



void prefix_sum(const int *a, int *b, int length)
{
	if(length<1)
		return;

	copy<<<length/BLOCK_SIZE, BLOCK_SIZE>>>(a,b,length);
	cudaDeviceSynchronize();
	int t= log((float)length)/log(2.0f)+1;
	for (int d =1 ;d<=t;d++){
		scan<<<length/BLOCK_SIZE, BLOCK_SIZE>>>(b,d);
		cudaDeviceSynchronize();
	}
}

__global__ void shared_scan_1block(int *b,int t)
{
    int k = threadIdx.x;
	int d2;
	__shared__ int bs[BLOCK_SIZE];

	if(k<BLOCK_SIZE)
		bs[k] = b[k];

	__syncthreads();
	for (int d =1 ;d<=t;d++){
		d2 = pow(2.0f,d-1);
		if(k>=d2)
			bs[k] = bs[k-d2] + bs[k];
		__syncthreads();
	}

	if(k<BLOCK_SIZE)
		b[k] = bs[k];
}

__global__ void shared_scan(int *b,int t)
{
    int k = threadIdx.x;
	int d2;
	__shared__ int bs[BLOCK_SIZE];
	__shared__ int tempSum;
	tempSum = 0;

	for(int i=0;i<=blockIdx.x;i++){
		int index = threadIdx.x+ (blockIdx.x * blockDim.x);
		if(k==0)
			bs[k] = b[index]+tempSum;
		else
			bs[k] = b[index];

		__syncthreads();
		for (int d =1 ;d<=t;d++){
			d2 = pow(2.0f,d-1);
			if(k>=d2)
				bs[k] = bs[k-d2] + bs[k];
			__syncthreads();
		}

		if(i==blockIdx.x)
			b[index] = bs[k];

		if(k == BLOCK_SIZE-1)
			tempSum = bs[k];

	}
}

void prefix_sum_shared(const int *a, int *b, int length)
{
	if(length<1)
		return;

	copy<<<length/BLOCK_SIZE, BLOCK_SIZE>>>(a,b,length);
	int t= log((float)length)/log(2.0f)+1;
	if(length<=BLOCK_SIZE)
		shared_scan_1block<<<1, length>>>(b,t);
	else
		shared_scan<<<length/BLOCK_SIZE, BLOCK_SIZE>>>(b,t);
}

__global__ void transform_to_boolean(const int *a, int *b, int length)
{
	int i = threadIdx.x+ (blockIdx.x * blockDim.x);
	if(i<length){
		b[i] = a[i]>0? 1:0;
	}
}

void scatter(const int *a, int *b, int length)
{
	if(length<1)
		return;

	transform_to_boolean<<<length/BLOCK_SIZE, BLOCK_SIZE>>>(a,b,length);
	int t= log((float)length)/log(2.0f)+1;
	if(length<=BLOCK_SIZE)
		shared_scan_1block<<<1, length>>>(b,t);
	else
		shared_scan<<<length/BLOCK_SIZE, BLOCK_SIZE>>>(b,t);
}

__global__ void compact(const int *a,int *temp, int *b, int length)
{
	int i = threadIdx.x+ (blockIdx.x * blockDim.x);
	if(i<length){
		if(a[i]!=0)
			b[temp[i]-1]=a[i];
	}
}

void stream_compact(const int *a, int *b, int length)
{
	int *temp = 0;
	cudaMalloc((void**)&temp, length * sizeof(int));
	scatter(a, temp,length);
	compact<<<length/BLOCK_SIZE, BLOCK_SIZE>>>(a,temp,b,length);
}

struct is_zero
{
   __host__ __device__
   bool operator()(const int x)
   {
     return x != 0;
   }
};


int main(int argc, char** argv)
{
	int size = 0;
	size = atoi(argv[1]); 
	int *a = new int[size];
	//size = 10;
	//int a[] = {0,0,3,4,0,6,6,7,0,1};
	int *b = new int[size];
	for(int i=0;i<size;i++)
		a[i] = 1;
	
	//clock_t time; 
	//time=clock(); 

	int *dev_a = 0;
    int *dev_b = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));

    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );	
	
	prefix_sum(dev_a, dev_b, size);
	//serial_prefix_sum(a,b,size);
	//prefix_sum_shared(dev_a,dev_b,size);
	//serial_scatter(a,b,size);
	//stream_compact(dev_a,dev_b,size);
	//thrust::copy_if(a, a+ size, b, is_zero());

    
	//time=clock()-time; 
	cudaDeviceSynchronize();
	cudaEventRecord( stop, 0 ); 
	cudaEventSynchronize( stop ); 
	cudaEventElapsedTime( &time, start, stop );
	cout<<time<<endl;
	cudaEventDestroy( start ); 
	cudaEventDestroy( stop );
    cudaStatus = cudaMemcpy(b, dev_b, size * sizeof(int), cudaMemcpyDeviceToHost);
	//for(int i=0;i<100;i++)
	//	cout<<b[i]<<endl;
	//cout<<double(time)<<endl; 

    return 0;
}
