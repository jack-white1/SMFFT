#include <cuda_fp16.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>

__global__ void axpy(int n, half a, half* x, half* y){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	half b = __hmul(a,x[i]);
	if (i < n){y[i] = __hadd(b,y[i]);}
}

__global__ void axpy(int n, float a, float* x, float* y){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n){y[i] = a*x[i] + y[i];}
}

__global__ void axpy(int n, double a, double* x, double* y){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n){y[i] = a*x[i] + y[i];}
}


int main() {
	int N = 1<<21;
	double *dx, *dy, *d_dx, *d_dy;
	float *sx, *sy, *d_sx, *d_sy;
	//half *hx, *hy, *d_hx, *d_hy;

	cudaEvent_t dstart, dstop, sstart, sstop, hstart, hstop;


	cudaEventCreate(&dstart);
	cudaEventCreate(&dstop);
	cudaEventCreate(&sstart);
	cudaEventCreate(&sstop);
	cudaEventCreate(&hstart);
	cudaEventCreate(&hstop);


	dx = (double*)malloc(N*sizeof(double));
	dy = (double*)malloc(N*sizeof(double));
	sx = (float*)malloc(N*sizeof(float));
	sy = (float*)malloc(N*sizeof(float));
	//hx = (half*)malloc(N*sizeof(half));
	//hy = (half*)malloc(N*sizeof(half));
	
	cudaMalloc(&d_dx, N*sizeof(double));
	cudaMalloc(&d_dy, N*sizeof(double));
	cudaMalloc(&d_sx, N*sizeof(float));
	cudaMalloc(&d_sy, N*sizeof(float));
	//cudaMalloc(&d_hx, N*sizeof(half));
	//cudaMalloc(&d_hy, N*sizeof(half));


	
	srand (static_cast <unsigned> (time(0)));
	for (int i = 0; i< N; i++){
		dx[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
		dy[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
		sx[i] = static_cast <float> (dx[i]);
		sy[i] = static_cast <float> (dy[i]);
		//hx[i] = __double2half(dx[i]);
		//hy[i] = __double2half(dy[i]);
	}

	cudaMemcpy(d_dx, dx, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dy, dy, N*sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(d_sx, sx, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sy, sy, N*sizeof(float), cudaMemcpyHostToDevice);

	cudaEventRecord(dstart);
	axpy<<<(N+255/256), 256>>>(N, static_cast <double> (2.0f), d_dx, d_dy);
	cudaEventRecord(dstop);

	cudaEventRecord(sstart);
	axpy<<<(N+255/256), 256>>>(N, 2.0f, d_sx, d_sy);
	cudaEventRecord(sstop);

	cudaMemcpy(dy, d_dy, N*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(sy, d_sy, N*sizeof(float), cudaMemcpyDeviceToHost);

	double dmaxError = 0;
	float smaxError = 0;
	for (int i = 0; i < N; i++){
		dmaxError = max(dmaxError, abs(dy[i]));
		smaxError = max(smaxError, abs(sy[i]));
	}

	cudaEventSynchronize(dstop);

	float d_milliseconds = 0;
	float s_milliseconds = 0;
	cudaEventElapsedTime(&d_milliseconds, dstart, dstop);
	cudaEventElapsedTime(&s_milliseconds, sstart, sstop);

	printf("Max error double precision: %f, took: %fms\n", dmaxError, d_milliseconds);
	printf("Max error single precision: %f, took: %fms\n", smaxError, s_milliseconds);

	//cudaFree(d_hx);
	//cudaFree(d_hy);
	cudaFree(d_sx);
	cudaFree(d_sy);
	cudaFree(d_dx);
	cudaFree(d_dy);
	//free(hx);
	//free(hy);
	free(sx);
	free(sy);
	free(dx);
	free(dy);

}
