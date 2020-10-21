#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "debug.h"
#include "timer.h"
#include "utils_cuda.h"
#include <stdio.h>

#include "debug.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
//#include <helper_cuda.h>
//#include <helper_functions.h>



#define NREUSES 100
#define NCUDABLOCKS 1000

#include "SM_FFT_parameters.cuh"

int device = 0;

//code from nvidia developer forum start
/*
  Copyright (c) 2016, Norbert Juffa
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  1. Redistributions of source code must retain the above copyright
	 notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
	 notice, this list of conditions and the following disclaimer in the
	 documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
__forceinline__ __device__ float raw_sin(float a)
{
	float r;
	asm("sin.approx.ftz.f32 %0,%1;" : "=f"(r) : "f"(a));
	return r;
}

__forceinline__ __device__ float raw_cos(float a)
{
	float r;
	asm("cos.approx.ftz.f32 %0,%1;" : "=f"(r) : "f"(a));
	return r;
}

__forceinline__ __device__ float copysignf_pos(float a, float b)
{
	float r;
	r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
	return r;
}

/* Use 'sf' suffix as per proposal N2016 in ISO/IEC JTC1 SC22 WG14 */
__device__ __inline__ void my_sincossf(half ah, half* sp, half* cp)
{
	float a, i, j, r, s, c;
	const float BIAS_ADJ = 8.8e-8f;

	a = __half2float(ah);

#if MORE_ACCURATE // maximal difference from correctly rounded result: 2 ulps   
	i = fmaf(a, 0.636619747f / 2, 12582912.0f); // 1/pi, 0x1.8p+23
	j = i - 12582912.0f; // 0x1.8p+23
	r = fmaf(j, -3.14159203e+0f, a); // -0x1.921fb0p+01 // pi_high
	r = fmaf(j, -6.27832947e-7f, r); // -0x1.5110b4p-21 // pi_low
	r = r + copysignf_pos(BIAS_ADJ, r); // correct bias of trig func intrinsics
	c = raw_cos(r);
	s = raw_sin(r);
	if (__float_as_int(i) & 1) {
		s = 0.0f - s; // don't change "sign" of NaNs or create negative zeros
		c = 0.0f - c; // don't change "sign" of NaNs or create negative zeros
	}
#else // maximal difference from correctly rounded result: 3 ulps 
	i = fmaf(a, 0.636619747f / 4, 12582912.0f); // 1/(2*pi), 0x1.8p+23
	j = i - 12582912.0f; // 0x1.8p+23
	r = fmaf(j, -6.28318405e+0f, a); // -0x1.921fb0p+02 // 2pi_high
	r = fmaf(j, -1.25566589e-6f, r); // -0x1.5110b4p-20 // 2pi_low
	r = r + copysignf_pos(BIAS_ADJ, r); // correct bias of trig func intrinsics
	c = raw_cos(r);
	s = raw_sin(r);
#endif
	* sp = __float2half_rn(s);
	*cp = __float2half_rn(c);
}

//end of code from nvidia developer forum



__device__ __inline__ half2 Get_W_value(int N, int m) {
	half2 ctemp;
	my_sincossf(__float2half(-6.283185308f) * __hdiv((half)m, (half)N), &ctemp.y, &ctemp.x);
	return(ctemp);
}

__device__ __inline__ half2 Get_W_value_inverse(int N, int m) {
	half2 ctemp;
	my_sincossf(__float2half(6.283185308f) * __hdiv((half)m, (half)N), &ctemp.y, &ctemp.x);
	return(ctemp);
}

__device__ __inline__ half shfl(half* value, int par) {
#if (CUDART_VERSION >= 9000)
	return(__shfl_sync(0xffffffff, (*value), par));
#else
	return(__shfl((*value), par));
#endif
}

__device__ __inline__ half shfl_xor(half* value, int par) {
#if (CUDART_VERSION >= 9000)
	return(__shfl_xor_sync(0xffffffff, (*value), par));
#else
	return(__shfl_xor((*value), par));
#endif
}

__device__ __inline__ half shfl_down(half* value, int par) {
#if (CUDART_VERSION >= 9000)
	return(__shfl_down_sync(0xffffffff, (*value), par));
#else
	return(__shfl_down((*value), par));
#endif
}

__device__ __inline__ void reorder_4_register(half2* A_DFT_value, half2* B_DFT_value, half2* C_DFT_value, half2* D_DFT_value) {
	half2 Af2temp, Bf2temp, Cf2temp, Df2temp;
	unsigned int target = (((unsigned int)__brev((threadIdx.x & 3))) >> (30)) + 4 * (threadIdx.x >> 2);
	Af2temp.x = shfl(&(A_DFT_value->x), target);
	Af2temp.y = shfl(&(A_DFT_value->y), target);
	Bf2temp.x = shfl(&(B_DFT_value->x), target);
	Bf2temp.y = shfl(&(B_DFT_value->y), target);
	Cf2temp.x = shfl(&(C_DFT_value->x), target);
	Cf2temp.y = shfl(&(C_DFT_value->y), target);
	Df2temp.x = shfl(&(D_DFT_value->x), target);
	Df2temp.y = shfl(&(D_DFT_value->y), target);
	__syncwarp();
	(*A_DFT_value) = Af2temp;
	(*B_DFT_value) = Bf2temp;
	(*C_DFT_value) = Cf2temp;
	(*D_DFT_value) = Df2temp;
}

__device__ __inline__ void reorder_8_register(half2* A_DFT_value, half2* B_DFT_value, half2* C_DFT_value, half2* D_DFT_value, int* local_id) {
	half2 Af2temp, Bf2temp, Cf2temp, Df2temp;
	unsigned int target = (((unsigned int)__brev(((*local_id) & 7))) >> (29)) + 8 * ((*local_id) >> 3);
	Af2temp.x = shfl(&(A_DFT_value->x), target);
	Af2temp.y = shfl(&(A_DFT_value->y), target);
	Bf2temp.x = shfl(&(B_DFT_value->x), target);
	Bf2temp.y = shfl(&(B_DFT_value->y), target);
	Cf2temp.x = shfl(&(C_DFT_value->x), target);
	Cf2temp.y = shfl(&(C_DFT_value->y), target);
	Df2temp.x = shfl(&(D_DFT_value->x), target);
	Df2temp.y = shfl(&(D_DFT_value->y), target);
	__syncwarp();
	(*A_DFT_value) = Af2temp;
	(*B_DFT_value) = Bf2temp;
	(*C_DFT_value) = Cf2temp;
	(*D_DFT_value) = Df2temp;
}

__device__ __inline__ void reorder_16_register(half2* A_DFT_value, half2* B_DFT_value, half2* C_DFT_value, half2* D_DFT_value, int* local_id) {
	half2 Af2temp, Bf2temp, Cf2temp, Df2temp;
	unsigned int target = (((unsigned int)__brev(((*local_id) & 15))) >> (28)) + 16 * ((*local_id) >> 4);
	Af2temp.x = shfl(&(A_DFT_value->x), target);
	Af2temp.y = shfl(&(A_DFT_value->y), target);
	Bf2temp.x = shfl(&(B_DFT_value->x), target);
	Bf2temp.y = shfl(&(B_DFT_value->y), target);
	Cf2temp.x = shfl(&(C_DFT_value->x), target);
	Cf2temp.y = shfl(&(C_DFT_value->y), target);
	Df2temp.x = shfl(&(D_DFT_value->x), target);
	Df2temp.y = shfl(&(D_DFT_value->y), target);
	__syncwarp();
	(*A_DFT_value) = Af2temp;
	(*B_DFT_value) = Bf2temp;
	(*C_DFT_value) = Cf2temp;
	(*D_DFT_value) = Df2temp;
}

__device__ __inline__ void reorder_32_register(half2* A_DFT_value, half2* B_DFT_value, half2* C_DFT_value, half2* D_DFT_value) {
	half2 Af2temp, Bf2temp, Cf2temp, Df2temp;
	unsigned int target = ((unsigned int)__brev(threadIdx.x)) >> (27);
	Af2temp.x = shfl(&(A_DFT_value->x), target);
	Af2temp.y = shfl(&(A_DFT_value->y), target);
	Bf2temp.x = shfl(&(B_DFT_value->x), target);
	Bf2temp.y = shfl(&(B_DFT_value->y), target);
	Cf2temp.x = shfl(&(C_DFT_value->x), target);
	Cf2temp.y = shfl(&(C_DFT_value->y), target);
	Df2temp.x = shfl(&(D_DFT_value->x), target);
	Df2temp.y = shfl(&(D_DFT_value->y), target);
	__syncwarp();
	(*A_DFT_value) = Af2temp;
	(*B_DFT_value) = Bf2temp;
	(*C_DFT_value) = Cf2temp;
	(*D_DFT_value) = Df2temp;
}

template<class const_params>
__device__ __inline__ void reorder_32(half2* s_input, half2* A_DFT_value, half2* B_DFT_value, half2* C_DFT_value, half2* D_DFT_value) {
	reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
}


template<class const_params>
__device__ __inline__ void reorder_64(half2* s_input, half2* A_DFT_value, half2* B_DFT_value, half2* C_DFT_value, half2* D_DFT_value) {
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x / const_params::warp;

	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);

	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	__syncthreads();
	unsigned int sm_store_pos = (local_id >> 4) + 2 * (local_id & 15) + warp_id * 132;
	s_input[sm_store_pos] = *A_DFT_value;
	s_input[sm_store_pos + 33] = *B_DFT_value;
	s_input[66 + sm_store_pos] = *C_DFT_value;
	s_input[66 + sm_store_pos + 33] = *D_DFT_value;

	// Read shared memory to get reordered input
	unsigned int sm_read_pos = (local_id & 1) * 32 + local_id + warp_id * 132;
	__syncthreads();
	*A_DFT_value = s_input[sm_read_pos];
	*B_DFT_value = s_input[sm_read_pos + 1];
	*C_DFT_value = s_input[sm_read_pos + 66];
	*D_DFT_value = s_input[sm_read_pos + 66 + 1];
}


template<class const_params>
__device__ __inline__ void reorder_128(half2* s_input, half2* A_DFT_value, half2* B_DFT_value, half2* C_DFT_value, half2* D_DFT_value) {
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x / const_params::warp;

	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);

	__syncwarp();
	unsigned int sm_store_pos = (local_id >> 3) + 4 * (local_id & 7) + warp_id * 132;
	s_input[sm_store_pos] = *A_DFT_value;
	s_input[sm_store_pos + 33] = *B_DFT_value;
	s_input[66 + sm_store_pos] = *C_DFT_value;
	s_input[66 + sm_store_pos + 33] = *D_DFT_value;

	// Read shared memory to get reordered input
	__syncwarp();
	unsigned int sm_read_pos = (local_id & 3) * 32 + local_id + warp_id * 132;
	*A_DFT_value = s_input[sm_read_pos];
	*B_DFT_value = s_input[sm_read_pos + 1];
	*C_DFT_value = s_input[sm_read_pos + 2];
	*D_DFT_value = s_input[sm_read_pos + 3];

	__syncwarp();
	reorder_4_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
}


template<class const_params>
__device__ __inline__ void reorder_256(half2* s_input, half2* A_DFT_value, half2* B_DFT_value, half2* C_DFT_value, half2* D_DFT_value) {
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x / const_params::warp;

	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);

	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	__syncthreads();
	unsigned int sm_store_pos = (local_id >> 2) + 8 * (local_id & 3) + warp_id * 132;
	s_input[sm_store_pos] = *A_DFT_value;
	s_input[sm_store_pos + 33] = *B_DFT_value;
	s_input[66 + sm_store_pos] = *C_DFT_value;
	s_input[66 + sm_store_pos + 33] = *D_DFT_value;

	// Read shared memory to get reordered input
	__syncthreads();
	unsigned int sm_read_pos = (local_id & 7) * 32 + local_id;
	*A_DFT_value = s_input[sm_read_pos + warp_id * 4 + 0];
	*B_DFT_value = s_input[sm_read_pos + warp_id * 4 + 1];
	*C_DFT_value = s_input[sm_read_pos + warp_id * 4 + 2];
	*D_DFT_value = s_input[sm_read_pos + warp_id * 4 + 3];

	__syncthreads();
	reorder_8_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value, &local_id);
}

template<class const_params>
__device__ __inline__ void reorder_512(half2* s_input, half2* A_DFT_value, half2* B_DFT_value, half2* C_DFT_value, half2* D_DFT_value) {
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x / const_params::warp;

	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);

	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	__syncthreads();
	unsigned int sm_store_pos = (local_id >> 1) + 16 * (local_id & 1) + warp_id * 132;
	s_input[sm_store_pos] = *A_DFT_value;
	s_input[sm_store_pos + 33] = *B_DFT_value;
	s_input[66 + sm_store_pos] = *C_DFT_value;
	s_input[66 + sm_store_pos + 33] = *D_DFT_value;

	// Read shared memory to get reordered input
	unsigned int sm_read_pos = (local_id & 15) * 32 + local_id + warp_id * 4;
	__syncthreads();
	*A_DFT_value = s_input[sm_read_pos + 0];
	*B_DFT_value = s_input[sm_read_pos + 1];
	*C_DFT_value = s_input[sm_read_pos + 2];
	*D_DFT_value = s_input[sm_read_pos + 3];

	__syncthreads();
	reorder_16_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value, &local_id);
}

template<class const_params>
__device__ __inline__ void reorder_1024(half2* s_input, half2* A_DFT_value, half2* B_DFT_value, half2* C_DFT_value, half2* D_DFT_value) {
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x / const_params::warp;

	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);

	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	__syncthreads();
	unsigned int sm_store_pos = (local_id >> 0) + 32 * (local_id & 0) + warp_id * 132;
	s_input[sm_store_pos] = *A_DFT_value;
	s_input[sm_store_pos + 33] = *B_DFT_value;
	s_input[66 + sm_store_pos] = *C_DFT_value;
	s_input[66 + sm_store_pos + 33] = *D_DFT_value;

	// Read shared memory to get reordered input
	unsigned int sm_read_pos = (local_id & 31) * 32 + local_id + warp_id * 4;
	__syncthreads();
	*A_DFT_value = s_input[sm_read_pos + 0];
	*B_DFT_value = s_input[sm_read_pos + 1];
	*C_DFT_value = s_input[sm_read_pos + 2];
	*D_DFT_value = s_input[sm_read_pos + 3];

	__syncthreads();
	reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
}

template<class const_params>
__device__ __inline__ void reorder_2048(half2* s_input, half2* A_DFT_value, half2* B_DFT_value, half2* C_DFT_value, half2* D_DFT_value) {
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x / const_params::warp;

	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);


	__syncthreads();
	//unsigned int sm_store_pos = (local_id>>0) + 32*(local_id&0) + warp_id*132;
	unsigned int sm_store_pos = local_id + warp_id * 132;
	s_input[sm_store_pos] = *A_DFT_value;
	s_input[sm_store_pos + 33] = *B_DFT_value;
	s_input[sm_store_pos + 66] = *C_DFT_value;
	s_input[sm_store_pos + 99] = *D_DFT_value;

	// Read shared memory to get reordered input
	__syncthreads();
	//unsigned int sm_read_pos = (local_id&31)*33 + warp_id*2;
	unsigned int sm_read_pos = local_id * 33 + warp_id * 2;
	*A_DFT_value = s_input[sm_read_pos + 0];
	*B_DFT_value = s_input[sm_read_pos + 1056];
	*C_DFT_value = s_input[sm_read_pos + 1];
	*D_DFT_value = s_input[sm_read_pos + 1056 + 1];

	__syncthreads();
	reorder_64<const_params>(s_input, A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
}



template<class const_params>
__device__ __inline__ void reorder_4096(half2* s_input, half2* A_DFT_value, half2* B_DFT_value, half2* C_DFT_value, half2* D_DFT_value) {
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x / const_params::warp;

	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);

	__syncthreads();
	//unsigned int sm_store_pos = (local_id>>0) + 32*(local_id&0) + warp_id*132;
	unsigned int sm_store_pos = local_id + warp_id * 132;
	s_input[sm_store_pos] = *A_DFT_value;
	s_input[sm_store_pos + 33] = *B_DFT_value;
	s_input[sm_store_pos + 66] = *C_DFT_value;
	s_input[sm_store_pos + 99] = *D_DFT_value;

	// Read shared memory to get reordered input
	__syncthreads();
	//unsigned int sm_read_pos = (local_id&31)*33 + warp_id*2;
	unsigned int sm_read_pos = local_id * 33 + warp_id;
	*A_DFT_value = s_input[sm_read_pos + 0];
	*B_DFT_value = s_input[sm_read_pos + 1056];
	*C_DFT_value = s_input[sm_read_pos + 2112];
	*D_DFT_value = s_input[sm_read_pos + 3168];

	__syncthreads();
	reorder_128<const_params>(s_input, A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
}



template<class const_params>
__device__ void do_SMFFT_CT_DIT(half2* s_input) {
	half2 A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value;
	half2 W;
	half2 Aftemp, Bftemp, Cftemp, Dftemp;

	int j, m_param;
	int parity, itemp;
	int A_read_index, B_read_index, C_read_index, D_read_index;
	int PoT, PoTp1, q;

	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x / const_params::warp;

#ifdef TESTING
	int A_load_id, B_load_id, i, A_n, B_n;
	A_load_id = threadIdx.x;
	B_load_id = threadIdx.x + const_params::fft_length_quarter;
	A_n = threadIdx.x;
	B_n = threadIdx.x + const_params::fft_length_quarter;
	for (i = 1; i < const_params::fft_exp; i++) {
		A_n >>= 1;
		B_n >>= 1;
		A_load_id <<= 1;
		A_load_id |= A_n & 1;
		B_load_id <<= 1;
		B_load_id |= B_n & 1;
	}
	A_load_id &= const_params::fft_length - 1;
	B_load_id &= const_params::fft_length - 1;

	//-----> Scrambling input
	A_DFT_value = s_input[A_load_id];
	B_DFT_value = s_input[A_load_id + 1];
	C_DFT_value = s_input[B_load_id];
	D_DFT_value = s_input[B_load_id + 1];
	__syncthreads();
	s_input[threadIdx.x] = A_DFT_value;
	s_input[threadIdx.x + const_params::fft_length_half] = B_DFT_value;
	s_input[threadIdx.x + const_params::fft_length_quarter] = C_DFT_value;
	s_input[threadIdx.x + const_params::fft_length_three_quarters] = D_DFT_value;
	__syncthreads();
#endif

	//-----> FFT
	//-->
	PoT = 1;
	PoTp1 = 2;

	//--> First iteration
	itemp = local_id & 1;
	parity = (1 - itemp * 2);
	A_DFT_value = s_input[local_id + (warp_id << 2) * const_params::warp];
	B_DFT_value = s_input[local_id + (warp_id << 2) * const_params::warp + const_params::warp];
	C_DFT_value = s_input[local_id + (warp_id << 2) * const_params::warp + 2 * const_params::warp];
	D_DFT_value = s_input[local_id + (warp_id << 2) * const_params::warp + 3 * const_params::warp];

	__syncthreads();

	A_DFT_value.x = __float2half((float)parity) * A_DFT_value.x + shfl_xor(&A_DFT_value.x, 1);
	A_DFT_value.y = __float2half((float)parity) * A_DFT_value.y + shfl_xor(&A_DFT_value.y, 1);
	B_DFT_value.x = __float2half((float)parity) * B_DFT_value.x + shfl_xor(&B_DFT_value.x, 1);
	B_DFT_value.y = __float2half((float)parity) * B_DFT_value.y + shfl_xor(&B_DFT_value.y, 1);
	C_DFT_value.x = __float2half((float)parity) * C_DFT_value.x + shfl_xor(&C_DFT_value.x, 1);
	C_DFT_value.y = __float2half((float)parity) * C_DFT_value.y + shfl_xor(&C_DFT_value.y, 1);
	D_DFT_value.x = __float2half((float)parity) * D_DFT_value.x + shfl_xor(&D_DFT_value.x, 1);
	D_DFT_value.y = __float2half((float)parity) * D_DFT_value.y + shfl_xor(&D_DFT_value.y, 1);

	//--> Second through Fifth iteration (no synchronization)
	PoT = 2;
	PoTp1 = 4;
	for (q = 1; q < 5; q++) {
		m_param = (local_id & (PoTp1 - 1));
		itemp = m_param >> q;
		parity = ((itemp << 1) - 1);

		if (const_params::fft_direction) W = Get_W_value_inverse(PoTp1, itemp * m_param);
		else W = Get_W_value(PoTp1, itemp * m_param);

		Aftemp.x = W.x * A_DFT_value.x - W.y * A_DFT_value.y;
		Aftemp.y = W.x * A_DFT_value.y + W.y * A_DFT_value.x;
		Bftemp.x = W.x * B_DFT_value.x - W.y * B_DFT_value.y;
		Bftemp.y = W.x * B_DFT_value.y + W.y * B_DFT_value.x;
		Cftemp.x = W.x * C_DFT_value.x - W.y * C_DFT_value.y;
		Cftemp.y = W.x * C_DFT_value.y + W.y * C_DFT_value.x;
		Dftemp.x = W.x * D_DFT_value.x - W.y * D_DFT_value.y;
		Dftemp.y = W.x * D_DFT_value.y + W.y * D_DFT_value.x;

		A_DFT_value.x = Aftemp.x + __float2half((float)parity) * shfl_xor(&Aftemp.x, PoT);
		A_DFT_value.y = Aftemp.y + __float2half((float)parity) * shfl_xor(&Aftemp.y, PoT);
		B_DFT_value.x = Bftemp.x + __float2half((float)parity) * shfl_xor(&Bftemp.x, PoT);
		B_DFT_value.y = Bftemp.y + __float2half((float)parity) * shfl_xor(&Bftemp.y, PoT);
		C_DFT_value.x = Cftemp.x + __float2half((float)parity) * shfl_xor(&Cftemp.x, PoT);
		C_DFT_value.y = Cftemp.y + __float2half((float)parity) * shfl_xor(&Cftemp.y, PoT);
		D_DFT_value.x = Dftemp.x + __float2half((float)parity) * shfl_xor(&Dftemp.x, PoT);
		D_DFT_value.y = Dftemp.y + __float2half((float)parity) * shfl_xor(&Dftemp.y, PoT);

		PoT = PoT << 1;
		PoTp1 = PoTp1 << 1;
	}

	itemp = local_id + (warp_id << 2) * const_params::warp;
	s_input[itemp] = A_DFT_value;
	s_input[itemp + const_params::warp] = B_DFT_value;
	s_input[itemp + 2 * const_params::warp] = C_DFT_value;
	s_input[itemp + 3 * const_params::warp] = D_DFT_value;

	if (const_params::fft_exp == 6) {
		__syncthreads();
		q = 5;
		m_param = threadIdx.x & (PoT - 1);
		j = threadIdx.x >> q;

		if (const_params::fft_direction) W = Get_W_value_inverse(PoTp1, m_param);
		else W = Get_W_value(PoTp1, m_param);

		A_read_index = j * (PoTp1 << 1) + m_param;
		B_read_index = j * (PoTp1 << 1) + m_param + PoT;
		C_read_index = j * (PoTp1 << 1) + m_param + PoTp1;
		D_read_index = j * (PoTp1 << 1) + m_param + 3 * PoT;

		Aftemp = s_input[A_read_index];
		Bftemp = s_input[B_read_index];
		A_DFT_value.x = Aftemp.x + W.x * Bftemp.x - W.y * Bftemp.y;
		A_DFT_value.y = Aftemp.y + W.x * Bftemp.y + W.y * Bftemp.x;
		B_DFT_value.x = Aftemp.x - W.x * Bftemp.x + W.y * Bftemp.y;
		B_DFT_value.y = Aftemp.y - W.x * Bftemp.y - W.y * Bftemp.x;

		Cftemp = s_input[C_read_index];
		Dftemp = s_input[D_read_index];
		C_DFT_value.x = Cftemp.x + W.x * Dftemp.x - W.y * Dftemp.y;
		C_DFT_value.y = Cftemp.y + W.x * Dftemp.y + W.y * Dftemp.x;
		D_DFT_value.x = Cftemp.x - W.x * Dftemp.x + W.y * Dftemp.y;
		D_DFT_value.y = Cftemp.y - W.x * Dftemp.y - W.y * Dftemp.x;

		s_input[A_read_index] = A_DFT_value;
		s_input[B_read_index] = B_DFT_value;
		s_input[C_read_index] = C_DFT_value;
		s_input[D_read_index] = D_DFT_value;

		PoT = PoT << 1;
		PoTp1 = PoTp1 << 1;
	}

	for (q = 5; q < (const_params::fft_exp - 1); q++) {
		__syncthreads();
		m_param = threadIdx.x & (PoT - 1);
		j = threadIdx.x >> q;

		if (const_params::fft_direction) W = Get_W_value_inverse(PoTp1, m_param);
		else W = Get_W_value(PoTp1, m_param);

		A_read_index = j * (PoTp1 << 1) + m_param;
		B_read_index = j * (PoTp1 << 1) + m_param + PoT;
		C_read_index = j * (PoTp1 << 1) + m_param + PoTp1;
		D_read_index = j * (PoTp1 << 1) + m_param + 3 * PoT;

		Aftemp = s_input[A_read_index];
		Bftemp = s_input[B_read_index];
		A_DFT_value.x = Aftemp.x + W.x * Bftemp.x - W.y * Bftemp.y;
		A_DFT_value.y = Aftemp.y + W.x * Bftemp.y + W.y * Bftemp.x;
		B_DFT_value.x = Aftemp.x - W.x * Bftemp.x + W.y * Bftemp.y;
		B_DFT_value.y = Aftemp.y - W.x * Bftemp.y - W.y * Bftemp.x;

		Cftemp = s_input[C_read_index];
		Dftemp = s_input[D_read_index];
		C_DFT_value.x = Cftemp.x + W.x * Dftemp.x - W.y * Dftemp.y;
		C_DFT_value.y = Cftemp.y + W.x * Dftemp.y + W.y * Dftemp.x;
		D_DFT_value.x = Cftemp.x - W.x * Dftemp.x + W.y * Dftemp.y;
		D_DFT_value.y = Cftemp.y - W.x * Dftemp.y - W.y * Dftemp.x;

		s_input[A_read_index] = A_DFT_value;
		s_input[B_read_index] = B_DFT_value;
		s_input[C_read_index] = C_DFT_value;
		s_input[D_read_index] = D_DFT_value;

		PoT = PoT << 1;
		PoTp1 = PoTp1 << 1;
	}

	//last iteration
	if (const_params::fft_exp > 6) {
		__syncthreads();
		m_param = threadIdx.x;

		if (const_params::fft_direction) W = Get_W_value_inverse(PoTp1, m_param);
		else W = Get_W_value(PoTp1, m_param);

		A_read_index = m_param;
		B_read_index = m_param + PoT;
		C_read_index = m_param + (PoT >> 1);
		D_read_index = m_param + 3 * (PoT >> 1);

		Aftemp = s_input[A_read_index];
		Bftemp = s_input[B_read_index];
		A_DFT_value.x = Aftemp.x + W.x * Bftemp.x - W.y * Bftemp.y;
		A_DFT_value.y = Aftemp.y + W.x * Bftemp.y + W.y * Bftemp.x;
		B_DFT_value.x = Aftemp.x - W.x * Bftemp.x + W.y * Bftemp.y;
		B_DFT_value.y = Aftemp.y - W.x * Bftemp.y - W.y * Bftemp.x;

		Cftemp = s_input[C_read_index];
		Dftemp = s_input[D_read_index];
		C_DFT_value.x = Cftemp.x + W.y * Dftemp.x + W.x * Dftemp.y;
		C_DFT_value.y = Cftemp.y + W.y * Dftemp.y - W.x * Dftemp.x;
		D_DFT_value.x = Cftemp.x - W.y * Dftemp.x - W.x * Dftemp.y;
		D_DFT_value.y = Cftemp.y - W.y * Dftemp.y + W.x * Dftemp.x;

		s_input[A_read_index] = A_DFT_value;
		s_input[B_read_index] = B_DFT_value;
		s_input[C_read_index] = C_DFT_value;
		s_input[D_read_index] = D_DFT_value;
	}
}

/*
template<class const_params>
__device__ void do_SMFFT_CT_DIT(half2 *s_input){
	half2 A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value;
	half2 W;
	half2 Aftemp, Bftemp, Cftemp, Dftemp;

	int j, m_param;
	int parity, itemp;
	int A_read_index, B_read_index, C_read_index, D_read_index;
	int PoT, PoTp1, q;

	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x/const_params::warp;
	A_DFT_value = s_input[local_id + (warp_id<<2)*const_params::warp];
	B_DFT_value = s_input[local_id + (warp_id<<2)*const_params::warp + const_params::warp];
	C_DFT_value = s_input[local_id + (warp_id<<2)*const_params::warp + 2*const_params::warp];
	D_DFT_value = s_input[local_id + (warp_id<<2)*const_params::warp + 3*const_params::warp];

	if(const_params::fft_reorder){
		if(const_params::fft_exp==5)       reorder_32<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
		else if(const_params::fft_exp==6)  reorder_64<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
		else if(const_params::fft_exp==7)  reorder_128<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
		else if(const_params::fft_exp==8)  reorder_256<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
		else if(const_params::fft_exp==9)  reorder_512<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
		else if(const_params::fft_exp==10) reorder_1024<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
		else if(const_params::fft_exp==11) reorder_2048<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
		else if(const_params::fft_exp==12) reorder_4096<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
	}

	//----> FFT
	PoT=1;
	PoTp1=2;

	//--> First iteration
	itemp=local_id&1;
	parity=(1-itemp*2);

	A_DFT_value.x = parity*A_DFT_value.x + shfl_xor(&A_DFT_value.x, 1);
	A_DFT_value.y = parity*A_DFT_value.y + shfl_xor(&A_DFT_value.y, 1);
	B_DFT_value.x = parity*B_DFT_value.x + shfl_xor(&B_DFT_value.x, 1);
	B_DFT_value.y = parity*B_DFT_value.y + shfl_xor(&B_DFT_value.y, 1);
	C_DFT_value.x = parity*C_DFT_value.x + shfl_xor(&C_DFT_value.x, 1);
	C_DFT_value.y = parity*C_DFT_value.y + shfl_xor(&C_DFT_value.y, 1);
	D_DFT_value.x = parity*D_DFT_value.x + shfl_xor(&D_DFT_value.x, 1);
	D_DFT_value.y = parity*D_DFT_value.y + shfl_xor(&D_DFT_value.y, 1);

	//--> Second through Fifth iteration (no synchronization)
	PoT=2;
	PoTp1=4;
	for(q=1;q<5;q++){
		m_param = (local_id & (PoTp1 - 1));
		itemp   = m_param>>q;
		parity  = ((itemp<<1)-1);

		if(const_params::fft_direction) W = Get_W_value_inverse(PoTp1, itemp*m_param);
		else W = Get_W_value(PoTp1, itemp*m_param);

		Aftemp.x = W.x*A_DFT_value.x - W.y*A_DFT_value.y;
		Aftemp.y = W.x*A_DFT_value.y + W.y*A_DFT_value.x;
		Bftemp.x = W.x*B_DFT_value.x - W.y*B_DFT_value.y;
		Bftemp.y = W.x*B_DFT_value.y + W.y*B_DFT_value.x;
		Cftemp.x = W.x*C_DFT_value.x - W.y*C_DFT_value.y;
		Cftemp.y = W.x*C_DFT_value.y + W.y*C_DFT_value.x;
		Dftemp.x = W.x*D_DFT_value.x - W.y*D_DFT_value.y;
		Dftemp.y = W.x*D_DFT_value.y + W.y*D_DFT_value.x;

		A_DFT_value.x = Aftemp.x + parity*shfl_xor(&Aftemp.x,PoT);
		A_DFT_value.y = Aftemp.y + parity*shfl_xor(&Aftemp.y,PoT);
		B_DFT_value.x = Bftemp.x + parity*shfl_xor(&Bftemp.x,PoT);
		B_DFT_value.y = Bftemp.y + parity*shfl_xor(&Bftemp.y,PoT);
		C_DFT_value.x = Cftemp.x + parity*shfl_xor(&Cftemp.x,PoT);
		C_DFT_value.y = Cftemp.y + parity*shfl_xor(&Cftemp.y,PoT);
		D_DFT_value.x = Dftemp.x + parity*shfl_xor(&Dftemp.x,PoT);
		D_DFT_value.y = Dftemp.y + parity*shfl_xor(&Dftemp.y,PoT);

		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}

	itemp = local_id + (warp_id<<2)*const_params::warp;
	s_input[itemp]                        = A_DFT_value;
	s_input[itemp + const_params::warp]   = B_DFT_value;
	s_input[itemp + 2*const_params::warp] = C_DFT_value;
	s_input[itemp + 3*const_params::warp] = D_DFT_value;

	if(const_params::fft_exp==6){
		__syncthreads();
		q = 5;
		m_param = threadIdx.x & (PoT - 1);
		j = threadIdx.x>>q;

		if(const_params::fft_direction) W = Get_W_value_inverse(PoTp1,m_param);
		else W = Get_W_value(PoTp1,m_param);

		A_read_index=j*(PoTp1<<1) + m_param;
		B_read_index=j*(PoTp1<<1) + m_param + PoT;
		C_read_index=j*(PoTp1<<1) + m_param + PoTp1;
		D_read_index=j*(PoTp1<<1) + m_param + 3*PoT;

		Aftemp = s_input[A_read_index];
		Bftemp = s_input[B_read_index];
		A_DFT_value.x=Aftemp.x + W.x*Bftemp.x - W.y*Bftemp.y;
		A_DFT_value.y=Aftemp.y + W.x*Bftemp.y + W.y*Bftemp.x;
		B_DFT_value.x=Aftemp.x - W.x*Bftemp.x + W.y*Bftemp.y;
		B_DFT_value.y=Aftemp.y - W.x*Bftemp.y - W.y*Bftemp.x;

		Cftemp = s_input[C_read_index];
		Dftemp = s_input[D_read_index];
		C_DFT_value.x=Cftemp.x + W.x*Dftemp.x - W.y*Dftemp.y;
		C_DFT_value.y=Cftemp.y + W.x*Dftemp.y + W.y*Dftemp.x;
		D_DFT_value.x=Cftemp.x - W.x*Dftemp.x + W.y*Dftemp.y;
		D_DFT_value.y=Cftemp.y - W.x*Dftemp.y - W.y*Dftemp.x;

		s_input[A_read_index]=A_DFT_value;
		s_input[B_read_index]=B_DFT_value;
		s_input[C_read_index]=C_DFT_value;
		s_input[D_read_index]=D_DFT_value;

		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}

	for(q=5;q<(const_params::fft_exp-1);q++){
		__syncthreads();
		m_param = threadIdx.x & (PoT - 1);
		j=threadIdx.x>>q;

		if(const_params::fft_direction) W = Get_W_value_inverse(PoTp1,m_param);
		else W = Get_W_value(PoTp1,m_param);

		A_read_index=j*(PoTp1<<1) + m_param;
		B_read_index=j*(PoTp1<<1) + m_param + PoT;
		C_read_index=j*(PoTp1<<1) + m_param + PoTp1;
		D_read_index=j*(PoTp1<<1) + m_param + 3*PoT;

		Aftemp = s_input[A_read_index];
		Bftemp = s_input[B_read_index];
		A_DFT_value.x=Aftemp.x + W.x*Bftemp.x - W.y*Bftemp.y;
		A_DFT_value.y=Aftemp.y + W.x*Bftemp.y + W.y*Bftemp.x;
		B_DFT_value.x=Aftemp.x - W.x*Bftemp.x + W.y*Bftemp.y;
		B_DFT_value.y=Aftemp.y - W.x*Bftemp.y - W.y*Bftemp.x;

		Cftemp = s_input[C_read_index];
		Dftemp = s_input[D_read_index];
		C_DFT_value.x=Cftemp.x + W.x*Dftemp.x - W.y*Dftemp.y;
		C_DFT_value.y=Cftemp.y + W.x*Dftemp.y + W.y*Dftemp.x;
		D_DFT_value.x=Cftemp.x - W.x*Dftemp.x + W.y*Dftemp.y;
		D_DFT_value.y=Cftemp.y - W.x*Dftemp.y - W.y*Dftemp.x;

		s_input[A_read_index]=A_DFT_value;
		s_input[B_read_index]=B_DFT_value;
		s_input[C_read_index]=C_DFT_value;
		s_input[D_read_index]=D_DFT_value;

		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}

	//last iteration
	if(const_params::fft_exp>6) {
		__syncthreads();
		m_param = threadIdx.x;

		if(const_params::fft_direction) W = Get_W_value_inverse(PoTp1,m_param);
		else W = Get_W_value(PoTp1,m_param);

		A_read_index = m_param;
		B_read_index = m_param + PoT;
		C_read_index = m_param + (PoT>>1);
		D_read_index = m_param + 3*(PoT>>1);

		Aftemp = s_input[A_read_index];
		Bftemp = s_input[B_read_index];
		A_DFT_value.x = Aftemp.x + W.x*Bftemp.x - W.y*Bftemp.y;
		A_DFT_value.y = Aftemp.y + W.x*Bftemp.y + W.y*Bftemp.x;
		B_DFT_value.x = Aftemp.x - W.x*Bftemp.x + W.y*Bftemp.y;
		B_DFT_value.y = Aftemp.y - W.x*Bftemp.y - W.y*Bftemp.x;

		Cftemp = s_input[C_read_index];
		Dftemp = s_input[D_read_index];
		C_DFT_value.x = Cftemp.x + W.y*Dftemp.x + W.x*Dftemp.y;
		C_DFT_value.y = Cftemp.y + W.y*Dftemp.y - W.x*Dftemp.x;
		D_DFT_value.x = Cftemp.x - W.y*Dftemp.x - W.x*Dftemp.y;
		D_DFT_value.y = Cftemp.y - W.y*Dftemp.y + W.x*Dftemp.x;

		s_input[A_read_index]=A_DFT_value;
		s_input[B_read_index]=B_DFT_value;
		s_input[C_read_index]=C_DFT_value;
		s_input[D_read_index]=D_DFT_value;
	}
}*/

template<class const_params>
__global__ void SMFFT_DIT_external(half2* d_input, half2* d_output) {
	__shared__ half2 s_input[const_params::fft_sm_required];

	s_input[threadIdx.x] = d_input[threadIdx.x + blockIdx.x * const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_length_quarter] = d_input[threadIdx.x + blockIdx.x * const_params::fft_length + const_params::fft_length_quarter];
	s_input[threadIdx.x + const_params::fft_length_half] = d_input[threadIdx.x + blockIdx.x * const_params::fft_length + const_params::fft_length_half];
	s_input[threadIdx.x + const_params::fft_length_three_quarters] = d_input[threadIdx.x + blockIdx.x * const_params::fft_length + const_params::fft_length_three_quarters];

	__syncthreads();
	do_SMFFT_CT_DIT<const_params>(s_input);

	__syncthreads();
	d_output[threadIdx.x + blockIdx.x * const_params::fft_length] = s_input[threadIdx.x];
	d_output[threadIdx.x + blockIdx.x * const_params::fft_length + const_params::fft_length_quarter] = s_input[threadIdx.x + const_params::fft_length_quarter];
	d_output[threadIdx.x + blockIdx.x * const_params::fft_length + const_params::fft_length_half] = s_input[threadIdx.x + const_params::fft_length_half];
	d_output[threadIdx.x + blockIdx.x * const_params::fft_length + const_params::fft_length_three_quarters] = s_input[threadIdx.x + const_params::fft_length_three_quarters];
}

template<class const_params>
__global__ void SMFFT_DIT_multiple(half2* d_input, half2* d_output) {
	__shared__ half2 s_input[const_params::fft_sm_required];

	s_input[threadIdx.x] = d_input[threadIdx.x + blockIdx.x * const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_length_quarter] = d_input[threadIdx.x + blockIdx.x * const_params::fft_length + const_params::fft_length_quarter];
	s_input[threadIdx.x + const_params::fft_length_half] = d_input[threadIdx.x + blockIdx.x * const_params::fft_length + const_params::fft_length_half];
	s_input[threadIdx.x + const_params::fft_length_three_quarters] = d_input[threadIdx.x + blockIdx.x * const_params::fft_length + const_params::fft_length_three_quarters];

	__syncthreads();
	for (int f = 0; f < NREUSES; f++) {
		do_SMFFT_CT_DIT<const_params>(s_input);
	}
	__syncthreads();

	d_output[threadIdx.x + blockIdx.x * const_params::fft_length] = s_input[threadIdx.x];
	d_output[threadIdx.x + blockIdx.x * const_params::fft_length + const_params::fft_length_quarter] = s_input[threadIdx.x + const_params::fft_length_quarter];
	d_output[threadIdx.x + blockIdx.x * const_params::fft_length + const_params::fft_length_half] = s_input[threadIdx.x + const_params::fft_length_half];
	d_output[threadIdx.x + blockIdx.x * const_params::fft_length + const_params::fft_length_three_quarters] = s_input[threadIdx.x + const_params::fft_length_three_quarters];
}

//---------------------------------- Device End -------------------<

void FFT_init() {
	//---------> Specific nVidia stuff
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
}

int FFT_external_benchmark(half2* d_input, half2* d_output, int FFT_size, int nFFTs, bool inverse, bool reorder, double* FFT_time) {
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize(nFFTs, 1, 1);
	dim3 blockSize(FFT_size / 4, 1, 1);
	if (FFT_size == 32) {
		gridSize.x = nFFTs / 4;
		blockSize.x = 32;
	}
	if (FFT_size == 64) {
		gridSize.x = nFFTs / 2;
		blockSize.x = 32;
	}

	//---------> FFT part
	timer.Start();
	switch (FFT_size) {
	case 32:
		if (inverse == false && reorder == true)  SMFFT_DIT_external<FFT_32_forward> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == false && reorder == false) SMFFT_DIT_external<FFT_32_forward_noreorder> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == true)   SMFFT_DIT_external<FFT_32_inverse> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == false)  SMFFT_DIT_external<FFT_32_inverse_noreorder> << <gridSize, blockSize >> > (d_input, d_output);
		break;

	case 64:
		if (inverse == false && reorder == true)  SMFFT_DIT_external<FFT_64_forward> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == false && reorder == false) SMFFT_DIT_external<FFT_64_forward_noreorder> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == true)   SMFFT_DIT_external<FFT_64_inverse> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == false)  SMFFT_DIT_external<FFT_64_inverse_noreorder> << <gridSize, blockSize >> > (d_input, d_output);
		break;

	case 128:
		if (inverse == false && reorder == true)  SMFFT_DIT_external<FFT_128_forward> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == false && reorder == false) SMFFT_DIT_external<FFT_128_forward_noreorder> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == true)   SMFFT_DIT_external<FFT_128_inverse> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == false)  SMFFT_DIT_external<FFT_128_inverse_noreorder> << <gridSize, blockSize >> > (d_input, d_output);
		break;

	case 256:
		if (inverse == false && reorder == true)  SMFFT_DIT_external<FFT_256_forward> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == false && reorder == false) SMFFT_DIT_external<FFT_256_forward_noreorder> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == true)   SMFFT_DIT_external<FFT_256_inverse> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == false)  SMFFT_DIT_external<FFT_256_inverse_noreorder> << <gridSize, blockSize >> > (d_input, d_output);
		break;

	case 512:
		if (inverse == false && reorder == true)  SMFFT_DIT_external<FFT_512_forward> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == false && reorder == false) SMFFT_DIT_external<FFT_512_forward_noreorder> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == true)   SMFFT_DIT_external<FFT_512_inverse> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == false)  SMFFT_DIT_external<FFT_512_inverse_noreorder> << <gridSize, blockSize >> > (d_input, d_output);
		break;

	case 1024:
		if (inverse == false && reorder == true)  SMFFT_DIT_external<FFT_1024_forward> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == false && reorder == false) SMFFT_DIT_external<FFT_1024_forward_noreorder> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == true)   SMFFT_DIT_external<FFT_1024_inverse> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == false)  SMFFT_DIT_external<FFT_1024_inverse_noreorder> << <gridSize, blockSize >> > (d_input, d_output);
		break;

	case 2048:
		if (inverse == false && reorder == true)  SMFFT_DIT_external<FFT_2048_forward> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == false && reorder == false) SMFFT_DIT_external<FFT_2048_forward_noreorder> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == true)   SMFFT_DIT_external<FFT_2048_inverse> << <gridSize, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == false)  SMFFT_DIT_external<FFT_2048_inverse_noreorder> << <gridSize, blockSize >> > (d_input, d_output);
		break;

		/*case 4096:
			if(inverse==false && reorder==true)  SMFFT_DIT_external<FFT_4096_forward><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==false && reorder==false) SMFFT_DIT_external<FFT_4096_forward_noreorder><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==true)   SMFFT_DIT_external<FFT_4096_inverse><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==false)  SMFFT_DIT_external<FFT_4096_inverse_noreorder><<<gridSize, blockSize>>>(d_input, d_output);
			break;
		*/
	default:
		printf("Error wrong FFT length!\n");
		break;
	}
	timer.Stop();

	*FFT_time += timer.Elapsed();
	return(0);
}

int FFT_multiple_benchmark(half2* d_input, half2* d_output, int FFT_size, int nFFTs, bool inverse, bool reorder, double* FFT_time) {
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	int nBlocks = (int)(nFFTs / NREUSES);
	if (nBlocks == 0) {
		*FFT_time = -1;
		return(1);
	}
	dim3 gridSize_multiple(nBlocks, 1, 1);
	dim3 blockSize(FFT_size / 4, 1, 1);
	if (FFT_size == 32) {
		gridSize_multiple.x = nFFTs / (4 * NREUSES);
		blockSize.x = 32;
	}
	if (FFT_size == 64) {
		gridSize_multiple.x = nFFTs / (2 * NREUSES);
		blockSize.x = 32;
	}

	//---------> FFT part
	timer.Start();
	switch (FFT_size) {
	case 32:
		if (inverse == false && reorder == true)  SMFFT_DIT_multiple<FFT_32_forward> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == false && reorder == false) SMFFT_DIT_multiple<FFT_32_forward_noreorder> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == true)   SMFFT_DIT_multiple<FFT_32_inverse> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == false)  SMFFT_DIT_multiple<FFT_32_inverse_noreorder> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		break;

	case 64:
		if (inverse == false && reorder == true)  SMFFT_DIT_multiple<FFT_64_forward> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == false && reorder == false) SMFFT_DIT_multiple<FFT_64_forward_noreorder> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == true)   SMFFT_DIT_multiple<FFT_64_inverse> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == false)  SMFFT_DIT_multiple<FFT_64_inverse_noreorder> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		break;

	case 128:
		if (inverse == false && reorder == true)  SMFFT_DIT_multiple<FFT_128_forward> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == false && reorder == false) SMFFT_DIT_multiple<FFT_128_forward_noreorder> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == true)   SMFFT_DIT_multiple<FFT_128_inverse> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == false)  SMFFT_DIT_multiple<FFT_128_inverse_noreorder> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		break;

	case 256:
		if (inverse == false && reorder == true)  SMFFT_DIT_multiple<FFT_256_forward> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == false && reorder == false) SMFFT_DIT_multiple<FFT_256_forward_noreorder> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == true)   SMFFT_DIT_multiple<FFT_256_inverse> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == false)  SMFFT_DIT_multiple<FFT_256_inverse_noreorder> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		break;

	case 512:
		if (inverse == false && reorder == true)  SMFFT_DIT_multiple<FFT_512_forward> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == false && reorder == false) SMFFT_DIT_multiple<FFT_512_forward_noreorder> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == true)   SMFFT_DIT_multiple<FFT_512_inverse> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == false)  SMFFT_DIT_multiple<FFT_512_inverse_noreorder> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		break;

	case 1024:
		if (inverse == false && reorder == true)  SMFFT_DIT_multiple<FFT_1024_forward> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == false && reorder == false) SMFFT_DIT_multiple<FFT_1024_forward_noreorder> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == true)   SMFFT_DIT_multiple<FFT_1024_inverse> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == false)  SMFFT_DIT_multiple<FFT_1024_inverse_noreorder> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		break;

	case 2048:
		if (inverse == false && reorder == true)  SMFFT_DIT_multiple<FFT_2048_forward> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == false && reorder == false) SMFFT_DIT_multiple<FFT_2048_forward_noreorder> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == true)   SMFFT_DIT_multiple<FFT_2048_inverse> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		if (inverse == true && reorder == false)  SMFFT_DIT_multiple<FFT_2048_inverse_noreorder> << <gridSize_multiple, blockSize >> > (d_input, d_output);
		break;
		/*
	case 4096:
		if(inverse==false && reorder==true)  SMFFT_DIT_multiple<FFT_4096_forward><<<gridSize_multiple, blockSize>>>(d_input, d_output);
		if(inverse==false && reorder==false) SMFFT_DIT_multiple<FFT_4096_forward_noreorder><<<gridSize_multiple, blockSize>>>(d_input, d_output);
		if(inverse==true && reorder==true)   SMFFT_DIT_multiple<FFT_4096_inverse><<<gridSize_multiple, blockSize>>>(d_input, d_output);
		if(inverse==true && reorder==false)  SMFFT_DIT_multiple<FFT_4096_inverse_noreorder><<<gridSize_multiple, blockSize>>>(d_input, d_output);
		break;
	*/
	default:
		printf("Error wrong FFT length!\n");
		break;
	}
	timer.Stop();

	*FFT_time += timer.Elapsed();
	return(0);
}


// ***********************************************************************************
// ***********************************************************************************
// ***********************************************************************************
int GPU_cuFFT(half2* h_input, half2* h_output, int FFT_size, int nFFTs, bool inverse, int nRuns, double* single_ex_time) {
	//---------> Initial nVidia stuff
	int devCount;
	size_t free_mem, total_mem;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if (devCount > device) checkCudaErrors(cudaSetDevice(device));

	//---------> Checking memory
	cudaMemGetInfo(&free_mem, &total_mem);
	if (DEBUG) printf("\n  Device has %0.3f MB of total memory, which %0.3f MB is available.\n", ((float)total_mem) / (1024.0 * 1024.0), (float)free_mem / (1024.0 * 1024.0));
	long long input_size = FFT_size * nFFTs;
	size_t output_size = FFT_size * nFFTs;
	size_t total_memory_required_bytes = input_size * sizeof(half2) + output_size * sizeof(half2);
	if (total_memory_required_bytes > free_mem) {
		printf("Error: Not enough memory! Input data are too big for the device.\n");
		return(1);
	}

	//----------> Memory allocation
	half2* d_input;
	half2* d_output;
	checkCudaErrors(cudaMalloc((void**)&d_input, sizeof(half2) * input_size));
	checkCudaErrors(cudaMalloc((void**)&d_output, sizeof(half2) * output_size));

	checkCudaErrors(cudaMemcpy(d_input, h_input, input_size * sizeof(half2), cudaMemcpyHostToDevice));

	//---------> Measurements
	double time_cuFFT = 0;
	GpuTimer timer;

	//--------------------------------------------------
	//-------------------------> cuFFT
	if (DEBUG) printf("  Running cuFFT...: \t\t");
	cufftHandle plan;
	cufftResult error;
	//error = cufftPlan1d(&plan, FFT_size, CUFFT_Z2Z, nFFTs);
	error = cufftCreate(&plan);
	assert(error == CUFFT_SUCCESS);
	size_t ws = 0;
	long long int rank = 1;
	long long int n[1]; n[0]=input_size;
	long long int nembed[1]; nembed[0]=input_size;
	long long int stride = 1;
	long long int dist = input_size;

	error = cufftXtMakePlanMany(plan,rank,n,nembed,stride,dist,CUDA_C_16F,nembed,stride,dist,CUDA_C_16F,1,&ws,CUDA_C_16F);



	//error = cufftXtMakePlanMany(plan,(long long) 1,	&input_size,(long long) &FFT_size,(long long) 1,(long long) FFT_size,CUDA_C_16F,(long long) &FFT_size, (long long) 1, (long long) 1,CUDA_C_16F,(long long) 1,&ws,CUDA_C_16F);

	printf("Error %d in cufftXtMakePlanMany()\n", error);
	switch (error)
    {
        case CUFFT_SUCCESS:
            printf("CUFFT_SUCCESS\n");
            break;
        case CUFFT_INVALID_PLAN:
            printf("CUFFT_INVALID_PLAN\n");
            break;
        case CUFFT_ALLOC_FAILED:
            printf("CUFFT_ALLOC_FAILED\n");
            break;
        case CUFFT_INVALID_TYPE:
            printf("CUFFT_INVALID_TYPE\n");
            break;
        case CUFFT_INVALID_VALUE:
            printf("CUFFT_INVALID_VALUE\n");
            break;
        case CUFFT_INTERNAL_ERROR:
            printf("CUFFT_INTERNAL_ERROR\n");
            break;
        case CUFFT_EXEC_FAILED:
            printf("CUFFT_EXEC_FAILED\n");
            break;
        case CUFFT_SETUP_FAILED:
            printf("CUFFT_SETUP_FAILED\n");
            break;
        case CUFFT_INVALID_SIZE:
            printf("CUFFT_INVALID_SIZE\n");
            break;
        case CUFFT_UNALIGNED_DATA:
            printf("CUFFT_UNALIGNED_DATA\n");
            break;
        case CUFFT_NOT_SUPPORTED:
        	printf("CUFFT_NOT_SUPPORTED\n");
        	break;
        default:
        	printf("NO CASE MATCH\n");
        	break;
    }



	//assert(error == CUFFT_SUCCESS);

	timer.Start();
	if (inverse) error = cufftXtExec(plan, d_input, d_output, CUFFT_INVERSE);
	else error = cufftXtExec(plan, d_input, d_output, CUFFT_FORWARD);
	timer.Stop();

	time_cuFFT += timer.Elapsed();

	cufftDestroy(plan);
	if (DEBUG) printf("done in %g ms.\n", time_cuFFT);
	*single_ex_time = time_cuFFT;
	//-----------------------------------<
	//--------------------------------------------------

	printf("  FFT size: %d; cuFFT time = %0.3f ms;\n", FFT_size, time_cuFFT);

	cudaDeviceSynchronize();

	//---------> Copy Device -> Host
	checkCudaErrors(cudaMemcpy(h_output, d_output, output_size * sizeof(half2), cudaMemcpyDeviceToHost));

	//---------> error check -----
	checkCudaErrors(cudaGetLastError());

	//---------> Feeing allocated resources
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_output));

	return(0);
}

int GPU_smFFT_4elements(half2* h_input, half2* h_output, int FFT_size, int nFFTs, bool inverse, bool reorder, int nRuns, double* single_ex_time, double* multi_ex_time) {
	//---------> Initial nVidia stuff
	int devCount;
	size_t free_mem, total_mem;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if (devCount > device) checkCudaErrors(cudaSetDevice(device));

	//---------> Checking edge cases
	if (FFT_size == 32 && (nFFTs % 4) != 0) return(1);
	if (FFT_size == 64 && (nFFTs % 2) != 0) return(1);

	//---------> Checking memory
	cudaMemGetInfo(&free_mem, &total_mem);
	if (DEBUG) printf("\n  Device has %0.3f MB of total memory, which %0.3f MB is available.\n", ((float)total_mem) / (1024.0 * 1024.0), (float)free_mem / (1024.0 * 1024.0));
	size_t input_size = FFT_size * nFFTs;
	size_t output_size = FFT_size * nFFTs;
	size_t total_memory_required_bytes = input_size * sizeof(half2) + output_size * sizeof(half2);
	if (total_memory_required_bytes > free_mem) {
		printf("Error: Not enough memory! Input data is too big for the device.\n");
		return(1);
	}

	//----------> Memory allocation
	half2* d_input;
	half2* d_output;
	checkCudaErrors(cudaMalloc((void**)&d_input, sizeof(half2) * input_size));
	checkCudaErrors(cudaMalloc((void**)&d_output, sizeof(half2) * output_size));

	//---------> Measurements
	double time_FFT_external = 0, time_FFT_multiple = 0;

	checkCudaErrors(cudaGetLastError());

	//--------------------------------------------------
	//-------------------------> 4way
	if (MULTIPLE) {
		if (DEBUG) printf("  Running shared memory FFT (Cooley-Tukey) 100 times per GPU kernel (eliminates device memory)... ");
		FFT_init();
		double total_time_FFT_multiple = 0;
		for (int f = 0; f < nRuns; f++) {
			//---> Copy Host -> Device
			checkCudaErrors(cudaMemcpy(d_input, h_input, input_size * sizeof(half2), cudaMemcpyHostToDevice));
			FFT_multiple_benchmark(d_input, d_output, FFT_size, nFFTs, inverse, reorder, &total_time_FFT_multiple);
		}
		time_FFT_multiple = total_time_FFT_multiple / nRuns;
		if (DEBUG) printf("done in %g ms.\n", time_FFT_multiple);
		*multi_ex_time = time_FFT_multiple;
	}

	checkCudaErrors(cudaGetLastError());

	if (EXTERNAL) {
		if (DEBUG) printf("  Running shared memory FFT (Cooley-Tukey)... ");
		FFT_init();
		double total_time_FFT_external = 0;
		for (int f = 0; f < nRuns; f++) {
			//---> Copy Host -> Device
			checkCudaErrors(cudaMemcpy(d_input, h_input, input_size * sizeof(half2), cudaMemcpyHostToDevice));
			FFT_external_benchmark(d_input, d_output, FFT_size, nFFTs, inverse, reorder, &total_time_FFT_external);
		}
		time_FFT_external = total_time_FFT_external / nRuns;
		if (DEBUG) printf("done in %g ms.\n", time_FFT_external);
		*single_ex_time = time_FFT_external;
	}

	checkCudaErrors(cudaGetLastError());
	//-----------------------------------<
	//--------------------------------------------------
	printf("  SH FFT normal = %0.3f ms; SM FFT multiple times = %0.3f ms\n", time_FFT_external, time_FFT_multiple);

	//---------> Copy Device -> Host
	checkCudaErrors(cudaMemcpy(h_output, d_output, output_size * sizeof(half2), cudaMemcpyDeviceToHost));

	//---------> error check -----
	checkCudaErrors(cudaGetLastError());

	//---------> Feeing allocated resources
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_output));

	return(0);
}





//float max_error = 1.0e-4;

void Generate_signal(half* signal, int samples) {
	float f1, f2, a1, a2;
	f1 = 1.0 / 8.0; f2 = 2.0 / 8.0; a1 = 1.0; a2 = 0.5;

	for (int f = 0; f < samples; f++) {
		signal[f] = __float2half(a1 * sin(2.0 * 3.141592654 * f1 * f) + a2 * sin(2.0 * 3.141592654 * f2 * f + (3.0 * 3.141592654) / 4.0));
	}
}
/*
half get_error(float A, float B) {
	float error, div_error = 10000, per_error = 10000, order = 0;
	int power;
	if (A < 0) A = -A;
	if (B < 0) B = -B;

	if (A > B) {
		div_error = A - B;
		if (B > 10) {
			power = (int)log10(B);
			order = pow(10, power);
			div_error = div_error / order;
		}
	}
	else {
		div_error = B - A;
		if (A > 10) {
			power = (int)log10(A);
			order = pow(10, power);
			div_error = div_error / order;
		}
	}

	if (div_error < per_error) error = div_error;
	else error = per_error;
	return(__float2half(error));
}*/

/*
int Compare_data(half2* cuFFT_result, half2* smFFT_result, int FFT_size, int nFFTs, half* cumulative_error, half* mean_error) {
	float error;
	int nErrors = 0;
	float dtemp;
	int cislo = 0;

	dtemp = (float)0;
	for (int i = 0; i < nFFTs; i++) {
		for (int f = 0; f < FFT_size; f++) {
			int pos = i * FFT_size + f;
			float error_real, error_img;
			error_real = __half2float(get_error(cuFFT_result[pos].x, smFFT_result[pos].x));
			error_img = __half2float(get_error(cuFFT_result[pos].y, smFFT_result[pos].y));
			if (error_real >= error_img) error = error_real; else error = error_img;
			if (error > max_error) {
				//printf("Error=%f; cuFFT=[%f;%f] smFFT=[%f;%f] pos=%d\n", error, cuFFT_result[pos].x, cuFFT_result[pos].y, smFFT_result[pos].x, smFFT_result[pos].y, pos);
				cislo++;
				nErrors++;
			}
			dtemp += error;
		}
	}
	*cumulative_error = dtemp;
	*mean_error = dtemp / (float)(FFT_size * nFFTs);
	return(nErrors);
}*/


int GPU_smFFT_4elements(half2* h_input, half2* h_output, int FFT_size, int nFFTs, bool inverse, bool reorder, int nRuns, double* single_ex_time, double* multi_ex_time);

//int GPU_cuFFT(half2* h_input, half2* h_output, int FFT_size, int nFFTs, bool inverse, int nRuns, half* single_ex_time);

/*
int main(int argc, char* argv[]) {
	if (argc != 6) {
		printf("Argument error!\n");
		printf(" 1) FFT length\n");
		printf(" 2) number of FFTs\n");
		printf(" 3) the number of kernel executions\n");
		printf(" 4) do inverse FFT 1=yes 0=no\n");
		printf(" 5) reorder elements to correct order 1=yes 0=no\n");
		printf("For example: FFT.exe 1024 100000 20 0 1\n");
		return(1);
	}
	char* pEnd;

	int FFT_size = strtol(argv[1], &pEnd, 10);
	int nFFTs = strtol(argv[2], &pEnd, 10);
	int nRuns = strtol(argv[3], &pEnd, 10);
	int i_inverse = strtol(argv[4], &pEnd, 10);
	int i_reorder = strtol(argv[5], &pEnd, 10);

	bool inverse = (i_inverse == 1 ? true : false);
	bool reorder = (i_reorder == 1 ? true : false);

	if (FFT_size == 32) {
		printf("FFT length is 32 making sure that the number of FFTs is divisible by 4. ");
		int itemp = (int)((nFFTs + 4 - 1) / 4);
		nFFTs = itemp * 4;
		printf("New number of FFTs is %d.\n", nFFTs);
	}
	if (FFT_size == 64) {
		printf("FFT length is 64 making sure that the number of FFTs is divisible by 2. ");
		int itemp = (int)((nFFTs + 2 - 1) / 2);
		nFFTs = itemp * 2;
		printf("New number of FFTs is %d.\n", nFFTs);
	}

	int input_size = nFFTs * FFT_size;
	int output_size = nFFTs * FFT_size;


	half2* h_input;
	half2* h_output_smFFT;
	half2* h_output_cuFFT;

	if (DEBUG) printf("Host memory allocation...\t");
	h_input = (half2*)malloc(input_size * sizeof(half2));
	h_output_smFFT = (half2*)malloc(output_size * sizeof(half2));
	h_output_cuFFT = (half2*)malloc(output_size * sizeof(half2));
	if (DEBUG) printf("done.\n");

	if (DEBUG) printf("Host memory memset...\t\t");
	memset(h_input, 0.0, input_size * sizeof(half2));
	memset(h_output_smFFT, 0.0, output_size * sizeof(half2));
	memset(h_output_cuFFT, 0.0, output_size * sizeof(half2));
	if (DEBUG) printf("done.\n");

	if (DEBUG) printf("Initializing data with random numbers...\t");
	srand(time(NULL));
	for (int f = 0; f < FFT_size * nFFTs; f++) {
		h_input[f].y = rand() / (float)RAND_MAX;
		h_input[f].x = rand() / (float)RAND_MAX;
	}
	if (DEBUG) printf("done.\n");

	//-----------> cuFFT
	half cuFFT_execution_time;
	GPU_cuFFT(h_input, h_output_cuFFT, FFT_size, nFFTs, inverse, nRuns, &cuFFT_execution_time);

	//-----------> custom FFT
	half smFFT_execution_time, smFFT_multiple_execution_time;
	GPU_smFFT_4elements(h_input, h_output_smFFT, FFT_size, nFFTs, inverse, reorder, nRuns, &smFFT_execution_time, &smFFT_multiple_execution_time);

	if (reorder) {
		half cumulative_error, mean_error;
		int nErrors = 0;
		nErrors = Compare_data(h_output_cuFFT, h_output_smFFT, FFT_size, nFFTs, &cumulative_error, &mean_error);
		if (nErrors == 0) printf("  FFT test:\033[1;32mPASSED\033[0m\n");
		else printf("  FFT test:\033[1;31mFAILED\033[0m\n");
	}
	else {
		printf("  There is no verification of the results if FFT are not reordered.\n");
	}

	free(h_input);
	free(h_output_smFFT);
	free(h_output_cuFFT);

	cudaDeviceReset();

	return (0);
}
*/