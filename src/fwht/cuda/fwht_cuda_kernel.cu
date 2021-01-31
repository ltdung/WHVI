/*  Copyright (c) 2019
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License as published by the
 *  Free Software Foundation, either version 3 of the License, or (at your
 *  option) any later version.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 *  Authors:
 *      Simone Rossi <simone.rossi@eurecom.fr>
 *      Maurizio Filippone <maurizio.filippone@eurecom.fr>
 */


#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


// ELEMENTARY_LOG2SIZE can be changed to another positive integer.
#define ELEMENTARY_LOG2SIZE 11


/*
Single in-global memory FWHT pass.
For strides exceeding ELEMENTARY_LOG2SIZE.
*/
template <typename scalar_t>
__global__ void fwht_batch2_kernel(scalar_t* __restrict__ d_output, int stride) {
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int N   = blockDim.x * gridDim.x * 4;

    scalar_t *d_Src = d_output  + blockIdx.y * N;
    scalar_t *d_Dst = d_output + blockIdx.y * N;

    int lo = pos & (stride - 1);
    int i0 = ((pos - lo) << 2) + lo;
    int i1 = i0 + stride;
    int i2 = i1 + stride;
    int i3 = i2 + stride;

    scalar_t D0 = d_Src[i0];
    scalar_t D1 = d_Src[i1];
    scalar_t D2 = d_Src[i2];
    scalar_t D3 = d_Src[i3];

    scalar_t T;
    T = D0;
    D0        = D0 + D2;
    D2        = T - D2;
    T = D1;
    D1        = D1 + D3;
    D3        = T - D3;
    T = D0;
    d_Dst[i0] = D0 + D1;
    d_Dst[i1] = T - D1;
    T = D2;
    d_Dst[i2] = D2 + D3;
    d_Dst[i3] = T - D3;
}


/*
Elementary in-shared memory FWHT.
For strides below (or equal to) ELEMENTARY_LOG2SIZE.
*/
template <typename scalar_t>
__global__ void fwht_batch1_kernel(scalar_t* __restrict__ d_output, int log2d)
{
    const int N = 1 << log2d;
    const int base = blockIdx.x << log2d;

    extern __shared__ unsigned char shared_mem[];
    scalar_t *s_data = reinterpret_cast<scalar_t *>(shared_mem);

    scalar_t *d_Src = d_output + base;
    scalar_t *d_Dst = d_output + base;

    for (int pos = threadIdx.x; pos < N; pos += blockDim.x)
    {
        s_data[pos] = d_Src[pos];
    }

    //Main radix-4 stages
    const int pos = threadIdx.x;

    for (int stride = N >> 2; stride > 0; stride >>= 2)
    {
        int lo = pos & (stride - 1);
        int i0 = ((pos - lo) << 2) + lo;
        int i1 = i0 + stride;
        int i2 = i1 + stride;
        int i3 = i2 + stride;

        __syncthreads();
        scalar_t D0 = s_data[i0];
        scalar_t D1 = s_data[i1];
        scalar_t D2 = s_data[i2];
        scalar_t D3 = s_data[i3];

        scalar_t T;
        T = D0;
        D0         = D0 + D2;
        D2         = T - D2;
        T = D1;
        D1         = D1 + D3;
        D3         = T - D3;
        T = D0;
        s_data[i0] = D0 + D1;
        s_data[i1] = T - D1;
        T = D2;
        s_data[i2] = D2 + D3;
        s_data[i3] = T - D3;
    }

    //Do single radix-2 stage for odd power of two
    if (log2d & 1)
    {
        __syncthreads();

        for (int pos = threadIdx.x; pos < N / 2; pos += blockDim.x)
        {
            int i0 = pos << 1;
            int i1 = i0 + 1;

            scalar_t D0 = s_data[i0];
            scalar_t D1 = s_data[i1];
            s_data[i0] = D0 + D1;
            s_data[i1] = D0 - D1;
        }
    }

    __syncthreads();

    for (int pos = threadIdx.x; pos < N; pos += blockDim.x)
    {
        d_Dst[pos] = s_data[pos];
    }
}


/*
CPU front-end for batched FWHT on tensor X.
Creates the grid and launches kernels.

:param X: input tensor of shape (batch_size, D). This tensor is overwritten.
:return: overwritten tensor X with the result of batched FWHT.
*/
at::Tensor fwht_cuda_frontend(at::Tensor X) {
	const int num_threads = 256;  // This can be changed to a different power of two.

	auto shape = X.sizes();
	int batch_size = shape[0];
	int log2D = (int) log2((float) shape[1]);

	int D = 1 << log2D;
	int blocks_per_grid = D / (4 * num_threads);
	int threads_per_block = batch_size;
	dim3 grid(blocks_per_grid, threads_per_block, 1);  // Create a 3D grid.

	// Launch kernels
	for (; log2D > ELEMENTARY_LOG2SIZE; log2D -= 2, D >>= 2, batch_size <<= 2) {
		AT_DISPATCH_FLOATING_TYPES(X.type(), "fwht_batch2_kernel", ([&] {
			fwht_batch2_kernel<<<grid, num_threads>>>(X.data<scalar_t>(), D / 4);
		}));
	}

	// Launch kernel
	AT_DISPATCH_FLOATING_TYPES(X.type(), "fwht_batch2_kernel", ([&] {
		fwht_batch1_kernel<<<batch_size, D / 4, D * sizeof(scalar_t)>>>(X.data<scalar_t>(), log2D);
	}));

	return X;
}
