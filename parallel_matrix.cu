#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <random>
#include<chrono>
#include <fstream>
#include"seq_matrix.h"

constexpr int threads_per_block = 32;

// make sure that diag elements are not 0
__global__ void non_zero_diag(float* matrix, float* identity, int N, int r) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < N && i == r) {
		if (matrix[i * N + i] == 0) {
			for (int k = i + 1; k < N; k++) {
				if (matrix[k * N + i] != 0.0) {
					matrix[i * N + j] += matrix[k * N + j];
					return;
				}
			}
		}
	}
}

// divide row elements by diag element
__global__ void normalize(float* matrix, float* identity, int N, int r) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < N) {
		if (i == r && i != j) {
			identity[i * N + j] /= matrix[i * N + i];
			matrix[i * N + j] /= matrix[i * N + i];
		}
		__syncthreads();
		if (i == r && i == j) {
			identity[i * N + j] /= matrix[i * N + i];
			matrix[i * N + j] /= matrix[i * N + i];
		}
	}
}

__global__ void matrix_inverse(float* matrix, float* identity, int N, int r) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < N) {
		if (i != r) {
			identity[i * N + j] -= identity[r * N + j] * matrix[i * N + r];
			if (j != r) {
 				matrix[i * N + j] -= matrix[r * N + j] * matrix[i * N + r];
			}
		}
	}
}

void run_test(size_t size, std::ofstream& file) {
	std::cout << '\n' << "Size: " << size << std::endl;

	// Prepare CUDA layout
	dim3 block_structure(threads_per_block, threads_per_block);
	dim3 grid_structure((size + threads_per_block - 1) / threads_per_block, (size + threads_per_block - 1) / threads_per_block);

	// Generate matricies
	auto input_matrix = generate_matrix(size);
	auto identity = create_identity(size);

	// Prepare host memory
	size_t size_bytes = size * size * sizeof(float);
	float* h_A = new float[size * size];
	float* h_I = new float[size * size];

	// copy generated matrix for parrell computation
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			h_A[i * size + j] = input_matrix[i][j];
			h_I[i * size + j] = identity[i][j];
		}
	}
	
	// Solve sequentially
	auto start_seq = std::chrono::high_resolution_clock::now();
	calculate_inverse_seq(input_matrix, identity);

	auto duration_seq = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_seq).count();

	std::cout 
		<< "Sequential inverse time: " 
		<< '\n'
		<< duration_seq
		<< " ms " 
		<< std::endl;

	// device memory
	float* d_A;
	float* d_I;
	cudaMalloc((void**)&d_A, size_bytes);
	cudaMalloc((void**)&d_I, size_bytes);

	// load device memory from host memory
	cudaMemcpy(d_A, h_A, size_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_I, h_I, size_bytes, cudaMemcpyHostToDevice);

	// run pararell algorithm
	cudaEvent_t start_pararell, stop_pararell;
	cudaEventCreate(&start_pararell);
	cudaEventCreate(&stop_pararell);
	cudaEventRecord(start_pararell, 0);
	float duration_par;

	for (int row = 0; row < size; row++) {
		non_zero_diag << <grid_structure, block_structure >> > (d_A, d_I, size, row);
		cudaDeviceSynchronize();
		normalize << <grid_structure, block_structure >> > (d_A, d_I, size, row);
		cudaDeviceSynchronize();
		matrix_inverse << <grid_structure, block_structure >> > (d_A, d_I, size, row);
		cudaDeviceSynchronize();
	}

	cudaEventRecord(stop_pararell, 0);
	cudaEventSynchronize(stop_pararell);
	cudaEventElapsedTime(&duration_par, start_pararell, stop_pararell);
	cudaEventDestroy(start_pararell);
	cudaEventDestroy(stop_pararell);

	std::cout
		<< "Pararell inverse time: "
		<< '\n'
		<< duration_par 
		<< " ms "
		<< std::endl;

	// get results
	cudaMemcpy(h_A, d_A, size_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_I, d_I, size_bytes, cudaMemcpyDeviceToHost);

	// save to csv file
	file << duration_par << ";" << duration_seq << std::endl;

	cudaFree(d_A);
	cudaFree(d_I);
	delete[] h_A;
	delete[] h_I;
}

int main(void) {
	std::locale pol("pl_PL");
	std::ofstream file("test.csv");
	file.imbue(pol);
	for (size_t size = 1; size <= 100; size++)
		run_test(size, file);
	file.close();
	return 0;
}
