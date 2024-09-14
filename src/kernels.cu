#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>

#include "tensor.h"
#include "kernels.h"


template<typename T>
__global__ void addT(const T* a, const T* b, T* result, size_t size){
	size_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size){
		result[index] = a[index] + b[index];
	}
}

template<typename T>
__global__ void subT(const T* a, const T* b, T* result, size_t size){
	size_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size){
		result[index] = a[index] - b[index];
	}
}

template<typename T>
__global__ void mulT(const T* a, const T* b, T* result, size_t size){
	size_t index = threadIdx.x + blockDim.x * blockIdx.x;
	if(index < size){
		result[index] = a[index] * b[index];
	}
}


template<typename T>
__global__ void divT(const T* a, const T* b, T* result, size_t size){
	size_t index = threadIdx.x + blockDim.x * blockIdx.x;
	if(index < size){
		result[index] = a[index] / b[index];
	}
}

template<typename T>
__global__ void addS(const T* a, const float scalar, T* result, size_t size){
	size_t index = threadIdx.x + blockDim.x * blockIdx.x;
	if(index < size){
		result[index] = a[index] + scalar;
	}
}


template<typename T>
__global__ void subS(const T* a, const float scalar, T* result, size_t size){
	size_t index = threadIdx.x + blockDim.x * blockIdx.x;
	if(index < size){
		result[index] = a[index] - scalar;
	}
}


template<typename T>
__global__ void mulS(const T* a, const float scalar, T* result, size_t size){
	size_t index = threadIdx.x + blockDim.x * blockIdx.x;
	if(index < size){
		result[index] = a[index] * scalar;
	}
}


template<typename T>
__global__ void divS(const T* a, const float scalar, T* result, size_t size){
	size_t index = threadIdx.x + blockDim.x * blockIdx.x;
	if(index < size){
		result[index] = a[index] / scalar;
	}
}


template<typename T>
void addTwoTensors(const Tensor<T>& tensor1, const Tensor<T>& tensor2, Tensor<T>& tensor3){
	size_t size1 = tensor1.size();
	size_t size2 = tensor2.size();

	size_t size3 = tensor3.size();
	if(size1 != size2 && size1 != size3){
		throw std::runtime_error("Invalid sizes for add tensors");
	}

	dim3 tpb(16, 16);
	dim3 bpg((size3 + tpb.x - 1)/tpb.x, (size3 + tpb.y - 1) / tpb.y);

	addT<<<bpg, tpb>>>(tensor1.device_data(), tensor2.device_data(), tensor3.device_data(), size3);

	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		throw std::runtime_error("CUDA addTensors kernel launch failed.");
	}
}

template<typename T>
void subTwoTensors(const Tensor<T>& tensor1, const Tensor<T>& tensor2, Tensor<T>& tensor3){
	size_t size1 = tensor1.size();
	size_t size2 = tensor2.size();
	size_t size3 = tensor3.size();
	if(size1 != size2 && size1 != size3){
		throw std::runtime_error("Invalid sizes for subtract tensors");
	}
	dim3 tpb(16, 16);
	dim3 bpg((size3 + tpb.x - 1) / tpb.x, (size3 + tpb.y - 1) / tpb.y);

	
	subT<<<bpg, tpb>>>(tensor1.device_data(), tensor2.device_data(), tensor3.device_data(), size3);
	
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		throw std::runtime_error("CUDA subTensors kernel launch failed.");
	}
}


template<typename T>
void mulTwoTensors(const Tensor<T>& tensor1, const Tensor<T>& tensor2, Tensor<T>& tensor3){
	size_t size1 = tensor1.size();
	size_t size2 = tensor2.size();
	size_t size3 = tensor3.size();
	if(size1 != size2 && size1 != size3){
		throw std::runtime_error("Invalid sizes for subtract tensors");
	}
	dim3 tpb(16, 16);
	dim3 bpg((size3 + tpb.x - 1) / tpb.x, (size3 + tpb.y - 1) / tpb.y);

	
	mulT<<<bpg, tpb>>>(tensor1.device_data(), tensor2.device_data(), tensor3.device_data(), size3);
	
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		throw std::runtime_error("CUDA subTensors kernel launch failed.");
	}
}


template<typename T>
void divTwoTensors(const Tensor<T>& tensor1, const Tensor<T>& tensor2, Tensor<T>& tensor3){
	size_t size1 = tensor1.size();
	size_t size2 = tensor2.size();
	size_t size3 = tensor3.size();
	if(size1 != size2 && size1 != size3){
		throw std::runtime_error("Invalid sizes for subtract tensors");
	}
	dim3 tpb(16, 16);
	dim3 bpg((size3 + tpb.x - 1) / tpb.x, (size3 + tpb.y - 1) / tpb.y);

	
	divT<<<bpg, tpb>>>(tensor1.device_data(), tensor2.device_data(), tensor3.device_data(), size3);
	
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		throw std::runtime_error("CUDA subTensors kernel launch failed.");
	}
}

template<typename T>
void addScalar(const Tensor<T>& tensor1, Tensor<T>& tensor3, const float scalar){
	size_t size1 = tensor1.size();
	size_t size3 = tensor3.size();
	if(size1 != size3){
		throw std::runtime_error("Invalid sizes for add scalar.");
	}
	dim3 tpb(16, 16);
	dim3 bpg((size3 + tpb.x - 1) / tpb.x, (size3 + tpb.y - 1) / tpb.y);
	
	addS<<<bpg, tpb>>>(tensor1.device_data(), scalar, tensor3.device_data(), size1);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		throw std::runtime_error("CUDA add scalar kernel launch failed.");
	}
}	


template<typename T>
void subScalar(const Tensor<T>& tensor1, Tensor<T>& tensor3, const float scalar){
	size_t size1 = tensor1.size();
	size_t size3 = tensor3.size();
	if(size1 != size3){
		throw std::runtime_error("Invalid sizes for add scalar.");
	}
	dim3 tpb(16, 16);
	dim3 bpg((size3 + tpb.x - 1) / tpb.x, (size3 + tpb.y - 1) / tpb.y);
	
	subS<<<bpg, tpb>>>(tensor1.device_data(), scalar, tensor3.device_data(), size1);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		throw std::runtime_error("CUDA add scalar kernel launch failed.");
	}
}	


template<typename T>
void mulScalar(const Tensor<T>& tensor1, Tensor<T>& tensor3, const float scalar){
	size_t size1 = tensor1.size();
	size_t size3 = tensor3.size();
	if(size1 != size3){
		throw std::runtime_error("Invalid sizes for add scalar.");
	}
	dim3 tpb(16, 16);
	dim3 bpg((size3 + tpb.x - 1) / tpb.x, (size3 + tpb.y - 1) / tpb.y);
	
	mulS<<<bpg, tpb>>>(tensor1.device_data(), scalar, tensor3.device_data(), size1);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		throw std::runtime_error("CUDA add scalar kernel launch failed.");
	}
}	


template<typename T>
void divScalar(const Tensor<T>& tensor1, Tensor<T>& tensor3, const float scalar){
	size_t size1 = tensor1.size();
	size_t size3 = tensor3.size();
	if(size1 != size3){
		throw std::runtime_error("Invalid sizes for add scalar.");
	}
	dim3 tpb(16, 16);
	dim3 bpg((size3 + tpb.x - 1) / tpb.x, (size3 + tpb.y - 1) / tpb.y);
	
	divS<<<bpg, tpb>>>(tensor1.device_data(), scalar, tensor3.device_data(), size1);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		throw std::runtime_error("CUDA add scalar kernel launch failed.");
	}
}	

