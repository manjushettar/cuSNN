#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <curand_kernel.h>
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
__global__ void mulT(const T* a, const T* b, T* result, size_t m, size_t n, size_t k){
	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	if(row < m && col < n){
		T dotProd = 0;
		for(int i = 0; i < k; i++){
			dotProd += a[row * k + i] * b[i * n + col];
		}
		result[row * n + col] = dotProd;
	}
}


template<typename T>
__global__ void addTV(const T* a, const T* b, T* result, size_t m, size_t n, size_t k){
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
		
	if(row < m && col < k){
		result[row * k + col] = a[row * k + col] + b[row];
	}
}

template<typename T>
void addTensorAndVector(const Tensor<T>& tensor1, const Tensor<T>& vector1, Tensor<T>& tensor3){
	size_t m, k, n;
	m = tensor1.shape()[0];
	k = tensor1.shape()[1];
	n = vector1.shape()[1];
	if(n != 1) throw std::runtime_error("Matrix-vector adding broadcasting issue.");
	
	if(m != tensor3.shape()[0] || k != tensor3.shape()[1]) throw std::runtime_error("Result tensor shape mismatch.");

	dim3 tpb(16, 16);
	dim3 bpg((k + tpb.x - 1) / tpb.x, (m + tpb.y - 1) / tpb.y);
	addTV<<<bpg, tpb>>>(tensor1.device_data(), vector1.device_data(), tensor3.device_data(), m, n, k);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		throw std::runtime_error("CUDA tensor x vector failed.");
	}
}


template<typename T>
void mulTwoTensors(const Tensor<T>& tensor1, const Tensor<T>& tensor2, Tensor<T>& tensor3){
	size_t m, k, n;
	m = tensor1.shape()[0];
	n = tensor2.shape()[1];
	k = tensor1.shape()[1];
	
	if(k != tensor2.shape()[0]) throw std::runtime_error("Matrix dim do not match - k value.");
	if(m != tensor3.shape()[0] || n != tensor3.shape()[1]) throw std::runtime_error("Result matrix dims are off");
	dim3 tpb(16, 16);
	dim3 bpg((n + tpb.x - 1) / tpb.x, (m + tpb.y - 1) / tpb.y);

	mulT<<<bpg, tpb>>>(tensor1.device_data(), tensor2.device_data(), tensor3.device_data(), m, n, k);
	
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

template<typename T>
__global__ void randN(T *a, const size_t size, unsigned long long seed){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size){
		curandState state;
		curand_init(seed, index, 0, &state);
		a[index] = curand_normal(&state) * sqrtf(2.f/(size/2));
	}
}

template<typename T>
void fillRandom(Tensor<T>& tensor1, const size_t size){
	dim3 tpb(256);
	dim3 bpg((size + tpb.x - 1) / tpb.x);

	randN<<<bpg, tpb>>>(tensor1.device_data(), size, time(NULL));
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		throw std::runtime_error("CUDA randN kernel launch failed.");
	}
}

template<typename T>
__global__ void forwardP(const T* in, const T* weights, const T* bias, T *out, const size_t m, const size_t k, const size_t n){
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	if(row < m && col < n){
		T dotProd = 0.f;
		for(size_t i = 0; i < k; i++){
			dotProd += in[row * k + i] * weights[n * i + col];
		}
		out[row * n + col] = dotProd + bias[row];
	}
}

template<typename T>
__global__ void relu(const T* in, T* out, const size_t m, const size_t n){
	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;

	if(row < m && col < n){
		if(in[row * n + col] <= 0) out[row * n + col] = 0;
		else out[row * n + col] = in[row * n + col];
	}
}

template<typename T>
void forwardCall(const Tensor<T>& in, const Tensor<T>& weights, const Tensor<T>& bias, Tensor<T>& out){
	size_t in_m, in_k;
	size_t w_k, w_n;
	in_m = in.shape()[0];
	in_k = in.shape()[1];
	w_k = weights.shape()[0];
	w_n = weights.shape()[1];
	
	size_t bias_m = bias.shape()[0];
	size_t bias_k = bias.shape()[1];

	dim3 tpb(16, 16);
	dim3 bpg( (w_n + tpb.x - 1) / tpb.x, (in_m + tpb.y - 1) / tpb.y);

	forwardP<<<bpg, tpb>>>(in.device_data(), weights.device_data(), bias.device_data(), out.device_data(), in_m, in_k, w_n);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		std::cerr << "CUDA ERR: " << cudaGetErrorString(err) << std::endl;
		throw std::runtime_error("forward kernel failed");
	}
}

template<typename T>
void reluCall(const Tensor<T>& in, Tensor<T>& out){
	const size_t in_m = in.shape()[0];
	const size_t in_n = in.shape()[1];
	
	dim3 tpb(16,16);
	dim3 bpg((in_n + tpb.x - 1) / tpb.x, (in_m + tpb.y - 1) / tpb.y);
	
	relu<<<bpg, tpb>>>(in.device_data(), out.device_data(), in_m, in_n);

	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		std::cerr << "CUDA ERR: " << cudaGetErrorString(err) << std::endl;
		throw std::runtime_error("forward kernel failed");
	}
}

template<typename T>
__global__ void tanh(const T* in, T* out, const size_t m, const size_t n){
	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;

	if(row < m && col < n){
		T val = in[row * n + col];
		T numerator = (exp(val) - exp(-1.0 * val));
		T denominator = (exp(val) + exp(-1.0 * val));

		out[row * n + col] = numerator / denominator;
	}
}



template<typename T>
void tanhCall(const Tensor<T>& in, Tensor<T>& out){
	const size_t in_m = in.shape()[0];
	const size_t in_n = in.shape()[1];
	
	dim3 tpb(16,16);
	dim3 bpg((in_n + tpb.x - 1) / tpb.x, (in_m + tpb.y - 1) / tpb.y);
	
	tanh<<<bpg, tpb>>>(in.device_data(), out.device_data(), in_m, in_n);

	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		std::cerr << "CUDA ERR: " << cudaGetErrorString(err) << std::endl;
		throw std::runtime_error("forward kernel failed");
	}
}
template<typename T>
__global__ void softmax(const T* in, T* out , const size_t m, const size_t n){
	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	if(row < m && col < n){
                float max_val = in[row * n];
                for(int i = 1; i < n; i++){
                        max_val = max(max_val, in[row * n + i]);
                }
                float divisor = 0.0f;
                for(int i = 0; i < n; i++){
                        divisor += exp(in[row * n + i] - max_val);
                }
                out[row * n + col] = exp(in[row * n + col] - max_val) / divisor;
        }
}

template<typename T>
void softmaxCall(const Tensor<T>& in, Tensor<T>& out){
	const size_t m = in.shape()[0];
	const size_t n = in.shape()[1];

	dim3 tpb(16, 16);
	dim3 bpg((tpb.x + n - 1) / tpb.x, (tpb.y + m - 1) / tpb.y);

	softmax<<<bpg, tpb>>>(in.device_data(), out.device_data(), m, n);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		throw std::runtime_error("softmax kernel failed.");
	}
}



template<typename T>
__global__ void ce_loss(const T* in, const T* labels, T* out, const size_t m, const size_t n){
	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	if(row < m){
		float loss = 0.f;
 		for(int i = 0; i < n; i++){
			float prob = max(1e-6f, in[row * n + i]);
			loss -= labels[row * n + i] * logf(prob);
		}
		out[row] = loss;	
	}
}

template<typename T>
void crossEntropyLoss(const Tensor<T>& in, const Tensor<T> &labels, Tensor<T>& out){
	// labels -> in.shape()[0] * in.shape()[1]
	const size_t m = in.shape()[0];
	const size_t n = in.shape()[1];

	dim3 tpb(16, 16);
	dim3 bpg( (tpb.x + n - 1) / tpb.x, (tpb.y + m - 1) / tpb.y);

	ce_loss<<<bpg, tpb>>>(in.device_data(), labels.device_data(), out.device_data(), m, n);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		throw std::runtime_error("cross entropy loss kernel failed.");
	}

}


template<typename T>
__global__ void ce_lossBackward(const T* in, const T* labels, T* out, const size_t m, const size_t n){
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < m && col < n){
		out[row * n + col] = in[row * n + col] - labels[row * n + col];
	}
}

template<typename T>
void crossEntropyLossBackward(const Tensor<T>& in, const Tensor<T>& labels, Tensor<T>& out){
	const size_t m = in.shape()[0];
	const size_t n = in.shape()[1];

	dim3 tpb(16, 16);
	dim3 bpg( (tpb.x + n - 1) / tpb.x, (tpb.y + m - 1) / tpb.y);

	ce_lossBackward<<<bpg, tpb>>>(in.device_data(), labels.device_data(), out.device_data(), m, n);
	cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess){
                throw std::runtime_error("cross entropy loss backward kernel failed.");
        }	
}

template<typename T>
__global__ void relu_backward(const T* in, const T* grad_out, T* grad_in, const size_t m, const size_t n) {
    	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
    	size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	    if (row < m && col < n) {
        	grad_in[row * n + col] = (in[row * n + col] > 0) ? grad_out[row * n + col] : 0;
    	}
}

template<typename T>
void reluBackwardCall(const Tensor<T>& in, const Tensor<T>& grad_out, Tensor<T>&grad_in){
	const size_t m = in.shape()[0];
    	const size_t n = in.shape()[1];

    	dim3 tpb(16, 16);
    	dim3 bpg((n + tpb.x - 1) / tpb.x, (m + tpb.y - 1) / tpb.y);

    	relu_backward<<<bpg, tpb>>>(in.data(), grad_out.data(), grad_in.data(), m, n);

    	cudaError_t err = cudaGetLastError();
    	if (err != cudaSuccess) {
        	throw std::runtime_error("ReLU backward kernel failed.");
    	}
}


template<typename T>
__global__ void softmax_backward(const T* softmax_out, const T* labels, T* grad_in, const size_t m, const size_t n){
	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
    	size_t col = threadIdx.x + blockDim.x * blockIdx.x;

    	if (row < m && col < n) {
        	grad_in[row * n + col] = softmax_out[row * n + col] - labels[row * n + col];
    	}
}

template<typename T>
void softmaxBackward(const Tensor<T>& softmax_out, const Tensor<T>& labels, Tensor<T>& grad_in){
	const size_t m = softmax_out.shape()[0];
	const size_t n = softmax_out.shape()[1];

    	dim3 tpb(16, 16);
    	dim3 bpg((n + tpb.x - 1) / tpb.x, (m + tpb.y - 1) / tpb.y);

    	softmax_backward<<<bpg, tpb>>>(softmax_out.data(), labels.data(), grad_in.data(), m, n);

    	cudaError_t err = cudaGetLastError();
  	  if (err != cudaSuccess) {
       		throw std::runtime_error("Softmax backward kernel failed.");
    	}
}

template<typename T>
__global__ void backward(const T* in, const T* weights, const T* bias, const T* grad_out, T* grad_in, T* grad_weights, T* grad_bias, const size_t m, const size_t n, const size_t k){
	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
        size_t col = threadIdx.x + blockDim.x * blockIdx.x;

        if(row < m && col < k){
                T grad_w = 0.0f;
                for(int i = 0; i < n; i++){
                        grad_w += in[row * n + i] * grad_in[row * k + col];
                }
                grad_weights[row * k + col] = grad_w;
		
		if(row == 0){
                	T grad_b = 0.0f;
         		for(int i = 0; i < m; i++){
				grad_b += grad_out[i * k + col];
			}
			grad_bias[col] = grad_b;
		}

                T grad_x = 0.0f;
                for(int i = 0; i < k; i++){
                        grad_x += weights[col * k + i] * grad_in[row * k + i];
                }
                grad_in[row * n + col] = grad_x;
        }
}


template<typename T>
void backwardPass(const Tensor<T>& in, const Tensor<T>& weights, const Tensor<T>& bias,const Tensor<T>& grad_out, Tensor<T>& d_in, Tensor<T>& d_weights, Tensor<T>& d_bias) {
	const size_t m = in.shape()[0];
	const size_t n = in.shape()[1];
	const size_t k = weights.shape()[1];
	dim3 tpb(16, 16);
   	dim3 bpg((n + tpb.x - 1) / tpb.x, (m + tpb.y - 1) / tpb.y);

    	backward<<<bpg, tpb>>>(in.device_data(), weights.device_data(), bias.device_data(), grad_out.device_data(), d_in.device_data(), d_weights.device_data(), d_bias.device_data(),m, n, k);

    	cudaError_t err = cudaGetLastError();
    	if(err != cudaSuccess) {
		std::cerr << "Cuda error: " << cudaGetErrorString(err) << std::endl;
        	throw std::runtime_error("backward kernel failed.");
    	}
}
