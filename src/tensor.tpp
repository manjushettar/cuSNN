#include "tensor.h"
#include "kernels.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

template<typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape) 
	: host_data_(nullptr), device_data_(nullptr), shape_(shape){
	size_ = calculateSize(shape_);
	allocateMemory();
}

template<typename T>
Tensor<T>::Tensor(const Tensor &other)
	: shape_(other.shape_), size_(other.size_){
	allocateMemory();
	std::copy(other.host_data_, other.host_data_ + size_, host_data_);
	cudaMemcpy(device_data_, other.device_data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
}

template<typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
	: shape_(other.shape_), size_(other.size_), host_data_(other.host_data_), device_data_(other.device_data_){
	other.host_data_ = nullptr;
	other.device_data_ = nullptr;
	other.size_ = 0;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other){
	if(this != &other){
		freeMemory();
		shape_ = other.shape_;
		size_ = other.size_;
		
		allocateMemory();
		std::copy(other.host_data_, other.host_data_ + size_, host_data_);
		cudaMemcpy(device_data_, other.device_data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice); 
	}
	return *this;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept{
	if(this != &other){
		freeMemory();
		host_data_ = other.host_data_;
		device_data_ = other.device_data_;
		
		shape_ = std::move(other.shape_);
		size_ = other.size_;
		

		other.host_data_ = nullptr;
		other.device_data_ = nullptr;
		other.size_ = 0;
	}	
	return *this;
}

template<typename T>
Tensor<T>::~Tensor(){
	freeMemory();
}

template<typename T>
void Tensor<T>::allocateMemory(){
	host_data_ = new T[size_];
	cudaError_t err = cudaMalloc(&device_data_, size_ * sizeof(T));
	if(err != cudaSuccess){
		throw std::runtime_error("Failed to allocate device memory");
	}
}

template<typename T>
void Tensor<T>::freeMemory(){
	delete[] host_data_;
	cudaFree(device_data_);
}


template<typename T>
size_t Tensor<T>::calculateSize(const std::vector<size_t>& shape) const{
	size_t total_size = 1;
	for(size_t dim: shape){
		total_size *= dim;
	}
	return total_size;
}

template<typename T>
T* Tensor<T>::data(){
	return host_data_;
}

template<typename T>
const T* Tensor<T>::data() const{
	return host_data_;
}

template<typename T>
T* Tensor<T>::device_data(){
	return device_data_;
}

template<typename T>
const T* Tensor<T>::device_data() const{
	return device_data_;
}

template<typename T>
void Tensor<T>::toDevice(cudaStream_t stream){
	cudaMemcpyAsync(device_data_, host_data_, size_ * sizeof(T), cudaMemcpyHostToDevice, stream);
}

template<typename T>
void Tensor<T>::toHost(cudaStream_t stream){
	cudaMemcpyAsync(host_data_, device_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
}

template<typename T>
const std::vector<size_t>& Tensor<T>::shape() const{
	return shape_;
}

template<typename T>
size_t Tensor<T>::size() const{
	return size_;
}

template<typename T>
T& Tensor<T>::operator[](size_t i){
	size_t index = i;
	if(index >= size_){
		throw std::out_of_range("Tensor index out of range");
	}
	return host_data_[index];
}

template<typename T>
const T& Tensor<T>::operator[](size_t i) const{
	size_t index = i;
	if(index >= size_){
		throw std::out_of_range("Tensor index out of range");
	}
	return host_data_[index];
}

template<typename T>
T& Tensor<T>::operator()(size_t i, size_t j){
	size_t index = i * shape_[1] + j;
	if(index >= size_){
		throw std::out_of_range("Tensor index out of range");
	}
	return host_data_[index];
}

template<typename T>
const T& Tensor<T>::operator()(size_t i, size_t j) const{
	size_t index = i * shape_[1] + j;
	if(index >= size_){
		throw std::out_of_range("Tensor index out of range");
	}
	return host_data_[index];
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& out){
	size_t size = out.size();
	os << "{";
	for(size_t index = 0; index < size; index++){
		if(index == size - 1){
			os << out[index];
		}else{
			os << out[index] << ", ";
		}
	}	
	os << "}";
	return os;
} 


template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& other){
	size_t m, k, n;
	m = shape_[0];
	k = shape_[1];
	n = other.shape_[1];
	
	if(m == other.shape_[0] && k != n){
		Tensor<T> result(shape_);
		if(this->device_data_ && other.device_data_){
			addTensorAndVector(*this, other, result);
			cudaDeviceSynchronize();
			cudaMemcpy(result.host_data_, result.device_data_, m * k * sizeof(T), cudaMemcpyDeviceToHost);
		}
		return result;
	}
	
	if(size_ != other.size_){
		throw std::out_of_range("Tensor values do not line up.");
	}
	Tensor<T> result(shape_);

	if(this->device_data_ && other.device_data_){
		addTwoTensors(*this, other, result);
		cudaDeviceSynchronize();
		cudaMemcpy(result.host_data_, result.device_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return result;
}


template<typename T>
Tensor<T> Tensor<T>::operator+(const float scalar){
	Tensor<T> result(shape_);

	if(this->device_data_){
		addScalar(*this, result, scalar);
		cudaDeviceSynchronize();
		cudaMemcpy(result.host_data_, result.device_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return result;
}


template<typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T>& other){
	if(size_ != other.size_){
		throw std::out_of_range("Tensor values do not line up.");
	}	
	Tensor<T> result(shape_);
	if(this->device_data_ && other.device_data_){
		subTwoTensors(*this, other, result);
		cudaDeviceSynchronize();
		cudaMemcpy(result.host_data_, result.device_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator-(const float scalar){
	Tensor<T> result(shape_);

	if(this->device_data_){
		subScalar(*this, result, scalar);
		cudaDeviceSynchronize();
		cudaMemcpy(result.host_data_, result.device_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return result;
}



template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& other){
	size_t m, n, k;
	m = shape_[0];
	n = other.shape_[1];
	k = other.shape_[0];
	
	if(k != shape_[1]){
		throw std::runtime_error("Invalid mm sizes.");
	}
	Tensor<T> result({m, n});
	if(this->device_data_ && other.device_data_){
		mulTwoTensors(*this, other, result);
		cudaDeviceSynchronize();
		cudaMemcpy(result.host_data_, result.device_data_, m * n * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return result;
}


template<typename T>
Tensor<T> Tensor<T>::operator*(const float scalar){
	Tensor<T> result(shape_);

	if(this->device_data_){
		mulScalar(*this, result, scalar);
		cudaDeviceSynchronize();
		cudaMemcpy(result.host_data_, result.device_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return result;
}



template<typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T>& other){
	if(size_ != other.size_){
		throw std::out_of_range("Tensor values do not line up.");
	}	
	Tensor<T> result(shape_);
	if(this->device_data_ && other.device_data_){
		divTwoTensors(*this, other, result);
		cudaDeviceSynchronize();
		cudaMemcpy(result.host_data_, result.device_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return result;
}


template<typename T>
Tensor<T> Tensor<T>::operator/(const float scalar){
	Tensor<T> result(shape_);

	if(this->device_data_){
		divScalar(*this, result, scalar);
		cudaDeviceSynchronize();
		cudaMemcpy(result.host_data_, result.device_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return result;
}


template<typename T>
Tensor<T> Tensor<T>::operator+=(const Tensor<T>& other){
	if(size_ != other.size_){
		throw std::out_of_range("Tensor values do not line up.");
	}
	if(this->device_data_ && other.device_data_){
		addTwoTensors(*this, other, *this);
		cudaDeviceSynchronize();
		cudaMemcpy(this->host_data_, this->device_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return *this;

}

template<typename T>
Tensor<T> Tensor<T>::operator+=(const float scalar){
	Tensor<T> result(shape_);

	if(this->device_data_){
		subScalar(*this, *this, scalar);
		cudaDeviceSynchronize();
		cudaMemcpy(result.host_data_, result.device_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return result;
}


template<typename T>
Tensor<T> Tensor<T>::operator-=(const Tensor<T>& other){
	if(size_ != other.size_){
		throw std::out_of_range("Tensor values do not line up.");
	}
	if(this->device_data_ && other.device_data_){
		subTwoTensors(*this, other, *this);
		cudaDeviceSynchronize();
		cudaMemcpy(this->host_data_, this->device_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return *this;
}


template<typename T>
Tensor<T> Tensor<T>::operator-=(const float scalar){
	Tensor<T> result(shape_);

	if(this->device_data_){
		subScalar(*this, *this, scalar);
		cudaDeviceSynchronize();
		cudaMemcpy(result.host_data_, result.device_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return result;
}



template<typename T>
Tensor<T> Tensor<T>::operator*=(const Tensor<T>& other){
	if(size_ != other.size_){
		throw std::out_of_range("Tensor values do not line up.");
	}
	if(this->device_data_ && other.device_data_){
		mulTwoTensors(*this, other, *this);
		cudaDeviceSynchronize();
		cudaMemcpy(this->host_data_, this->device_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return *this;
}

template<typename T>
Tensor<T> Tensor<T>::operator*=(const float scalar){
	Tensor<T> result(shape_);

	if(this->device_data_){
		mulScalar(*this, *this, scalar);
		cudaDeviceSynchronize();
		cudaMemcpy(result.host_data_, result.device_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return result;
}



template<typename T>
Tensor<T> Tensor<T>::operator/=(const Tensor<T>& other){
	if(size_ != other.size_){
		throw std::out_of_range("Tensor values do not line up.");
	}
	if(this->device_data_ && other.device_data_){
		divTwoTensors(*this, other, *this);
		cudaDeviceSynchronize();
		cudaMemcpy(this->host_data_, this->device_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return *this;
}


template<typename T>
Tensor<T> Tensor<T>::operator/=(const float scalar){
	Tensor<T> result(shape_);

	if(this->device_data_){
		divScalar(*this, *this, scalar);
		cudaDeviceSynchronize();
		cudaMemcpy(result.host_data_, result.device_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return result;
}

template<typename T>
Tensor<T> Tensor<T>::randn(const std::vector<size_t>& shape){
	Tensor<T> result(shape);
	fillRandom(result, result.size());
	cudaDeviceSynchronize();
	cudaMemcpy(result.data(), result.device_data(), result.size() * sizeof(T), cudaMemcpyDeviceToHost);
	return result;
} 

template<typename T>
Tensor<T> Tensor<T>::forwardPass(const Tensor<T>& in, const Tensor<T>& weights, const Tensor<T>& bias){
	size_t in_m, in_k;
	in_m = in.shape()[0];
	in_k = in.shape()[1];
	
	size_t weights_k, weights_n;
	weights_k = weights.shape()[0];
	weights_n = weights.shape()[1];
	
	size_t bias_m;
	bias_m = bias.shape()[0];

	if(in_k != weights_k) throw std::runtime_error("bad forward in/weights");
	if(bias_m != in_m) throw std::runtime_error("bad forward bias");
	
	Tensor<T> result({in_m, weights_n});
	if(in.device_data_ && weights.device_data_ && bias.device_data_){
		forwardCall(in, weights, bias, result);
		cudaDeviceSynchronize();
		cudaMemcpy(result.host_data_, result.device_data_, in_m * weights_n * sizeof(T), cudaMemcpyDeviceToHost); 
	}
	return result;
} 

template class Tensor<float>;



