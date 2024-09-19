#include "tensor.h"
#include "kernels.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <sstream>


// default constructor
// host_data_ -> nullptr
// device_data_ -> nullptr
// shape_ -> input shape
// size_ -> calculated size of the tensor
	// Usage:
	// Tensor<float> t1({2, 3});
template<typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape) 
	: host_data_(nullptr), device_data_(nullptr), shape_(shape){
	size_ = calculateSize(shape_);
	allocateMemory();
}

// copy constructor - calculates size and shape from other tensor
// allocates new memory on host and device, copies host and device data from other tensor
// host_data_ -> other.host_data_
// device_data_ -> other.device_data_
// shape_ -> other.shape_
// size_ -> other.size_
	// Usage:
	// Tensor<float> t1({2, 3});
	// Tensor<float> t2(t1);
template<typename T>
Tensor<T>::Tensor(const Tensor &other)
	: shape_(other.shape_), size_(other.size_){
	allocateMemory();
	std::copy(other.host_data_, other.host_data_ + size_, host_data_);
	cudaMemcpy(device_data_, other.device_data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
}

// move constructor - takes ownership of resources from other tensor
// sets other tensor pointers to nullptr and size to 0
	// Usage:
	// Tensor<float> t1({2, 3});
	// Tensor<float> t2(std::move(t1));	
template<typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
	: shape_(other.shape_), size_(other.size_), host_data_(other.host_data_), device_data_(other.device_data_){
	other.host_data_ = nullptr;
	other.device_data_ = nullptr;
	other.size_ = 0;
}

// deallocates memory on host and device
	// Usage:
	// Tensor<float> t1({2, 3});
	// Tensor<float> t2(t1);
	// t1.~Tensor();
template<typename T>
Tensor<T>::~Tensor(){
	freeMemory();
}

// copy assignment operator - calculates size and shape from other tensor
// allocates new memory on host and device, copies host and device data from other tensor
// host_data_ -> other.host_data_
// device_data_ -> other.device_data_
// shape_ -> other.shape_
// size_ -> other.size_
	// Usage:
	// Tensor<float> t1({2, 3});
	// Tensor<float> t2({2, 3});
	// t2 = t1;
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

// move assignment operator - takes ownership of resources from other tensor
// sets other tensor pointers to nullptr and size to 0
	// Usage:
	// Tensor<float> t1({2, 3});
	// Tensor<float> t2({2, 3});
	// t2 = std::move(t1);
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

// allocates memory on host and device
// host_data_ -> new T[size_]
// device_data_ -> cudaMalloc(&device_data_, size_ * sizeof(T))
// size_ -> input size
template<typename T>
void Tensor<T>::allocateMemory() {
    host_data_ = new T[size_];
    cudaError_t err = cudaMalloc(&device_data_, size_ * sizeof(T));
    if (err != cudaSuccess) {
        std::stringstream ss;
        ss << "Failed to allocate device memory for tensor: "
           << "Shape: [";
        for (size_t i = 0; i < shape_.size(); ++i) {
            ss << shape_[i];
            if (i < shape_.size() - 1) ss << ", ";
        }
        ss << "], Size: " << size_
           << ", Bytes: " << (size_ * sizeof(T))
           << ". Error: " << cudaGetErrorString(err);
        throw std::runtime_error(ss.str());
    }
}

// deallocates memory on host and device
// deletes host_data_
// cudaFree(device_data_)
template<typename T>
void Tensor<T>::freeMemory(){
	delete[] host_data_;
	cudaFree(device_data_);
}

// calculates size of the tensor
// total_size = 1
// for each dimension in shape, multiply total_size by the dimension
// return total_size
template<typename T>
size_t Tensor<T>::calculateSize(const std::vector<size_t>& shape) const{
	size_t total_size = 1;
	for(size_t dim: shape){
		total_size *= dim;
	}
	return total_size;
}

// returns host_data_
// Usage:
// Tensor<float> t1({2, 3});
// t1.data(); -> pointer to host_data_
template<typename T>
T* Tensor<T>::data(){
	return host_data_;
}

// returns host_data_
// Usage:
// Tensor<float> t1({2, 3});
// t1.data(); -> pointer to host_data_
template<typename T>
const T* Tensor<T>::data() const{
	return host_data_;
}

// returns device_data_
// Usage:
// Tensor<float> t1({2, 3});
// t1.device_data(); -> pointer to device_data_
template<typename T>
T* Tensor<T>::device_data(){
	return device_data_;
}

// returns device_data_
// Usage:
// Tensor<float> t1({2, 3});
// t1.device_data(); -> pointer to device_data_
template<typename T>
const T* Tensor<T>::device_data() const{
	return device_data_;
}

// copies host_data_ to device_data_
// Usage:
// Tensor<float> t1({2, 3});
// t1.toDevice();
template<typename T>
void Tensor<T>::toDevice(cudaStream_t stream){
	cudaMemcpyAsync(device_data_, host_data_, size_ * sizeof(T), cudaMemcpyHostToDevice, stream);
}

// copies device_data_ to host_data_
// Usage:
// Tensor<float> t1({2, 3});
// t1.toHost();
template<typename T>
void Tensor<T>::toHost(cudaStream_t stream){
	cudaMemcpyAsync(host_data_, device_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
}

// returns shape_
// Usage:
// Tensor<float> t1({2, 3});
// t1.shape(); -> {2, 3}
template<typename T>
const std::vector<size_t>& Tensor<T>::shape() const{
	return shape_;
}

template<typename T>
size_t Tensor<T>::size() const{
	return size_;
}

// returns host_data_[i]
// Usage:
// Tensor<float> t1({2, 3});
// t1[0]; -> host_data_[0]
template<typename T>
T& Tensor<T>::operator[](size_t i){
	size_t index = i;
	if(index >= size_){
		throw std::out_of_range("Tensor index out of range");
	}
	return host_data_[index];
}

// returns host_data_[i]
// Usage:
// Tensor<float> t1({2, 3});
// t1[0]; -> host_data_[0]
template<typename T>
const T& Tensor<T>::operator[](size_t i) const{
	size_t index = i;
	if(index >= size_){
		throw std::out_of_range("Tensor index out of range");
	}
	return host_data_[index];
}

// returns host_data_[i * shape_[1] + j]
// Usage:
// t1(0, 512); -> host_data_[512]
template<typename T>
T& Tensor<T>::operator()(size_t i, size_t j){
	size_t index = i * shape_[1] + j;
	if(index >= size_){
		throw std::out_of_range("Tensor index out of range");
	}
	return host_data_[index];
}

// returns host_data_[i * shape_[1] + j]
// Usage:
// t1(1, 512); -> host_data_[1024]
template<typename T>
const T& Tensor<T>::operator()(size_t i, size_t j) const{
	size_t index = i * shape_[1] + j;
	if(index >= size_){
		throw std::out_of_range("Tensor index out of range");
	}
	return host_data_[index];
}

// prints the tensor
// Usage:
// Tensor<float> t1({2, 3});
// std::cout << t1 << std::endl;
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

// adds two tensors and returns a new tensor with the result
// Usage:
// Tensor<float> t1({2, 3});
// Tensor<float> t2({2, 3});
// t1 + t2; -> {2, 3}
template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& other){
	size_t m, k, n;
	m = shape_[0];
	k = shape_[1];
	n = other.shape_[1];
	
	if(m == other.shape_[0] && k != n){ // broadcasting 
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

// adds a scalar to the tensor
// Usage:
// Tensor<float> t1({2, 3});
// t1 + 1; -> {2, 3}
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

// subtracts two tensors and returns a new tensor with the result
// Usage:
// Tensor<float> t1({2, 3});
// Tensor<float> t2({2, 3});
// t1 - t2; -> {2, 3}
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

// subtracts a scalar from the tensor
// Usage:
// Tensor<float> t1({2, 3});
// t1 - 1; -> {2, 3}
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

// multiplies two tensors and returns a new tensor with the dotproduct result
// Usage:
// Tensor<float> t1({2, 3});
// Tensor<float> t2({3, 2});
// t1 * t2; -> {2, 2}
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

// multiplies a scalar to the tensor
// Usage:
// Tensor<float> t1({2, 3});
// t1 * 2; -> {2, 3}
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
	cudaMemcpy(result.data_, result.device_data_, result.size() * sizeof(T), cudaMemcpyDeviceToHost);
	return result;
} 


// performs a forward pass on the tensor
// Usage:
// Tensor<float> in({2, 3});
// Tensor<float> weights({3, 2});
// Tensor<float> bias({2, 2});
// in.forwardPass(weights, bias); -> {2, 2}
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
	if(bias_m != weights_n) throw std::runtime_error("bad forward bias");
	
	Tensor<T> result({in_m, weights_n});
	if(in.device_data_ && weights.device_data_ && bias.device_data_){
		forwardCall(in, weights, bias, result);
		cudaDeviceSynchronize();
		cudaMemcpy(result.host_data_, result.device_data_, in_m * weights_n * sizeof(T), cudaMemcpyDeviceToHost); 
	}
	return result;
} 


// performs a relu activation on the tensor
// Usage:
// Tensor<float> in({2, 3});
// Tensor<float> relu(in); -> {2, 3}
template<typename T>
Tensor<T> Tensor<T>::relu(const Tensor<T>& in){
	Tensor<T> result(in.shape());
	if(in.device_data_){
		reluCall(in, result);
		cudaDeviceSynchronize();
		cudaMemcpy(result.host_data_, result.device_data_, result.shape()[0] * result.shape()[1] * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return result;
}
// performs a softmax activation on the tensor
// Usage:
// Tensor<float> in({2, 3});
// Tensor<float> softmax(in); -> {2, 3}
template<typename T>
Tensor<T> Tensor<T>::softmax(const Tensor<T>& in){
	Tensor<T> result(in.shape());
	if(in.device_data_){
		softmaxCall(in, result);
		cudaDeviceSynchronize();
		cudaMemcpy(result.host_data_, result.device_data_, result.shape()[0] * result.shape()[1] * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return result;
}

// performs a tanh activation on the tensor
// Usage:
// Tensor<float> in({2, 3});
// Tensor<float> tanh(in); -> {2, 3}
template<typename T>
Tensor<T> Tensor<T>::tanh(const Tensor<T>& in){
	Tensor<T> result(in.shape());
	if(in.device_data_){
		tanhCall(in, result);
		cudaDeviceSynchronize();
		cudaMemcpy(result.host_data_, result.device_data_, result.shape()[0] * result.shape()[1] * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return result;
}

// performs a cross entropy loss on the tensor
// Usage:
// Tensor<float> in({2, 3});
// Tensor<float> labels({2, 3});
// Tensor<float> ceLoss(in, labels); -> {2, 1}
template<typename T>
Tensor<T> Tensor<T>::ceLoss(const Tensor<T>& in, const Tensor<T>& labels){
	Tensor<T> result({labels.shape()[0], 1});
	if(in.device_data_ && labels.device_data_){
		crossEntropyLoss(in, labels, result);
		cudaDeviceSynchronize();
		cudaMemcpy(result.host_data_, result.device_data_, result.shape()[0] * sizeof(T), cudaMemcpyDeviceToHost);
	}
	return result;
}

// performs a backward pass on the tensor
// Usage:
// Tensor<float> in({2, 3});
// Tensor<float> weights({3, 2});
// Tensor<float> bias({2, 2});
// Tensor<float> labels({2, 3});
// in.backward(weights, bias, labels, grad_weights, grad_bias, grad_in); -> {2, 3}
template<typename T>
void Tensor<T>::backward(const Tensor<T>& in, const Tensor<T>& weights, const Tensor<T>& bias, const Tensor<T>& labels, Tensor<T>& grad_weights, Tensor<T>& grad_bias, Tensor<T>& grad_in) {
	Tensor<T> grad_out({in.shape()[0], weights.shape()[1]});
	crossEntropyLossBackward(in, labels, grad_out);
	
	cudaDeviceSynchronize();

	dim3 tpb(16, 16);
   	dim3 bpg((weights.shape()[1] + tpb.x - 1) / tpb.x, (weights.shape()[0] + tpb.y - 1) / tpb.y);
	backwardPass(in, weights, bias, grad_out, grad_in, grad_weights, grad_bias);
	cudaDeviceSynchronize();

	cudaMemcpy(grad_weights.host_data_, grad_weights.device_data_, grad_weights.shape()[0] * grad_weights.shape()[1] * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(grad_bias.host_data_, grad_bias.device_data_, grad_bias.shape()[0] * grad_bias.shape()[1] * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(grad_in.host_data_, grad_in.device_data_, grad_in.shape()[0] * grad_in.shape()[1] * sizeof(T), cudaMemcpyDeviceToHost);
}


template class Tensor<float>;



