#pragma once
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

template<typename T>
class Tensor{
public:
	Tensor(const std::vector<size_t>& shape);
	Tensor(const Tensor &other); // Copy constructor
	Tensor(Tensor&& other) noexcept; // Move constructor
	Tensor& operator=(const Tensor& other); // Copy assignment
	Tensor& operator=(Tensor&& other) noexcept; // Move assignment
	
	~Tensor();

	T* data();
	const T* data() const;
	T* device_data(); 
	const T* device_data() const;
	
	// mem transfers
	void toDevice(cudaStream_t stream = 0); 
	void toHost(cudaStream_t stream = 0);
	
	// shape info
	const std::vector<size_t>& shape() const;
	size_t size() const;

	// indexing
	T& operator[](size_t i);
	const T& operator[](size_t) const;
	T& operator()(size_t i, size_t j);
	const T& operator()(size_t i, size_t j) const;
	
	template<typename U>
	friend std::ostream& operator<<(std::ostream& os, const Tensor<U>& out);	
	
	Tensor<T> operator+(const Tensor<T>& other);
	Tensor<T> operator+(const float scalar);
	
	Tensor<T> operator-(const Tensor<T>& other);
	Tensor<T> operator-(const float scalar);
	
	Tensor<T> operator*(const Tensor<T>& other);
	Tensor<T> operator*(const float scalar);
	
	Tensor<T> operator/(const Tensor<T>& other);	
	Tensor<T> operator/(const float scalar);
	
	
	Tensor<T> operator+=(const Tensor<T>& other);
	Tensor<T> operator+=(const float scalar);
	
	Tensor<T> operator-=(const Tensor<T>& other);
	Tensor<T> operator-=(const float scalar);

	Tensor<T> operator*=(const Tensor<T>& other);
	Tensor<T> operator*=(const float scalar);

	Tensor<T> operator/=(const Tensor<T>& other);
	Tensor<T> operator/=(const float scalar);
	
	static Tensor<T> relu(const Tensor<T>& in);	
	static Tensor<T> forwardPass(const Tensor<T>& in, const Tensor<T>& weights, const Tensor<T>& bias);
	static Tensor<T> softmax(const Tensor<T>& in);
	static Tensor<T> tanh(const Tensor<T>& in);
	static Tensor<T> ceLoss(const Tensor<T>& in, const Tensor<T>& labels);
	
	static Tensor<T> ceLossBackward(const Tensor<T>& in, const Tensor<T>& labels);
	static Tensor<T> softmaxBackward(const Tensor<T>& in);
	static Tensor<T> reluBackward(const Tensor<T>& in);
	
	void backward(const Tensor<T>& in, const Tensor<T>& weights, const Tensor<T>& bias, const Tensor<T>& labels, Tensor<T>& grad_weights, Tensor<T>& grad_bias, Tensor<T>& grad_in);
	
	static Tensor<T> randn(const std::vector<size_t>& shape);	

private:
	void allocateMemory();
	void freeMemory();
	size_t calculateSize(const std::vector<size_t>& shape) const;

	T* host_data_;
	T* device_data_;
	std::vector<size_t> shape_;
	size_t size_;
};

#include "tensor.tpp"
