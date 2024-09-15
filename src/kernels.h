
#pragma once

template<typename T>
__global__ void addT(const T* a, const T* b, T* result, size_t size);

template<typename T>
__global__ void subT(const T* a, const T* b, T* result, size_t size);

template<typename T>
__global__ void mulT(const T* a, const T* b, T* result, size_t size);

template<typename T>
__global__ void divT(const T* a, const T* b, T* result, size_t size);

template<typename T>
__global__ void addS(const T* a, T* result, const float scalar, size_t size);

template<typename T>
__global__ void subS(const T* a, T* result, const float scalar, size_t size);

template<typename T>
__global__ void mulS(const T* a, T* result, const float scalar, size_t size);

template<typename T>
__global__ void divS(const T* a, T* result, const float scalar, size_t size);


template<typename T>
__global__ void randN(T* a, const size_t size, unsigned long long seed); 

template<typename T>
void addTwoTensors(const Tensor<T>& tensor1, const Tensor<T>& tensor2, Tensor<T>& tensor3);

template<typename T>
void subTwoTensors(const Tensor<T>& tensor1, const Tensor<T>& tensor2, Tensor<T>& tensor3);

template<typename T>
void mulTwoTensors(const Tensor<T>& tensor1, const Tensor<T>& tensor2, Tensor<T>& tensor3);

template<typename T>
void divTwoTensors(const Tensor<T>& tensor1, const Tensor<T>& tensor2, Tensor<T>& tensor3);

template<typename T>
void addScalar(const Tensor<T>& tensor1, Tensor<T>& tensor3, const float scalar);

template<typename T>
void subScalar(const Tensor<T>& tensor1, Tensor<T>& tensor3, const float scalar); 

template<typename T>
void mulScalar(const Tensor<T>& tensor1, Tensor<T>& tensor3, const float scalar); 

template<typename T>
void divScalar(const Tensor<T>& tensor1, Tensor<T>& tensor3, const float scalar); 

template<typename T>
void fillRandom(Tensor<T>& tensor1, const size_t size);
