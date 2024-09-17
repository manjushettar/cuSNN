
#pragma once

template<typename T>
__global__ void addT(const T* a, const T* b, T* result, size_t size);

template<typename T>
__global__ void subT(const T* a, const T* b, T* result, size_t size);

template<typename T>
__global__ void mulT(const T* a, const T* b, T* result, size_t m, size_t n, size_t k);

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
__global__ void addTV(const T* a, const T* b, T* result, size_t m, size_t n, size_t k);

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

template<typename T>
void addTensorAndVector(const Tensor<T>& tensor1, const Tensor<T>& vector1, Tensor<T>& tensor3);

template<typename T>
__global__ void forwardP(const T* in, const T* weights, const T* bias, T* out, const size_t m, const size_t k, const size_t n);

template<typename T>
void forwardCall(const Tensor<T>& in, const Tensor<T>& weights, const Tensor<T>& bias, Tensor<T>& out);

template<typename T>
__global__ void relu(const T* in, T* out, const size_t m, const size_t n);

template<typename T>
void reluCall(const Tensor<T>& in, Tensor<T>& out);

template<typename T>
__global__ void tanh(const T* in, T* out, const size_t m, const size_t n);

template<typename T>
void tanhCall(const Tensor<T>& in, Tensor<T>& out);

template<typename T>
__global__ void softmax(const T* in, T* out , const size_t m, const size_t n);

template<typename T>
void softmaxCall(const Tensor<T>& in, Tensor<T>& out);

template<typename T>
__global__ void ce_loss(const T* in, const T* labels, T* out, const size_t m, const size_t n);

template<typename T>
void crossEntropyLoss(const Tensor<T>& in, const Tensor<T> &labels, Tensor<T>& out);

template<typename T>
__global__ void ce_lossBackward(const T* in, const T* labels, T* out, const size_t m, const size_t n);

template<typename T>
void crossEntropyLossBackward(const Tensor<T>& in, const Tensor<T>& labels, Tensor<T>& out);

template<typename T>
__global__ void relu_backward(const T* in, T* out, const size_t m, const size_t n);

template<typename T>
void reluBackward(const Tensor<T>& in, Tensor<T>& out);

template<typename T>
__global__ void softmax_backward(const T* in, T* out, const size_t m, const size_t n);

template<typename T>
void softmaxBackward(const Tensor<T>& in, Tensor<T>& out);

template<typename T>
__global__ void backward(const T* in, const T* weights, const T* bias, T* grad_in, T* grad_weights, T* grad_bias, const size_t m, const size_t n);

template<typename T>
void backwardPass(const Tensor<float>& in, const Tensor<float>& weights, const Tensor<float>& bias,
                  Tensor<float>& grad_in, Tensor<float>& grad_weights, Tensor<float>& grad_bias); 
