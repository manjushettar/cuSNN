# cuSNN

Tensor<float> in({m,k})\
Tensor<float> weights({k,n})\
Tensor<float> bias({m, 1})\

Tensor<float> res = Tensor<float>::forwardPass(in, weights, bias);

-> git clone \
-> cd cuSNN \
-> mkdir build \
-> cd build \
-> cmake .. \
-> make \
-> ./snn\_cuda 
