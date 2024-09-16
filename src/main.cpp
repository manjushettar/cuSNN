#include <iostream>

#include "tensor.h"
int main(){
	try{
		Tensor<float> tensor({3,3});
		Tensor<float> tensor2({3,3});

		for(size_t i = 0; i < tensor.size(); i++){
			tensor[i] = static_cast<float>(i);
			tensor2[i] = static_cast<float>(-1.0 * i);
		}
			
		Tensor<float> in = Tensor<float>::randn({100,20});
		Tensor<float> weights = Tensor<float>::randn({20, 70});
		Tensor<float> biases = Tensor<float>::randn({100,1});

		Tensor<float> res = Tensor<float>::forwardPass(in, weights, biases);
		res = Tensor<float>::relu(res);
		std::cout << res << res.shape()[0] << res.shape()[1] << std::endl;	
	}catch(const std::exception& ex){
		std::cerr << "Error: " << ex.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
