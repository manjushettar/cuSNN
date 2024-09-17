#include <iostream>

#include "tensor.h"
int main(){
	try{
		Tensor<float> in = Tensor<float>::randn({784,128});
		Tensor<float> weights = Tensor<float>::randn({128, 256});
		Tensor<float> biases = Tensor<float>::randn({784,1});

	//	Tensor<float> res = Tensor<float>::forwardPass(in, weights, biases);
	//	res = Tensor<float>::tanh(res);
		
		// layer 2 
	//	Tensor<float> in2 = res;
		Tensor<float> weights2 = Tensor<float>::randn({256, 64});
		Tensor<float> bias2 = Tensor<float>::randn({784, 1});

		//res = Tensor<float>::forwardPass(in2, weights2, bias2);
		//res = Tensor<float>::tanh(res);

		// layer 3
	//	Tensor<float> in3 = res;
		Tensor<float> weights3 = Tensor<float>::randn({64, 10});
		Tensor<float> bias3 = Tensor<float>::randn({784, 1});
		
		//res = Tensor<float>::forwardPass(in3, weights3, bias3);
		//res = Tensor<float>::softmax(res);
		
		Tensor<float> res({784, 256});
		int EPOCHS = 5;
		for(int i = 0; i < EPOCHS; i++){
			res = Tensor<float>::forwardPass(in, weights, biases);
                	res = Tensor<float>::relu(res);
			
			res = Tensor<float>::forwardPass(res, weights2, bias2);
			res = Tensor<float>::tanh(res);

			res = Tensor<float>::forwardPass(res, weights3, bias3);
			res = Tensor<float>::softmax(res);
		}
				
		std::cout << res << std::endl;

		//std::cout << res.shape()[0] << ", " <<  res.shape()[1] << std::endl;	
	}catch(const std::exception& ex){
		std::cerr << "Error: " << ex.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
