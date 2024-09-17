#include <iostream>
#include <random>
#include "tensor.h"

void rand_labels(Tensor<float>& labels){
	const size_t m = labels.shape()[0];
	const size_t n = labels.shape()[1];
	
	std::random_device rd;
    	std::mt19937 gen(rd());
   	std::uniform_int_distribution<> dis(0, n - 1);

	for (size_t i = 0; i < m; ++i) {
     	   int random_class = dis(gen);  // randomly choose one class for each sample
        	for (size_t j = 0; j < n; ++j) {
       		     	labels(i, j) = (j == random_class) ? 1.0f : 0.0f;  // set one-hot encoding
        	}
    	}
}
int main(){
	try{
		Tensor<float> in = Tensor<float>::randn({16,784});
		Tensor<float> weights = Tensor<float>::randn({784, 128});
		Tensor<float> biases = Tensor<float>::randn({128,1});

	//	Tensor<float> res = Tensor<float>::forwardPass(in, weights, biases);
	//	res = Tensor<float>::tanh(res);
		
		// layer 2 
	//	Tensor<float> in2 = res;
		Tensor<float> weights2 = Tensor<float>::randn({128, 256});
		Tensor<float> bias2 = Tensor<float>::randn({256, 1});

		//res = Tensor<float>::forwardPass(in2, weights2, bias2);
		//res = Tensor<float>::tanh(res);

		// layer 3
	//	Tensor<float> in3 = res;
		Tensor<float> weights3 = Tensor<float>::randn({256, 10});
		Tensor<float> bias3 = Tensor<float>::randn({10, 1});
		
		//res = Tensor<float>::forwardPass(in3, weights3, bias3);
		//res = Tensor<float>::softmax(res);
		
		Tensor<float> res({16, 128});

		Tensor<float> labels({16, 10});
		rand_labels(labels);
		labels.toDevice();	
		Tensor<float> loss({16, 1});

		int EPOCHS = 1;
		for(int i = 0; i < EPOCHS; i++){
			res = Tensor<float>::forwardPass(in, weights, biases);
                	res = Tensor<float>::relu(res);
			
			res = Tensor<float>::forwardPass(res, weights2, bias2);
			res = Tensor<float>::tanh(res);

			res = Tensor<float>::forwardPass(res, weights3, bias3);
			res = Tensor<float>::softmax(res);

			loss = Tensor<float>::ceLoss(res, labels);
		}
				
		std::cout << loss << std::endl;

		std::cout << loss.shape()[0] << ", " <<  loss.shape()[1] << std::endl;	
	}catch(const std::exception& ex){
		std::cerr << "Error: " << ex.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
