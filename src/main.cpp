#include <iostream>

#include "tensor.h"
int main(){
	try{
		Tensor<float> tensor({3,3});
		Tensor<float> tensor2({3,3});

		float* data = tensor.data();
		float* data2 = tensor2.data();
		for(size_t i = 0; i < tensor.size(); i++){
			data[i] = static_cast<float>(i);
			data2[i] = static_cast<float>(1);
		}
		tensor.toDevice();
		tensor2.toDevice();

		Tensor<float> result = tensor * tensor2;
		tensor = tensor2 / 3.0;
			
		data = result.data();
		std::cout << (result*result + result) << std::endl << tensor << std::endl << tensor2;		
		std::cout << std::endl;
	}catch(const std::exception& ex){
		std::cerr << "Error: " << ex.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
