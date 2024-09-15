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

		
		std::vector<size_t> shape = {4,4};
		Tensor<float> result = Tensor<float>::randn({2,2});

		data = result.data();
		std::cout << result << std::endl << tensor << std::endl << tensor2;		
		std::cout << std::endl;
	}catch(const std::exception& ex){
		std::cerr << "Error: " << ex.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
