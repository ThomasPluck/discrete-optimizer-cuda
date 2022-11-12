#include"standard_includes.h"

#include"layer/layer.h"

int main(){

    std::cout << "initializing teensyflow..." << std::endl;

    std::cout << "getting device properties..." << std::endl;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&Launch::deviceProp, 0));

    std::cout << "making dummy layers..." << std::endl;
    FcLayer layer1 = FcLayer(128,128,128);
    FcLayer layer2 = FcLayer(128,128,128);

    std::cout << "filling data with random values..." << std::endl;
    layer1.input.fill_random();
    layer1.weights.fill_random();
    layer1.biases.fill_random();
    layer1.output.fill_random();
    layer1.weight_counters.fill_random();
    layer1.bias_counters.fill_random();

    std::cout << "linking together the layers..." << std::endl;
    link(&layer1, &layer2); // this is legal, it will be on us in codegen to make sure this doesn't overflow

    std::cout << "going forward on layer one..." << std::endl;
    layer1.forward();

    std::cout << "going forward on layer two..." << std::endl;
    layer2.forward();

    std::cout << "going back on layer two..." << std::endl;
    layer2.back();

    std::cout << "going back on layer one..." << std::endl;
    layer1.back();

    std::cout << "finished." << std::endl;
    return 0;
}