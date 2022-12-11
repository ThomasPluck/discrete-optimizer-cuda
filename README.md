# Discrete Optimizer CUDA Implementation

The following is a CUDA implementation of the [discrete optimizer concept](https://github.com/ThomasPluck/discrete-python) - a calculus-free method to optimize Binarized Neural Networks.

### Requirements

- CUDA >11.0
- Nvidia Post-Turing Series GPU (Ampere for best results)

### Quick Start

Clone the repository:
```
$ git clone https://github.com/ThomasPluck/discrete-cuda-optimizer
```
Navigate to the top-level folder of the repository. Data used to train and test models given in the `examples` folder have to be copied with the Python script, it contains utilities to currently download and parse `MNIST`.
```
$ python download_data.py -d mnist
```
This will download and unzip data for all files. We are now ready to build and run models seen in the example directory.

If you are running VSCode, these can be build with the preconfigured tasks (assuming you are on an Ubuntu system with CUDA 12.0) with `Ctrl+Shift+B` - if not, build instructions can be easily deciphered and reconfigured from the `tasks.json` and `c_cpp_properties.json` files in the `.vscode` folder.

Having downloaded the required data and built the desired example - we are ready to train a Binary Neural Net without calculus.