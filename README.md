# Discrete Optimizer CUDA Implementation

The following is a CUDA implementation of the discrete optimizer concept.

### Requirements

- CUDA >11.0
- Nvidia Post-Turing Series GPU (Ampere for best results)

### Quick Start

Clone the repository:
```
git clone https://github.com/ThomasPluck/discrete-cuda-optimizer
```
Navigate to the top-level folder of the repository. Data used to train and test models given in the `examples` folder have to be copied using the bash script:
```
./download_data.sh
```
This will download and unzip data for all files (in future a simple utility will be added to allow you to choose which datasets to download). We are now ready to build and run models seen in the example directory. If you are running VSCode, these can be build with the preconfigured tasks (assuming you are on an Ubuntu system) with `Ctrl+Shift+B` - if not, build instructions can be easily reconstructed from the `tasks.json` and `c_cpp_properties.json` files in the `.vscode` folder.