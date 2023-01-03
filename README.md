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

### Inference

Inference behaves similarly to [Ang Li's TCBNN](https://github.com/pnnl/tcbnn) and is significantly faster than training, in fact, more time is spent loading the weights once into memory than running inference on a dataset like MNIST with TCBNN!

### Training

Training follows the procedure with Boolean gradients described in [Jupyter notebooks](https://github.com/ThomasPluck/discrete-python), this is so slow, however, that it warranted a proper investigation of how to optimize the training algorithm in CUDA. The training procedure works according to the following steps:

1. Data is partitioned into batches (of a multiple of 8 due to CUDA BMMA constraints).

2. Forward inference takes place on this batch, but unlike TCBNN, one XOR BMMA per warp occurs.

**NOTE: The output of one CUDA XOR BMMA is a int[64] array.**

3. From results, the final results are written into global memory using a thread reduction sum.

4. This is then redistributed across the blocks such that each warp focuses on exactly 32 ints from the global BMMA

5. These integers are compared to int bias terms which leads to the final inference term that creates the binary output of the BNN.

6. These binary outputs are compared to binary ground truth variables that form the basis of false positive/false negative vectors.

7. These are then 