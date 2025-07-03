# IRON Implementation of Linear GRU Layer
This README file outlines the implementation of the linear layer in a GRU cell. Inputs to the linear layer are the input matrix (`X`), weight matrix (`W`), and bias array (`b`). The input matrix is technically three dimensional (`B`x`T`x`D`), but the batch value is defaulted to 1. The weight matrix `W` is assumed to have been transposed to its inner dimension matches with `X`. The bias array is a one-dimensional array used for row-wise bias addition after the matrix multiplication between `X` and `W`. 

The README file outlines the necessary files included in order for the hardware design to be ran on AI Engine. 

## linear.py
This file describes the detailed design and dataflows within the AI Engine for linear layer computation. It has several key parts:
1. The main function parsed the input parameters and set defaults, including matrix dimensions and tiling block sizes. It then made a call to the linear function with parsed parameters. Here `-hval` is used as the parameter name since `-h` refers to the help function in python.
2. The linear function starts by checking whether or not the matrix dimensions (`T`, `D`, `H`) are divisible by their tiling dimension (`t`, `d`, `h`). The divisibility of the tiling dimensions themselves by intrinsic dimensions (`r`, `p`, `q`) required by microkernel MAC instructions is then verified. The program assumes that before being lowered to this step, the matrices have been padded to ensure divisibility.
3. Within device body, the following has to be done:
   * Declare tiling types with tiling dimensions
   * Declare any external or kernel functions needed. In this case, the two functions used are `matmul_scalarbf16_bf16` and `row_wise_bias_add_bf16_bf16`.
   * Declare shim, memory, and compute tiles. A significant restriction here is the number of DMA channels available to shim outputs and compute inputs. For both, the aie2 hardware have a capacity of 2 DMA channels. For any more inputs or outputs, multiple shim and compute tiles have to be used in AI Engine.
   * Declare dataflow within the AI Engine tile using `object_fifo`. The most typical setup involves loading data from shim tile, moving to memory tile, and loading into compute tile from memory. In this process, tiling happens. Tiling is unavailable when data is loaded from shim tile, but is available when being offloaded into shim tiles. 
5. Within device core/ compute tiles (part of the device body), the following was done:
   * Allocate one compute tile for matrix multiplication, the other for row-wise bias addition. Use locks in this process to access data inside compute tiles
6. Within runtime_sequence (part of the device body), the following was done:
   * Declare how the tracing is done
   * Load data directly with varioys offsets, sizes, and strides. Tiling the loaded data here is necessary.

## test.cpp
This file verifies the design performs exactly the way that's expected. In the linear layer, this file would compute output to the linear layer and compare it to the values compute with the design in `linear.py`. Necessary inclusions needed in this file are outlined below. This C++ code is a testbench for the design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the script verifies the memcpy results and optionally outputs trace data.
* Verification function: this function computes the correct answers and accepts a tolerance to how far the values from AI Engine computation can deviate.
* Main function: declare the dimensions from the matrices from parsing input parameters, and load them into arrays for verification. The matrices and arrays are filled up with random `bfloat16` values.
* The rest of the `test.cpp` file is standard in calculating time of computation on NPU and displaying errors if there are any above tolerance.  

## Makefile
The file outlines the commands to run the design successfully. Specifically, it should include:
* Commands that were ran in the case of `make`, `make run`, and  `make clean` commands.
* Build various MLIR and `.o` files from `.cc` files. Most of the functions relying on external kernels need to have these files build prior to running successfully.
* Build an executable `.exe` file that runs the `test.cpp` file and run design verification.


## Usage

### C++ Testbench

To compile the design and C++ testbench:
```shell
make
```

To run the design:
```shell
make run
```

To generate a [trace file](../../../programming_guide/section-4/section-4b/README.md):
```shell
env use_placed=1 make trace
```
