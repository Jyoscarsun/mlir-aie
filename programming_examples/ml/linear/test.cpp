#include "cxxopts.hpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>

#include <vector>
#include <cmath>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "common.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
#ifndef DTYPE_IN 
#define DTYPE_IN std::bfloat16_t
#endif
#ifndef DTYPE_OUT
#define DTYPE_OUT std::bfloat16_t
#endif
#ifndef DTYPE_ACC
#define DTYPE_ACC float
#endif
using X_DATATYPE = DTYPE_IN;
using W_DATATYPE = DTYPE_IN;
using B_DATATYPE = DTYPE_IN;
using Y_DATATYPE = DTYPE_OUT;
using ACC_DATATYPE = DTYPE_ACC;
#endif

#define XSTR(X) STR(X)
#define STR(X) #X

// ----------------------------------------------------------------------------
// Verify results (specific to our design example)
// ----------------------------------------------------------------------------
namespace test_utils{
    template <typename T>
    bool eq(T a, T b, T tolerance){
        return std::abs(a - b) <= tolerance;
    }
}

template <typename T>
int verify(
    const std::vector<std::vector<std::vector<T>>>& X,
    const std::vector<std::vector<T>>& W,
    const std::vector<T>& B,
    const std::vector<std::vector<T>>& Y,
    int verbosity = 0,
    T tolerance = static_cast<T>(0.00390635)
){
    int errors = 0;
    size_t t = X[0].size(); // Number of time steps t
    size_t d = X[0][0].size(); // Input dimension d
    size_t h = W.size() // Output dimension h

    for(size_t i = 0; i < t; ++i){
        for(size_t j = 0; j < h; ++j){
            T ref = B[j];
            for(size_t k = 0; k < d; ++k){
                ref += X[0][i][k]*W[j][k];
            }
            if(!test_utils::eq(ref, Y[i][j], tolerance)){
                std::cout << "Error at Y[" << i << "][" << j << "]: "
                          << Y[i][j] << "!=" << ref
                          << " from dot (X[0][" << i << "], W[" << j << "]) + B[" << j << "]\n";
                errors++;
            } 
            else if(verbosity > 1){
                std::cout << "Correct output Y[" << i << "][" << j << "]: "
                          << Y[i][j] << " == " << ref << "\n";
            }
        }
    }
    return errors
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, const char *argv[]){

    // ------------------------------------------------------
    // Parse program argumentsposter
    // ------------------------------------------------------
    cxxopts::Options options("Linear Layer Test");
    cxxopts::ParseResult vm;
    test_utils::add_default_options(options);

    test_utils::parse_options(argc, argv, options, vm);
    int verbosity = vm["verbosity"].as<int>();
    int do_verify = vm["verify"].as<bool>();
    int n_iterations = vm["iters"].as<int>();
    int n_warmup_iterations = vm["warmup"].as<int>();
    int trace_size = vm["trace_sz"].as<int>();

    // ------------------------------------------------------
    // Configure this to match your design's buffer size
    // ------------------------------------------------------
    int B = 1;
    int T = 56;
    int D = 41;
    int H = 4;

    size_t INX_SIZE = B*T*D * sizeof(X_DATATYPE);
    size_t INW_SIZE = H*D * sizeof(W_DATATYPE);
    size_t INB_SIZE = H * sizeof(B_DATATYPE);
    size_t OUTY_SIZE = T*H * sizeof(Y_DATATYPE);

    size_t OUT_SIZE = OUTY_SIZE + trace_size;

    srant(time(NULL));
    
    // Load isntruction sequence
    std::vector<uint32_t> instr_v = 
        test_utils::load_instr_binary(vm["instr"].as<std::string>());
    if (verbosity >= 1)
        std::cout << "Sequence instr count: " << instr_v.size() << "\n";

    // ------------------------------------------------------
    // Get device, load the xclbin & kernel and register them
    // ------------------------------------------------------
    // Get a device handle
    unsigned int device_index = 0;
    auto device = xrt::device(device_index);

    // Load the xclbin
    if (verbosity >= 1)
        std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
    auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
    
    // Load the kernel
    if (verbosity >= 1)
        std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
    std::string Node = vm["kernel"].as<std::string>();

    // Get the kernel from the xclbin
    auto xkernels = xclbin.get_kernels();
    auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                                [Node, verbosity](xrt::xclbin::kernel &k) {
                                    auto name = k.get_name();
                                    if (verbosity >= 1){
                                        std::cout << "Name: " << name << std::endl;
                                    }
                                    return name.rfind(Node, 0) == 0;
                                });
    auto kernelName = xkernel.get_name();

    // Register xclbin
    if(verbosity >= 1)
        std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
                  << "\n";
    device.register_xclbin(xclbin);

    // Get a hardware context
    if (verbosity >= 1)
        std::cout << "Getting hardware context.\n";
    xrt::hw_context context(device, xclbin.get_uuid());

    // Get a kernel handle
    if (verbosity >= 1)
        std::cout << "Getting handle to kernel:" << kernelName << "\n";
    auto kernel = xrt::kernel(context, kernelName);

    // ------------------------------------------------------
    // Initialize input/ output buffer sizes and sync them
    // ------------------------------------------------------
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_inx = 
        xrt::bo(device, INX_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_inw = 
        xrt::bo(device, INW_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_inb = 
        xrt::bo(device, INB_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    // Assumes trace will only be added to outY - not sure what this means
    auto bo_outy = 
        xrt::bo(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
    
    if (verbosity >= 1)
        std::cout << "Writing data into buffer objects.\n";

    // Initialize instruction buffer
    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

    // Initialize inX buffer
    INX_DATATYPE *bufInX = bo_inx.map<INX_DATATYPE *>();
    std::vector<std::vector<std::vector<INX_DATATYPE>>> XVec(
        B, 
        std::vector<std::vector<INX_DATATYPE>>(T, 
            std::vector<INX_DATATYPE>(D)));
    for(int i = 0; i < B; ++i){
        for(int j = 0; j < T; ++j){
            for(int k = 0; k < D; ++k){
                XVec[i][j][k] = test_utils::random_bfloat16_t((std::bfloat16_t)1.0,
                                                            (std::bfloat16_t)-0.5);
            }
        }    std::vector<std::vector<INOUT0_DATATYPE>> AVec2D(INOUT0_VOLUME, std::vector<INOUT0_DATATYPE>(INOUT1_VOLUME));

    }
    // Flatten to buffer
    for(int i = 0; i < B; ++i){
        for(int j = 0; j < T; ++j){
            for(int k = 0; k < D; ++k){
                size_t flat_ind = i * B * T + j * T + k;
                bufInX[flat_ind] = XVec[i][j][k];
                // this replaced memcpy(bufInOut0, AVec.data(), (AVec.size() * sizeof(INOUT0_DATATYPE)));
            }
        }
    }


    // Initialize inW buffer
    INW_DATATYPE *bufInW = bo_inw.map<INW_DATATYPE *>();
    std::vector<std::vector<INW_DATATYPE>> WVec(H, std::vector<INW_DATATYPE>(D));
    for(int i = 0; i < H; ++i){
        for(int j = 0; j < D; ++j){
            WVec[i][j] = test:utils::random_bfloat16_t((std::bfloat16_t)1.0,
                                                        (std::bfloat16_t)-0.5);
                    
        }
    }
    // Flatten to buffer
    for(int i = 0; i < H; ++i){
        for(int j = 0; j < D; ++j){
            size_t flat_ind = i * H + j;
            bufInW[flat_ind] = WVec[i][j];
            // this replaced memcpy(bufInOut1, BVec.data(), (BVec.size() * sizeof(INOUT1_DATATYPE)));
        }
    }

    // Initialize inB buffer
    INB_DATATYPE *bufInB = bo_inb.map<INB_DATATYPE *>();
    std::vector<INB_DATATYPE> BVec(H);
    for(int i = 0; i < H; i++)
        BVec[i] = test_utils::random_bfloat16_t((std::bfloat16_t)1.0,
                                                (std::bfloat16_t)-0.5);
    memcpy(bufInB, BVec.data(), (BVec.size() * sizeof(INB_DATATYPE)));

    // Initialize outY buffer
    char *bufOutY = bo_outy.map<char *>();
    memset(bufOutY, 0, OUT_SIZE);

    // Sync buffers to update input buffer values
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inx.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_outy.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ------------------------------------------------------
    // Initialize run configs
    // ------------------------------------------------------
    unsigned num_iter = n_iterations + n_warmup_iterations;
    float npu_time_total = 0;
    float npu_time_min = 9999999;
    float npu_time_max = 0;

    int errors = 0;

    // ------------------------------------------------------
    // Main run loop
    // ------------------------------------------------------
    for (unsigned iter = 0; iter < num_iter; iter++){
        if (verbosity >= 1){
            std::cout << "Running Kernel.\n";
        }

        // Run kernel
        if (verbosity >= 1)
            std::cout << "Running Kernel.\n";
        auto start = std::chrono::high_resolution_clock::now();
        unsigned int opcode = 3;
        auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inx, bo_inw, bo_inb, bo_outy);
        run.wait();
        auto stop = std::chrono::high_resolution_clock::now();
        bo_outy.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        if(iter < n_warmup_iterations){
            /* Warmup iterations do not count towards average runtime. */
            continue;
        }
        
        // Copy output results and verify they are correct
        OUTY_DATATYPE* typedBuf = reinterpret_cast<OUTY_DATATYPE*>(bufOutY);
        // Create a 2D vector to store the result
        std::vector<std::vector<OUTY_DATATYPE>> YVec(T, std::vector<OUTY_DATATYPE>(H));
        // Fill CMatrix with the flattened output
        for (int i = 0; i < T; ++i) {
            for (int j = 0; j < H; ++j) {
                size_t flat_index = i * T + j;  // Row-major layout
                YVec[i][j] = typedBuf[flat_index];
            }
        }

        if(do_verify){
            if(verbosity >= 1){
                std::cout << "Verifying results ..." << std::endl;
            }
            auto vstart = std::chrono::system_clock::now();
            errors = verify(XVec, WVec, BVec, YVec, verbosity);
            auto vstop = std::chrono::system_clock::now();
            float vtime = 
                std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
                    .count();
            if (verbosity >= 1){
                std::cout << "Verify time: " << vtime << "secs." << std::endl;
            }
        }
        else{
            if(verbosity >= 1){
                std::cout << "WARNING: results not verified." << std:endl;
            }
        }

        // Write trace values if trace_size > 0
        if(trace_size > 0){
            test_utils.write_out_trace(((char *)bufOutY) + OUTY_SIZE, trace_size, 
                                                vm["trace_file"].as<std::string>());
        }

        // Accumulate run times
        float npu_time =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
                .count();

        npu_time_total += npu_time;
        npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
        npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
    }

    // ------------------------------------------------------
    // Initialize run configs
    // ------------------------------------------------------

    // TODO - Mac count to guide gflops
    float macs = 0;
    std::cout << std::endl
              << "Avg NPU time: " << npu_time_total / n_iterations << "us."
              << std::endl;
    if (macs > 0)
        std::cout << "Avg NPU gflops: "
                  << macs / (1000 * npu_time_total / n_iterations) << std::endl;
    
    std::cout << std::endl 
              << "Min NPU time: " << npu_time_min << "us." << std::endl;
    if (macs > 0)
        std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min)
                  << std::endl;
    
    std::cout << std::endl
              << "Max NPU time: " << npu_time_max << "us." << std::endl;
    if(macs > 0)
        std::cout << "Min NPU gflops: " << macs/ (1000 * npu_time_max)
                  << std::endl;

    if(!errors){
        std::cout << "\nPass!\n\n";
        return 0;
    }
    else{
        std::cout << "\nError count: " << errors << "\n\n";
        std::cout << "\nFailed.\n\n";
        return 1;
    }
}