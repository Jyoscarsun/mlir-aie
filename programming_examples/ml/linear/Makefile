##===- Makefile -----------------------------------------------------------===##
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

all: build/final.xclbin

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = linear
trace_size = 8192

aie_py_src = ${targetname}.py 
use_placed?=0

m=64
n=32

t?=64
d?=64
h?=32

aie_py_src = ${targetname}.py 
use_placed ?= 0

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

VPATH := ${srcdir}/../../../aie_kernels/aie2

# Define compile flags based on device
ifeq ($(devicename),npu)
  COMPILE_FLAGS = ${PEANOWRAP2_FLAGS}
else ifeq ($(devicename),npu2)
  COMPILE_FLAGS = ${PEANOWRAP2P_FLAGS} -Dbf16_bf16_ONLY
else 
  $(error "Device type $(devicename) not supported")
endif

# Core 1: Matrix Multiplication core
build/linear_mm_${t}x${d}x${h}.o: ${srcdir}/../../../aie_kernels/aie2/mm.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${COMPILE_FLAGS} -DDIM_m=${t} -DDIM_k=${d} -DDIM_n=${h} -c $< -o ${@F}

# Core 2: Bias Addition core
build/linear_bias_${t}x${d}x${h}.o: ${srcdir}/kernel.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${COMPILE_FLAGS} -DDIM_m=${t} -DDIM_n=${h} -c $< -o ${@F}

# Standard kernels
build/kernel.o: ${srcdir}/kernel.cc
	mkdir -p ${@D}
ifeq (${devicename}, npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DDIM_m=$m -DDIM_n=$n -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DDIM_m=$m -DDIM_n=$n -c $< -o ${@F}
endif

build/mm.o: ${srcdir}/../../../aie_kernels/aie2/mm.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -Dbf16_bf16_ONLY -c $< -o ${@F}
else 
	echo "Device type not supported"
endif

build/%.o: %.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -Dbf16_bf16_ONLY -c $< -o ${@F}
else 
	echo "Device type not supported"
endif

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< --dev=${devicename} --trace_size=8192 -T 256 -D 256 -H 256 -t ${t} -d ${d} -hval ${h} > $@

build/aie_trace.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< --dev=${devicename} --trace_size=${trace_size} -T 256 -D 256 -H 256 -t ${t} -d ${d} -hval ${h} > $@

build/final.xclbin: build/aie.mlir build/mm.o build/kernel.o build/linear_mm_${t}x${d}x${h}.o build/linear_bias_${t}x${d}x${h}.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin ${<F}

build/final_trace.xclbin: build/aie_trace.mlir build/mm.o build/kernel.o build/linear_mm_${t}x${d}x${h}.o build/linear_bias_${t}x${d}x${h}.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin ${<F}
	
${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@
endif

run: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE

trace: ${targetname}.exe build/final_trace.xclbin
	${powershell} ./$< -x build/final_trace.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../utils/parse_trace.py --input trace.txt --mlir build/aie_trace.mlir --output trace_linear.json
	${srcdir}/../../utils/get_trace_summary.py --input trace_linear.json

clean:
	rm -rf build _build ${targetname}.exe
