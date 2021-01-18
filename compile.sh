#!/bin/bash
nvcc -std=c++11 -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI src/NonInertial.cu -Iinclude/ -o CUDAABP
