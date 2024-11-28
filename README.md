## Notice
All my code is based on this project:
Comprehensive Study on GPU Program Numerical Issues
https://github.com/GPU-Program-Bug-Study/Comprehensive-Study-on-GPU-Program-Numerical-Issues.github.io/tree/main

This project has a complete pipeline with generation, compilation, and execution. However, I only need the CUDA program generation at the moment, so I have cut it down.

Though I am a beginner, I may add some new features in the future.

## input file
functions_to_test.txt
```commandline
# TYPE1:
FUNCTION:__hdiv (__half, __half)
FUNCTION:__hfma (__half, __half, __half)
FUNCTION:__hfma_relu (__half, __half, __half)
FUNCTION:__hfma_sat (__half, __half, __half)
FUNCTION:__hmul (__half, __half)
FUNCTION:__hmul_rn (__half, __half)
FUNCTION:__hmul_sat (__half, __half)
FUNCTION:__hneg (__half)

# TYPE2:
OPERATOR:*(__half, __half)
OPERATOR:+(__half, __half)
OPERATOR:-(__half, __half)
```

## Run
python gen_prog.py functions_to_test.txt


## TODO
1. make our template code more simpler.
2. add compilation pipeline
3. ?
