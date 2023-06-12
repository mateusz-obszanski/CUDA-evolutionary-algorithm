nvcc .\dev-gpu.cu -o dev-gpu.exe --std=c++20 --include-path include --gpu-architecture=sm_75 --extended-lambda --verbose
