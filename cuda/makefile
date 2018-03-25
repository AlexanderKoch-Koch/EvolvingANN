
# ---- Link ---------------------------
eann.so:  objects/EvolvingANN.o objects/Brain.o objects/BrainSetup.o objects/Visualize.o objects/Compute.o
	gcc --shared  -o eann.so  objects/EvolvingANN.o objects/Brain.o objects/BrainSetup.o objects/Visualize.o objects/Compute.o -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -l cudart
 
# ---- Cuda compile ------------------
objects/EvolvingANN.o:  EvolvingANN.cu
	nvcc --compiler-options '-fPIC' -o objects/EvolvingANN.o -c EvolvingANN.cu  -I/usr/include/python3.5
	
objects/Brain.o:  Brain.cu
	nvcc --compiler-options '-fPIC' -o objects/Brain.o -c Brain.cu
	
objects/BrainSetup.o:  BrainSetup.cu
	nvcc --compiler-options '-fPIC' -o objects/BrainSetup.o -c BrainSetup.cu
	
objects/Visualize.o:  Visualize.cu
	nvcc --compiler-options '-fPIC' -o objects/Visualize.o -c Visualize.cu
	
objects/Compute.o:  Compute.cu
	nvcc --compiler-options '-fPIC' -o objects/Compute.o -c Compute.cu