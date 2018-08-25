OBJDIR = objects
SRCDIR = src
HEADERDIR = headers

LDFLAGS = -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -l cudart -l tensorflow_framework
NVCCFLAGS = -I/usr/include/python3.6 -Iheaders/ -I./

SOURCES = $(wildcard $(SRCDIR)/*.cu)
OBJECTS = $(patsubst %.cu, %.o, $(addprefix $(OBJDIR)/, $(notdir $(SOURCES))))



eann.so:  $(OBJDIR)/. $(OBJECTS)
	g++-6 --shared -o eann.so  $(OBJECTS) $(OBJDIR)/TensorboardInterface.o $(LDFLAGS) 
	
$(OBJDIR)/.:
	mkdir -p $@
	
$(OBJDIR)/TensorboardInterface.o: TensorboardInterface/TensorboardInterface.cpp headers/TensorboardInterface.h
	g++ -c -o $@ -fPIC -I/usr/local/include/tensorflow -I/usr/local/include/tensorflow/bazel-genfiles -I$(HEADERDIR) TensorboardInterface/TensorboardInterface.cpp -ltensorflow_framework
	
$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(HEADERDIR)/%.h Hyperparameters.h $(OBJDIR)/TensorboardInterface.o
	nvcc --compiler-options '-fPIC' -c $< $(NVCCFLAGS) -o $@ 

$(OBJDIR)/EvolvingANN.o: $(SRCDIR)/EvolvingANN.cu Hyperparameters.h
	nvcc --compiler-options '-fPIC' -c $< $(NVCCFLAGS) -o $@	
	
clean:
	rm -f $(OBJDIR)/*.o eann.so