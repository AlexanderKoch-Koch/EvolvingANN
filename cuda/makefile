OBJDIR = objects
SRCDIR = src
HEADERDIR = headers

LDFLAGS = -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -l cudart
NVCCFLAGS = -I/usr/include/python3.6 -Iheaders/ -I./

SOURCES = $(wildcard $(SRCDIR)/*.cu)
OBJECTS = $(patsubst %.cu, %.o, $(addprefix $(OBJDIR)/, $(notdir $(SOURCES))))



eann.so:  $(OBJECTS) $(OBJDIR)/.
	gcc --shared  -o eann.so  $(OBJECTS) $(LDFLAGS)
	
$(OBJDIR)/.:
	mkdir -p $@
	
$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(HEADERDIR)/%.h Hyperparameters.h
	nvcc --compiler-options '-fPIC' -c $< $(NVCCFLAGS) -o $@

$(OBJDIR)/EvolvingANN.o: $(SRCDIR)/EvolvingANN.cu Hyperparameters.h
	nvcc --compiler-options '-fPIC' -c $< $(NVCCFLAGS) -o $@	
	
clean:
	rm -f $(OBJDIR)/*.o eann.so