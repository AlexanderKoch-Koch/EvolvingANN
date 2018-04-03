
OBJDIR = objects
SRCDIR = src
HEADERDIR = headers

LDFLAGS = -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -l cudart
NVCCFLAGS = -I/usr/include/python3.5 -Iheaders/ -I./

SOURCES = $(wildcard $(SRCDIR)/*.cu)
OBJECTS = $(patsubst %.cu, %.o, $(addprefix $(OBJDIR)/, $(notdir $(SOURCES))))




eann.so:  $(OBJECTS)
	gcc --shared  -o eann.so  $(OBJECTS) $(LDFLAGS)
	
	
$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(HEADERDIR)/%.h Hyperparameters.h
	nvcc --compiler-options '-fPIC' -c $< $(NVCCFLAGS) -o $@
	
	
clean:
	rm -f $(OBJDIR)/*.o eann.so