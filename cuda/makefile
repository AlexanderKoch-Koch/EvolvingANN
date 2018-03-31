
OBJDIR = objects
SRCDIR = src

LDFLAGS = -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -l cudart
NVCCFLAGS = -I/usr/include/python3.5 -Iheaders/ -I./

SOURCES = $(wildcard $(SRCDIR)/*.cu)
OBJECTS = $(patsubst %.cu, %.o, $(addprefix $(OBJDIR)/, $(notdir $(SOURCES))))

HEADERS = $(wildcard headers/*.h)


eann.so:  $(OBJECTS)
	gcc --shared  -o eann.so  $(OBJECTS) $(LDFLAGS)
	
	
$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	nvcc --compiler-options '-fPIC' -c $< $(NVCCFLAGS) -o $@
	
	
clean:
	rm -f $(OBJDIR)/*.o eann.so