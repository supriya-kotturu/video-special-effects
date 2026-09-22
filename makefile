CXX = g++

# OpenCV flags come from pkg-config (apt: libopencv-dev)
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4)

CXXFLAGS = -std=c++17 -Wall -Iinclude $(OPENCV_CFLAGS)
LDLIBS   = $(OPENCV_LIBS)

SRCDIR = src
OBJDIR = obj
BINDIR = out

.PHONY: all run clean

# each program has its own main(); filters.o is shared
all: $(BINDIR)/main $(BINDIR)/vid $(BINDIR)/img

$(BINDIR)/main: $(OBJDIR)/main.o | $(BINDIR)
	$(CXX) $^ -o $@ $(LDLIBS)

$(BINDIR)/vid: $(OBJDIR)/vidDisplay.o $(OBJDIR)/filters.o | $(BINDIR)
	$(CXX) $^ -o $@ $(LDLIBS)

$(BINDIR)/img: $(OBJDIR)/imgDisplay.o $(OBJDIR)/filters.o | $(BINDIR)
	$(CXX) $^ -o $@ $(LDLIBS)

# rebuild objects when a header changes
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(wildcard include/*.h) | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR) $(BINDIR):
	mkdir -p $@

# usage: make run            (runs out/main)
#        make run P=vid      (runs out/vid)
P ?= main
run: $(BINDIR)/$(P)
	./$(BINDIR)/$(P) $(ARGS)

clean:
	rm -rf $(OBJDIR) $(BINDIR)/main $(BINDIR)/vid $(BINDIR)/img
