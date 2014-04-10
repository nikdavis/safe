CC = g++
OUTPUT = sdla
CFLAGS = -Wall `sdl-config --cflags`
LIBS = `sdl-config --libs`

OBJS  = sdla.o main.o

default : CFLAGS += -O3
default : $(OUTPUT)

debug : CFLAGS += -g -DDEBUG
debug : $(OUTPUT)

all: ${OUTPUT}

$(OUTPUT) : $(OBJS)
	$(CC) -o $(OUTPUT) $(OBJS) $(LIBS)

main.o : main.cpp sdla.hpp
	$(CC) $(CFLAGS) -c main.cpp

sdla.o : sdla.cpp sdla.hpp
	$(CC) $(CFLAGS) -c sdla.cpp

clean:
	rm -f *.o $(OUTPUT)

.PHONY : default debug clean



