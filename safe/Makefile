CC = g++
OUTPUT = safe
CFLAGS = -Wall
OCVFLAGS = `pkg-config --cflags opencv`
OCVLIBS = `pkg-config --libs opencv`

OBJS = fsource.o cvwin.o main.o

all : CFLAGS += -O3
all : $(OUTPUT)

debug : CFLAGS += -g -DDEBUG
debug : $(OUTPUT)

$(OUTPUT) : $(OBJS)
	$(CC) -o $(OUTPUT) $(OBJS) $(OCVLIBS)

main.o : main.cpp defs.hpp fsource.hpp
	$(CC) $(OCVFLAGS) $(CFLAGS) -c main.cpp

cvwin.o : cvwin.cpp defs.hpp cvwin.hpp
	$(CC) $(OCVFLAGS) $(CFLAGS) -c cvwin.cpp

fsource.o : fsource.cpp defs.hpp fsource.hpp
	$(CC) $(OCVFLAGS) $(CFLAGS) -c fsource.cpp

clean:
	rm -f *.o $(OUTPUT)

.PHONY : all debug clean



