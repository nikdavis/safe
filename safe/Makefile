CC = g++
OUTPUT = safe
CFLAGS = -Wall
OCVFLAGS = -I/usr/include/flycapture `pkg-config --cflags opencv`
OCVLIBS = -lflycapture-c `pkg-config --libs opencv`

OBJS = lmmin.o errorNIETO.o MSAC.o fireflymv_camera.o fsource.o cvwin.o main.o

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

fsource.o : fsource.cpp defs.hpp fsource.hpp fireflymv_camera.hpp
	$(CC) $(OCVFLAGS) $(CFLAGS) -c fsource.cpp

fireflymv_camera.o : fireflymv_camera.cpp fireflymv_camera.hpp
	$(CC) $(OCVFLAGS) $(CFLAGS) -c fireflymv_camera.cpp

MSAC.o : MSAC.cpp MSAC.hpp errorNIETO.hpp
	$(CC) $(OCVFLAGS) $(CFLAGS) -c MSAC.cpp

errorNIETO.o : errorNIETO.cpp errorNIETO.hpp
	$(CC) $(OCVFLAGS) $(CFLAGS) -c errorNIETO.cpp

lmmin.o : lmmin.c lmmin.h lmstruct.h
	$(CC) $(OCVFLAGS) $(CFLAGS) -c lmmin.c

clean:
	rm -f *.o $(OUTPUT)

.PHONY : all debug clean



