CC = g++
OUTPUT = safe
CFLAGS = -Wall
OCVFLAGS = -I/usr/include/flycapture `pkg-config --cflags opencv`
#IPP directory
OCVLIBS = -L /opt/intel/composer_xe_2013_sp1.0.061/ipp/lib/intel64
OCVLIBS += -L /opt/intel/composer_xe_2013_sp1.0.061/compiler/lib/intel64
OCVLIBS += -lflycapture-c `pkg-config --libs opencv`

OBJS  = lmmin.o errorNIETO.o MSAC.o fireflymv_camera.o fsource.o cvwin.o
OBJS += timer.o homography.o bayesSeg.o main.o

default : CFLAGS += -O3
default : $(OUTPUT)

debug : CFLAGS += -g -DDEBUG
debug : $(OUTPUT)

all: ${OUTPUT}

$(OUTPUT) : $(OBJS)
	$(CC) -o $(OUTPUT) $(OBJS) $(OCVLIBS)

main.o : main.cpp defs.hpp fsource.hpp timer.hpp cvwin.hpp MSAC.hpp bayesSeg.hpp
	$(CC) $(OCVFLAGS) $(CFLAGS) -c main.cpp

homography.o: homography.cpp homography.hpp
	$(CC) $(OCVFLAGS) $(CFLAGS) -c homography.cpp

timer.o : timer.cpp timer.hpp
	$(CC) $(OCVFLAGS) $(CFLAGS) -c timer.cpp

cvwin.o : cvwin.cpp defs.hpp cvwin.hpp
	$(CC) $(OCVFLAGS) $(CFLAGS) -c cvwin.cpp

fsource.o : fsource.cpp defs.hpp fsource.hpp fireflymv_camera.hpp
	$(CC) $(OCVFLAGS) $(CFLAGS) -c fsource.cpp

fireflymv_camera.o : fireflymv_camera.cpp defs.hpp fireflymv_camera.hpp
	$(CC) $(OCVFLAGS) $(CFLAGS) -c fireflymv_camera.cpp

MSAC.o : MSAC.cpp defs.hpp MSAC.hpp errorNIETO.hpp
	$(CC) $(OCVFLAGS) $(CFLAGS) -c MSAC.cpp

errorNIETO.o : errorNIETO.cpp errorNIETO.hpp
	$(CC) $(OCVFLAGS) $(CFLAGS) -c errorNIETO.cpp

lmmin.o : lmmin.c lmmin.h lmstruct.h
	$(CC) $(OCVFLAGS) $(CFLAGS) -c lmmin.c

bayesSeg.o : bayesSeg.cpp bayesSeg.hpp
	$(CC) $(OCVFLAGS) $(CFLAGS) -c bayesSeg.cpp

clean:
	rm -f *.o $(OUTPUT)

.PHONY : default debug clean



