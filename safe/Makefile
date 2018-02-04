CC = g++
OUTPUT = safe
CFLAGS += -Wall
CFLAGS += `sdl-config --cflags`
CFLAGS += `pkg-config --cflags opencv`
LIBS += `pkg-config --libs opencv`
LIBS += `sdl-config --libs`

OBJS += lmmin.o errorNIETO.o MSAC.o fsource.o cvwin.o
OBJS += timer.o homography.o kalman.o bayesSeg.o sdla.o carTracking.o EKF.o main.o

default : CFLAGS += -O3
default : $(OUTPUT)

board : CFLAGS += -O3 -march=native
board : $(OUTPUT)

debug : CFLAGS += -g -DDEBUG
debug : $(OUTPUT)

all: ${OUTPUT}

$(OUTPUT) : $(OBJS)
	$(CC) -o $(OUTPUT) $(OBJS) $(LIBS)

main.o : main.cpp defs.hpp fsource.hpp cvwin.hpp timer.hpp MSAC.hpp kalman.hpp homography.hpp bayesSeg.hpp carTracking.hpp sdla.hpp
	$(CC) $(CFLAGS) -c main.cpp

sdla.o : sdla.cpp defs.hpp sdla.hpp
	$(CC) $(CFLAGS) -c sdla.cpp

homography.o: homography.cpp homography.hpp
	$(CC) $(CFLAGS) -c homography.cpp

timer.o : timer.cpp timer.hpp
	$(CC) $(CFLAGS) -c timer.cpp

kalman.o : kalman.cpp kalman.hpp
	$(CC) $(CFLAGS) -c kalman.cpp

cvwin.o : cvwin.cpp defs.hpp cvwin.hpp
	$(CC) $(CFLAGS) -c cvwin.cpp

fsource.o : fsource.cpp defs.hpp fsource.hpp
	$(CC) $(CFLAGS) -c fsource.cpp

MSAC.o : MSAC.cpp defs.hpp MSAC.hpp errorNIETO.hpp lmmin.h
	$(CC) $(CFLAGS) -c MSAC.cpp

errorNIETO.o : errorNIETO.cpp errorNIETO.hpp
	$(CC) $(CFLAGS) -c errorNIETO.cpp

lmmin.o : lmmin.c lmmin.h lmstruct.h
	$(CC) $(CFLAGS) -c lmmin.c

bayesSeg.o : bayesSeg.cpp bayesSeg.hpp
	$(CC) $(CFLAGS) -c bayesSeg.cpp

carTracking.o : carTracking.cpp EKF.hpp carTracking.hpp
	$(CC) $(CFLAGS) -c carTracking.cpp

EKF.o : EKF.cpp EKF.hpp
	$(CC) $(CFLAGS) -c EKF.cpp

clean:
	rm -f *.o $(OUTPUT)

.PHONY : default debug clean
