CC = g++
OUTPUT = main
CFLAGS = -Wall -std=c++11 -Wconversion -O3 -fPIC -U_FORTIFY_SOURCE
INCLUDE = `pkg-config --cflags opencv`
SHVER = 2
OS = $(shell uname)
#IPP directory
IPP_DIR = -L /opt/intel/composer_xe_2013_sp1.0.061/ipp/lib/intel64
IPP_DIR += -L /opt/intel/composer_xe_2013_sp1.0.061/compiler/lib/intel64
LIBS = ${IPP_DIR} `pkg-config --libs opencv`

OBJS = main.o homography.o bayesSeg.o timer.o carTracking.o helpFn.o  EKF.o

all: ${OUTPUT}

${OUTPUT}: ${OBJS}
	${CC} -o ${OUTPUT} ${OBJS} ${LIBS}

main.o: main.cpp
	${CC} ${INCLUDE} ${CFLAGS} -c main.cpp

homography.o: homography.cpp homography.hpp
	${CC} ${INCLUDE} ${CFLAGS} -c homography.cpp
	
bayesSeg.o: bayesSeg.cpp bayesSeg.hpp
	${CC} ${INCLUDE} ${CFLAGS} -c bayesSeg.cpp

timer.o: timer.cpp timer.hpp
	${CC} ${INCLUDE} ${CFLAGS} -c timer.cpp
	
carTracking.o: carTracking.cpp carTracking.hpp
	${CC} ${INCLUDE} ${CFLAGS} -c carTracking.cpp
	
helpFn.o: helpFn.cpp helpFn.hpp
	${CC} ${INCLUDE} ${CFLAGS} -c helpFn.cpp
	
EKF.o: EKF.cpp EKF.hpp
	${CC} ${INCLUDE} ${CFLAGS} -c EKF.cpp

clean:
	rm -f *.o $(OUTPUT) libsvm.so.$(SHVER)
