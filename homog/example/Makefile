CC = g++
OUTPUT = transform
CFLAGS = -Wall
INCLUDE = -I/usr/include/flycapture `pkg-config --cflags opencv`
#IPP directory
IPP_DIR = -L /opt/intel/composer_xe_2013_sp1.0.061/ipp/lib/intel64
IPP_DIR += -L /opt/intel/composer_xe_2013_sp1.0.061/compiler/lib/intel64
LIBS = ${IPP_DIR} `pkg-config --libs opencv`
LIBS += -lflycapture-c -lflycapture

OBJS = fireflymv_camera.o homography.o ${OUTPUT}.o

all: ${OUTPUT}

${OUTPUT}: ${OBJS}
	${CC} -o ${OUTPUT} ${OBJS} ${LIBS}

fireflymv_camera.o: fireflymv_camera.cpp fireflymv_camera.hpp
	${CC} ${INCLUDE} ${CFLAGS} -c fireflymv_camera.cpp

homography.o: homography.cpp homography.hpp
	${CC} ${INCLUDE} ${CFLAGS} -c homography.cpp

${OUTPUT}.o: ${OUTPUT}.cpp
	${CC} ${INCLUDE} ${CFLAGS} -c ${OUTPUT}.cpp

clean:
	rm -f $(OBJS) $(OUTPUT)
