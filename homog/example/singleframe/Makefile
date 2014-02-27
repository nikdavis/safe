CC = g++
OUTPUT = transform
CFLAGS = -Wall
INCLUDE = -I/usr/include/flycapture `pkg-config --cflags opencv`
LIBS = -L../../lib/C -lflycapture-c -L../../lib -lflycapture `pkg-config --libs opencv`

OBJS = fireflymv_camera.o ${OUTPUT}.o

${OUTPUT}: ${OBJS}
	${CC} -o ${OUTPUT} ${OBJS} ${LIBS}


fireflymv_camera.o: fireflymv_camera.cpp fireflymv_camera.hpp
	${CC} ${INCLUDE} ${CFLAGS} -c fireflymv_camera.cpp

${OUTPUT}.o: ${OUTPUT}.cpp
	${CC} ${INCLUDE} ${CFLAGS} -c ${OUTPUT}.cpp

clean:
	rm -f $(OBJS) $(OUTPUT)
