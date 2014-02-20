CC = g++
OUTPUT = stream
CFLAGS = -Wall
INCLUDE = -I/usr/include/flycapture `pkg-config --cflags opencv`
LIBS = -L../../lib/C -lflycapture-c -L../../lib -lflycapture `pkg-config --libs opencv`

OBJS = fireflymv_camera.o stream.o

${OUTPUT}: ${OBJS}
	${CC} -o ${OUTPUT} ${OBJS} ${LIBS}


fireflymv_camera.o: fireflymv_camera.cpp fireflymv_camera.hpp
	${CC} ${INCLUDE} ${CFLAGS} -c fireflymv_camera.cpp

stream.o: stream.cpp
	${CC} ${INCLUDE} ${CFLAGS} -c stream.cpp

clean:
	rm -f $(OBJS) $(OUTPUT)
