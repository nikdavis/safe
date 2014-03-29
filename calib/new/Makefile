CC = g++
OUTPUT = calib
CFLAGS = -Wall
INCLUDE = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

OBJS = calib.o

${OUTPUT}: ${OBJS}
	${CC} -o ${OUTPUT} ${OBJS} ${LIBS}

calib.o: calib.cpp
	${CC} ${INCLUDE} ${CFLAGS} -c calib.cpp

clean:
	rm -f $(OBJS) $(OUTPUT)
