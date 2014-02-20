CC = g++
OUTPUT = main
CFLAGS = -Wall
INCLUDE = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

OBJS = main.o calib.o

${OUTPUT}: ${OBJS}
	${CC} -o ${OUTPUT} ${OBJS} ${LIBS}

main.o: main.cpp
	${CC} ${INCLUDE} ${CFLAGS} -c main.cpp

calib.o: calib.cpp calib.hpp
	${CC} ${INCLUDE} ${CFLAGS} -c calib.cpp

clean:
	rm -f $(OBJS) $(OUTPUT)
