CC = g++
OUTPUT = main
CFLAGS = -Wall
INCLUDE = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

OBJS = main.o bayesSeg.o timer.o

${OUTPUT}: ${OBJS}
	${CC} -o ${OUTPUT} ${OBJS} ${LIBS}

main.o: main.cpp
	${CC} ${INCLUDE} ${CFLAGS} -c main.cpp

timer.o : timer.cpp timer.hpp
	$(CC) $(INCLUDE) $(CFLAGS) -c timer.cpp

bayesSeg.o: bayesSeg.cpp bayesSeg.hpp
	${CC} ${INCLUDE} ${CFLAGS} -c bayesSeg.cpp

clean:
	rm -f *.o $(OUTPUT)
