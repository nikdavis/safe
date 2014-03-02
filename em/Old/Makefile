CC = g++
OUTPUT = main
CFLAGS = -Wall
INCLUDE = `pkg-config --cflags opencv`
#IPP directory
IPP_DIR = -L /opt/intel/composer_xe_2013_sp1.0.061/ipp/lib/intel64
IPP_DIR += -L /opt/intel/composer_xe_2013_sp1.0.061/compiler/lib/intel64
LIBS = ${IPP_DIR} `pkg-config --libs opencv`

OBJS = main.o bayesSeg.o timer.o

all: ${OUTPUT}

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
