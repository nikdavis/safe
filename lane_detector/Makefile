CC = g++
OUTPUT = main
#INCLUDE = -I/usr/include/flycapture
#LIBS = -L../../lib/C -lflycapture-c${D} -L../../lib -lflycapture${D}
INCLUDE = `pkg-config -- cflags opencv`
LIBS = `pkg-config --libs opencv`

${OUTPUT}: ${OUTPUT}.o
	${CC} -o ${OUTPUT} ${OUTPUT}.o ${LIBS} ${COMMON_LIBS}

${OUTPUT}.o: main.cpp
	${CC} ${CFLAGS} ${INCLUDE} -c -O3 -fomit-frame-pointer -funroll-loops -ftree-vectorize main.cpp

clean:
	rm -f ./*.o ./*.d ./${OUTPUT}
