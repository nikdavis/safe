#
# http://www.gnu.org/software/make/manual/make.html
#
# -ggdb: http://gcc.gnu.org/onlinedocs/gcc-4.3.3/gcc/Debugging-Options.html#Debugging-Options
CC = g++
OUTPUT = main
CFLAGS = -Wno-deprecated -Wall -ggdb -std=c++11 -Wno-deprecated-declarations
CFLAGS += -g -O2 -D_REENTRANT -I/usr/local/include/SDL2   -DHAVE_OPENGL -g
INCLUDES =  `pkg-config --cflags libavformat libavcodec libswscale libavutil`
INCLUDES += `sdl-config --cflags`
LIBS = `pkg-config --libs libavformat libavcodec libswscale libavutil` 
LIBS += -lm
LIBS += `sdl-config --libs`
LIBS += -Wl,-rpath,/usr/local/lib -L/usr/local/lib -lSDL2 -lpthread  
OBJS = main.o timer.o stereoAudio.o



all: ${OUTPUT}

${OUTPUT}: ${OBJS}
	${CC} -o ${OUTPUT} ${OBJS} ${LIBS}

main.o: 
	${CC} ${INCLUDE} ${CFLAGS} -c main.cpp
	
timer.o : timer.cpp timer.hpp
	$(CC) $(INCLUDE) $(CFLAGS) -c timer.cpp
	
stereoAudio.o : stereoAudio.cpp stereoAudio.hpp
	$(CC) $(INCLUDE) $(CFLAGS) -c stereoAudio.cpp

clean:
	rm -f *.o $(OUTPUT)

