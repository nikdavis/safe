#include "sdla.hpp"
#include <iostream>

int main( int argc, char* argv[] ) {
    sdla audio( "boop.wav" );

    audio.play_WAV();
    while ( true ) {
        audio.play_WAV();
        sleep(1);
    }

    return 0;
}



