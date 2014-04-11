#include "sdla.hpp"
#include <iostream>

int main( int argc, char* argv[] ) {
    sdla audio( "boop.wav" );

    audio.set_interval( 0, 100 );

    /*for ( int i = 100; i >= 0; i -= 2 ) {
        audio.set_volume( i * 0.01, i * 0.01 );
        sleep( 1 );
    }*/

    /*for ( int i = 0; i <=180; i += 3 ) {
        audio.set_position( 1, i );
        sleep( 1 );
    }*/

    for ( float i = 4; i >= 1; i -= 0.05 ) {
        audio.set_position_deg( i, 90 );
        sleep( 1 );
    }

    return 0;
}



