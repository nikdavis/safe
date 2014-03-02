#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

int main( void ) {

    int file;
    char buf[128] = {0};

    if( ( file = open( "/dev/FPGA", O_RDWR ) ) == -1) {
        printf( "Failed to open file\n");
        return -1;
    }

    read( file, buf, 127 );

    printf( "%s", buf );

    close( file );
    return 0;
}
