// FILE: sdla.cpp

#include "sdla.hpp"
#include "SDL.h"
#include "defs.hpp"
#include <iostream>
#include <csignal>
#include <sys/time.h>
#include <cmath>

#define PI 3.14159265

#define LBYTE( word )   ( ( word ) & 0x00FF )
#define HBYTE( word ) ( ( ( word ) & 0xFF00 ) >> 8 )
#define MWORD( lo, hi ) ( ( ( int )( hi ) << 8 ) | ( lo ) )

static sdla *instance = 0;
static struct sigaction act;
static struct itimerval itimer;

// Callback used by SDL audio thread to fill audio buffer
void audio_callback( void *userdata, unsigned char *stream, int len ) {
    sdla *psdla = (sdla*)userdata;

    if ( !psdla->_isvalid ) return;

    // Only copy if we have samples left
    if ( psdla->cur_len == 0 ) return;

    // Copy as much data as possible
    len = ( (unsigned)len > psdla->cur_len ? psdla->cur_len : len );
    memcpy( stream, psdla->cur_pos, len );

    psdla->cur_pos += len;
    psdla->cur_len -= len;
}

void timer_callback( int signal ) {
    if( instance ) instance->play_WAV();
}

void sdla::set_interval( int sec, int msec ) {
    if ( !_isvalid ) return;

    int usec = msec * 1000;
    itimer.it_interval.tv_sec = sec;
    itimer.it_interval.tv_usec = usec;
    itimer.it_value.tv_sec = sec;
    itimer.it_value.tv_usec = usec;

    // Setup interval timer which generates SIGALRM on expiration
    setitimer( ITIMER_REAL, &itimer, 0 );
}

// This mixer does not perform clipping. Audio can glitch if volumes are over 1
void sdla::set_volume( float v_left, float v_right ) {
    if ( !_isvalid ) return;
    DMESG( "slda: Setting volume to [L,R]:\t\t" << v_left << ",\t" << v_right );

    // Because we are touching data structures accessed in the callback, lock it
    SDL_LockAudio();

    unsigned char *oiter = WAV_buf;
    unsigned char *miter = mixed_WAV_buf;

    // Channel data is interleaved in left/right ordering for stereo
    // Assuming S16LSB and LSB Host - so four bytes per sample, LLRR
    // Divide len by sample size by right shift 2
    for ( unsigned int i = 0; i < ( WAV_len >> 2 ); ++i ) {
        int L = MWORD( *oiter, *( oiter + 1 ) );
        if ( L & 0x8000 ) L |= 0xFFFF0000; // Sign extend
        oiter += 2;
        int R = MWORD( *oiter, *( oiter + 1 ) );
        if ( R & 0x8000 ) R |= 0xFFFF0000; // Sign extend
        oiter += 2;

        // Perform mixing
        R *= v_left;
        L *= v_right;

        *miter = LBYTE( L );
        *( miter + 1 ) = HBYTE( L );
        miter += 2;
        *miter = LBYTE( R );
        *( miter + 1 ) = HBYTE( R ) ;
        miter += 2;
    }
    SDL_UnlockAudio();
}

// Radius should be > 1, angle should be between 0 and PI
void sdla::set_position( float radius, float angle ) {
    if ( !_isvalid ) return;
    DMESG( "slda: Setting position to [r, a]:\t" << radius << "'\t" << angle );

    double scale = 1.0 / ( radius * radius );
    double RS = cos( angle ) + 1.0;
    double LS = 2 - RS;
    LS *= scale;
    RS *= scale;
    set_volume( LS, RS );
}

void sdla::set_position_deg( float radius, int angle ) {
    if ( !_isvalid ) return;
    set_position( radius, angle * ( PI / 180.0 ) );
}

void sdla::play_WAV( void ) {
    if ( !_isvalid ) return;

    // Because we are touching data structures accessed in the callback, lock it
    SDL_LockAudio();

    // Reset buffer index to start of loaded WAV file
    cur_pos = mixed_WAV_buf;
    cur_len = WAV_len;

    SDL_UnlockAudio();
}

sdla::sdla( const char *filename ) : _isvalid( false ), cur_len( 0 ) {
    if ( instance ) {
        std::cerr << "slda: Cannot create multiple instances." << std::endl;
        return;
    }
    DMESG( "slda: Creating instance..." );

    SDL_AudioSpec requested_spec, obtained_spec;

    // Open WAV file, load data into buffer, and read spec from header
    if ( SDL_LoadWAV( filename, &requested_spec,
                &WAV_buf, &WAV_len ) == 0 ) {
        std::cerr << "sdla: Couldn't load " << filename << ": "
                    << SDL_GetError() << std::endl;
        return;
    }

    mixed_WAV_buf = new unsigned char[WAV_len];
    if ( !mixed_WAV_buf ) {
        std::cerr << "sdla: Failed buffer allocation." << std::endl;
        SDL_FreeWAV( WAV_buf );
        delete mixed_WAV_buf;
        return;
    }

    memcpy( mixed_WAV_buf, WAV_buf, WAV_len );

    DMESG( "slda: WAV: Freq: " << requested_spec.freq <<
        " Channels: " << (int)requested_spec.channels <<
        " Samples: " << requested_spec.samples );

    switch ( requested_spec.format ) {
        case AUDIO_S8:      DMESG( "sdla: WAV: Format: AUDIO_S8" );      break;
        case AUDIO_U8:      DMESG( "sdla: WAV: Format: AUDIO_U8" );      break;
        case AUDIO_S16LSB:  DMESG( "sdla: WAV: Format: AUDIO_S16LSB" );  break;
        case AUDIO_S16MSB:  DMESG( "sdla: WAV: Format: AUDIO_S16MSB" );  break;
        case AUDIO_U16LSB:  DMESG( "sdla: WAV: Format: AUDIO_U16LSB" );  break;
        case AUDIO_U16MSB:  DMESG( "sdla: WAV: Format: AUDIO_U16MSB" );  break;
        default:            DMESG( "sdla: WAV: Format: Unknown" );       break;
    }

    requested_spec.callback = audio_callback;
    requested_spec.userdata = this;

    if ( SDL_OpenAudio( &requested_spec, &obtained_spec ) < 0 ) {
        std::cerr << "sdla: Failed audio init: " << SDL_GetError() << std::endl;
        SDL_FreeWAV( WAV_buf );
        delete mixed_WAV_buf;
        return;
    }

    // Start audio (buffer wont fill until play_WAV() is called)
    SDL_PauseAudio( 0 );

    act.sa_handler = timer_callback;
    sigemptyset( &act.sa_mask);
    act.sa_flags = 0;

    // Install handler for alarm signal, sent by expiration of an interval timer
    if ( sigaction( SIGALRM, &act, NULL ) < 0 ) {
        std::cerr << "sdla: Couldn't install SIGALRM handler." << std::endl;
        SDL_CloseAudio();
        SDL_FreeWAV( WAV_buf );
        delete mixed_WAV_buf;
        return;
    }

    instance = this;
    _isvalid = true;
    DMESG( "slda: Created" );
}

sdla::~sdla( void ) {
    if ( !_isvalid ) return;
    DMESG( "slda: Destroying instance..." );
    _isvalid = false;
    instance = 0;
    set_interval( 0, 0 );
    SDL_CloseAudio();
    SDL_FreeWAV( WAV_buf );
    delete mixed_WAV_buf;
    DMESG( "slda: Destroyed" );
}



