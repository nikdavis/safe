// FILE: sdla.cpp

#include "sdla.hpp"
#include <iostream>

extern "C"
{
#include "SDL.h"
}

#ifdef DEBUG
#define DMESG( v ) do { std::cout << v << std::endl; } while ( false )
#else
#define DMESG( v ) do { } while ( false )
#endif

// Callback used by SDL audio thread to fill audio buffer
void audio_callback( void *userdata, unsigned char *stream, int len ) {
    sdla *psdla = (sdla*)userdata;

    // Only copy if we have samples left
    if ( psdla->cur_len == 0 ) return;

    // Copy as much data as possible
    len = ( (unsigned)len > psdla->cur_len ? psdla->cur_len : len );
    memcpy( stream, psdla->cur_pos, len );

    psdla->cur_pos += len;
    psdla->cur_len -= len;
}

void sdla::play_WAV( void ) {
    // As we are touching data structures accessed in the callback, lock it
    SDL_LockAudio();

    // Reset buffer index to start of loaded WAV file
    cur_pos = WAV_buf;
    cur_len = WAV_len;

    SDL_UnlockAudio();
}

sdla::sdla( const char *filename ) : _isvalid( false ), cur_len( 0 ) {
    SDL_AudioSpec requested_spec, obtained_spec;

    // Open WAV file, load data into buffer, and read spec from header
    if ( SDL_LoadWAV( filename, &requested_spec,
                &WAV_buf, &WAV_len ) == 0 ) {
        std::cerr << "Couldn't open " << filename << ": "
                    << SDL_GetError() << std::endl;
        return;
    }

    DMESG( "Freq: " << requested_spec.freq <<
          " Channels: " << (int)requested_spec.channels <<
          " Samples: " << requested_spec.samples );

    requested_spec.callback = audio_callback;
    requested_spec.userdata = this;

    if ( SDL_OpenAudio( &requested_spec, &obtained_spec ) < 0 ) {
        std::cerr << "Couldn't open audio: " << SDL_GetError() << std::endl;
        SDL_FreeWAV( WAV_buf );
        
        return;
    }

    SDL_PauseAudio( 0 );

    _isvalid = true;
}

sdla::~sdla( void ) {
    if ( !_isvalid ) return;
    SDL_CloseAudio();
    SDL_FreeWAV( WAV_buf );
    _isvalid = false;
}



