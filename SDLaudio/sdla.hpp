// FILE: sdla.hpp

#ifndef _SDLA_HPP_
#define _SDLA_HPP_

class sdla {
    public:
        sdla( const char *filename );
        ~sdla( void );

        void play_WAV( void );

        bool isvalid( void ) const { return _isvalid; }

    private:
        bool _isvalid;

        unsigned char *WAV_buf;
        unsigned int WAV_len;

        unsigned char *cur_pos;
        unsigned int cur_len;

        friend void audio_callback( void *, unsigned char *, int );
};

#endif // ifndef _SDLA_HPP_



