// FILE: sdla.hpp

#ifndef _SDLA_HPP_
#define _SDLA_HPP_

class sdla {
    public:
        sdla( const char *filename );
        ~sdla( void );

        void play_WAV( void );

        void set_interval( int sec, int msec ); // Zero disables timer
        void set_volume( float vol_left, float vol_right );
        void set_position( float r, float a );  // Call in place of set_volume
        void set_position_deg( float r, int a );

        bool isvalid( void ) const { return _isvalid; }

    private:
        bool _isvalid;

        unsigned char *WAV_buf;         // Original loaded samples
        unsigned char *mixed_WAV_buf;   // Mixed samples
        unsigned int WAV_len;           // Buffer size in bytes

        unsigned char *cur_pos;
        unsigned int cur_len;

        friend void audio_callback( void *, unsigned char *, int );
};

#endif // ifndef _SDLA_HPP_



