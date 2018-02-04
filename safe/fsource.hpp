// FILE: fsource.hpp

#ifndef _FSOURCE_HPP_
#define _FSOURCE_HPP_

#include "opencv2/opencv.hpp"
#include <string>

// All derived classes implement get_frame. This member function is passed a
// Mat by reference. If a frame is available, the Mat is filled and the func
// returns 0. Once all frames are depleted, -1 is retuned.
class frame_source {
    public:
        frame_source( void ) : _valid( false ), _width( 0 ), _height( 0 ) {};
        virtual ~frame_source( void ) {};

        virtual int get_frame( cv::Mat &frame ) { return -1; }
        int is_valid( void ) const { return _valid; }

        int frame_width( void ) const { return _width; }
        int frame_height( void ) const { return _height; }

    protected:
        bool _valid;
        int _width, _height;
};

class fs_image : public frame_source {
    public:
        fs_image( std::string file );
        ~fs_image( void );

        int get_frame( cv::Mat &frame );

    private:
        cv::Mat image;
        bool used;
};

class fs_video : public frame_source {
    public:
        fs_video( std::string file );
        fs_video( int cameraIndex );
        ~fs_video( void );

        int get_frame( cv::Mat &frame );

    private:
       cv::VideoCapture video;
};

#endif // ifndef _FSOURCE_HPP_
