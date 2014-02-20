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
        frame_source( void ) : valid( false ) {};
        virtual ~frame_source( void ) {};

        virtual int get_frame( cv::Mat &frame ) { return -1; }
        int is_valid( void ) const { return valid; }

    protected:
        bool valid;
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
        ~fs_video( void );

        int get_frame( cv::Mat &frame );

    private:
       cv::VideoCapture video;
};

class FireflyMVCamera; // Forward declare Firefly camera class

class fs_camera : public frame_source {
    public:
        fs_camera( std::string number );
        ~fs_camera( void );

        int get_frame( cv::Mat &frame );

    private:
        FireflyMVCamera *pFFCam;
};

#endif // ifndef _FSOURCE_HPP_



