// FILE: fsource.cpp

#include "defs.hpp"
#include "fsource.hpp"

// fs_image
fs_image::fs_image( std::string file ) {
    image = cv::imread( file, CV_LOAD_IMAGE_GRAYSCALE );
    if ( image.data == NULL ) return;
    used = false;
    valid = true;
}

fs_image::~fs_image( void ) {}

int fs_image::get_frame( cv::Mat &frame ) {
    if ( valid == false ) return -1;
    if ( used ) return - 1;
    frame = image;
    used = true;
    return 0;
}

// fs_video
fs_video::fs_video( std::string file ) {
    video.open( file );
    if ( !video.isOpened() ) return;
    valid = true;
}

fs_video::~fs_video( void ) {}

int fs_video::get_frame( cv::Mat &frame ) {
    if ( valid == false ) return -1;
    if ( video.read( frame ) == false ) return -1;
    if( frame.channels() > 1 ) cvtColor( frame, frame, CV_BGR2GRAY );
    return 0;
}

// fs_camera
fs_camera::fs_camera( std::string number ) {
    if ( std::stringstream(number) >> camera_select ) return;
    return; // Todo
    valid = true;
}

fs_camera::~fs_camera( void ) {}

int fs_camera::get_frame( cv::Mat &frame ) {
    if ( valid == false ) return -1;
    return -1; // Todo
    return 0;
}



