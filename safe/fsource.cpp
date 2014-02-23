// FILE: fsource.cpp

#include "defs.hpp"
#include "fsource.hpp"
#include "fireflymv_camera.hpp"

// fs_image
fs_image::fs_image( std::string file ) : used( false ) {
    image = cv::imread( file, CV_LOAD_IMAGE_GRAYSCALE );
    if ( image.data == NULL ) {
        std::cerr << "Failed to open image \"" << file << '\"' << std::endl;
        return;
    }
    _width = image.cols;
    _height = image.rows;
    _valid = true;
}

fs_image::~fs_image( void ) {}

int fs_image::get_frame( cv::Mat &frame ) {
    if ( used ) return - 1;
    frame = image;
    used = true;
    return 0;
}

// fs_video
fs_video::fs_video( std::string file ) {
    DMESG( "Opening video file \"" << file << '\"' );
    video.open( file );
    if ( !video.isOpened() ) {
        std::cerr << "Failed to open video \"" << file << '\"' << std::endl;
        return;
    }
    _width = (int) video.get( CV_CAP_PROP_FRAME_WIDTH );
    _height = (int) video.get( CV_CAP_PROP_FRAME_HEIGHT );
    _valid = true;
}

fs_video::~fs_video( void ) {}

int fs_video::get_frame( cv::Mat &frame ) {
    if ( video.read( frame ) == false ) return -1;
    if ( frame.channels() > 1 ) cvtColor( frame, frame, CV_BGR2GRAY );
    return 0;
}

// fs_camera
fs_camera::fs_camera( void ) {
    pFFCam = new FireflyMVCamera();
    if ( pFFCam == NULL ) {
        std::cerr << "Failed to allocate Firefly camera" << std::endl;
        return;
    }
    if ( !pFFCam->ready() ) {
        std::cerr << "Failed to initialize Firefly camera" << std:: endl;
        return;
    }
    _width = CAM_WIDTH;
    _height = CAM_HEIGHT;
    _valid = true;
}

fs_camera::~fs_camera( void ) {
    delete pFFCam;
}

int fs_camera::get_frame( cv::Mat &frame ) {
    return pFFCam->grabFrame( frame );
}



