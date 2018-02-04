// FILE: fsource.cpp

#include "defs.hpp"
#include "fsource.hpp"

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

// From video file
fs_video::fs_video( std::string file ) {
    std::cout << "Opening video file \"" << file << '\"' << std::endl;
    video.open( file );
    if ( !video.isOpened() ) {
        std::cerr << "Failed to open video \"" << file << '\"' << std::endl;
        return;
    }
    _width = (int) video.get( CV_CAP_PROP_FRAME_WIDTH );
    _height = (int) video.get( CV_CAP_PROP_FRAME_HEIGHT );
    _valid = true;
    std::cout << "Opened video with resolution " << _width << "x" << _height << std::endl;
}

// From standard webcam / camera
fs_video::fs_video( int cameraIndex ) {
    std::cout << "Opening camera with index \"" << cameraIndex << '\"' << std::endl;
    video.open(cameraIndex);
    if ( !video.isOpened() ) {
        std::cerr << "Failed to open camera \"" << cameraIndex << '\"' << std::endl;
        return;
    }
    _width = (int) video.get( CV_CAP_PROP_FRAME_WIDTH );
    _height = (int) video.get( CV_CAP_PROP_FRAME_HEIGHT );
    _valid = true;
    std::cout << "Opened camera with resolution " << _width << "x" << _height << std::endl;
}

fs_video::~fs_video( void ) {}

int fs_video::get_frame( cv::Mat &frame ) {
    bool ret;

    ret = video.grab();
    if (!ret) {
      std::cout << "Unable to grab next frame" << std::endl;
    } else {
      std::cout << "Grabbed next frame" << std::endl;
    }

    ret = video.retrieve(frame);
    if(!ret) {
      std::cout << "Unable to retrieve frame" << std::endl;
    }

    if ( video.read( frame ) == false ) return -1;
    if ( frame.channels() > 1 ) cvtColor( frame, frame, CV_BGR2GRAY );
    return 0;
}
