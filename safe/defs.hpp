// FILE: defs.hpp

#ifndef _DEFS_HPP_
#define _DEFS_HPP_

#include <iostream>

typedef unsigned char uchar;

// Pause after processing each frame
#define SINGLE_STEP             false

// Lane marker filter parameters
#define ROTATE_TAU              false
#define MIN_TAU                 5
#define MAX_TAU                 15
#define TAU                     7
#define TAU_DELTA               10
#define LANE_FILTER_ROW_OFFSET  0

#define draw_cross( img, center, color, d ) do {        \
    line( img, cv::Point( center.x - d, center.y - d ), \
               cv::Point( center.x + d, center.y + d ), \
               color, 1, CV_AA, 0);                     \
    line( img, cv::Point( center.x + d, center.y - d ), \
               cv::Point( center.x - d, center.y + d ), \
               color, 1, CV_AA, 0 );                    \
    } while ( false )

#ifdef DEBUG
#define DMESG( v ) do { std::cout << v << std::endl; } while ( false )
#else
#define DMESG( v ) do { } while ( false )
#endif

#endif // ifndef _DEFS_HPP_



