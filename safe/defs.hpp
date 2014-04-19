// FILE: defs.hpp

#ifndef _DEFS_HPP_
#define _DEFS_HPP_

#include <iostream>

typedef unsigned char uchar;

#define VERBOSE_MSAC            false

#define CAM_WIDTH               640
#define CAM_HEIGHT              480

// Transitory state thresholds TODO: Pick based on actual data; I just guessed
#define MU_DELTA                ( 5 )
#define SIGMA_DELTA             ( 5 )

// Lane marker filter parameters
#define ROTATE_TAU              false
#define MIN_TAU                 5
#define MAX_TAU                 15
#define TAU                     10
#define TAU_DELTA               10
#define LANE_FILTER_ROW_OFFSET  0

// Kalman filter parameters
#define DELAY_MS                62.5
#define kfdt                    ( DELAY_MS / 1000.0 )
#define SAMPLE_FREQ             16.0
#ifndef kfdt
#define kfdt                    ( 1.0 / SAMPLE_FREQ )
#endif

#define MEAS_NOISE              0.005
#define PROCESS_NOISE           0.5

// Helper macros
#define draw_cross( img, center, color, d ) do {            \
    cv::line( img, cv::Point( center.x - d, center.y - d ), \
               cv::Point( center.x + d, center.y + d ),     \
               color, 1, CV_AA, 0);                         \
    cv::line( img, cv::Point( center.x + d, center.y - d ), \
               cv::Point( center.x - d, center.y + d ),     \
               color, 1, CV_AA, 0 );                        \
    } while ( false )

#ifdef DEBUG
#define DMESG( v ) do { std::cout << v << std::endl; } while ( false )
#else
#define DMESG( v ) do { } while ( false )
#endif

#endif // ifndef _DEFS_HPP_



