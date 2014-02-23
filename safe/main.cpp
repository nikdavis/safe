// FILE: main.cpp

#include "defs.hpp"
#include "fsource.hpp"
#include "cvwin.hpp"
#include "timer.hpp"
#include "MSAC.hpp"
#include "opencv2/opencv.hpp"
#include <string>

inline void lane_marker_filter( const cv::Mat &src, cv::Mat &dst );
inline void show_hough( cv::Mat &dst, const std::vector<cv::Vec4i> lines );
inline bool calc_intersect( const cv::Vec4i l1, const cv::Vec4i l2,
                                                cv::Point &intersect );

int main( int argc, char* argv[] ) {
    cvwin win_frame( "frame" );
    cvwin win_hough( "hough" );
    timer ctimer( "Canny edge detection" );
    timer htimer( "Hough transform     " );
    timer rtimer( "RANSAC              " );
    frame_source* fsrc = NULL;
    cv::Mat frame, lmf_frame, hough_frame;
    MSAC msac;
    cv::Size image_size;
    int key = -1;

    DMESG( "Parsing arguments" );
    if ( argc < 2 ) {
        std::cerr << "Usage is " << argv[0] << " -i/-f/-c [file]\n"
                     "\t-i image\n\t-v video\n\t-c camera" << std::endl;
        return -1;
    }
    if ( argv[1][0] != '-' ) {
        std::cerr << "Must specify source type -i/-f/-c" << std::endl;
        return -1;
    }
    switch( argv[1][1] ) {
        case 'i':
            DMESG( "Creating image frame source" );
            if ( argc < 3 ) {
                std::cerr << "No file specified" << std::endl;
                return -1;
            }
            fsrc = new fs_image( argv[2] );
            break;
        case 'v':
            DMESG( "Creating video frame source" );
            if ( argc < 3 ) {
                std::cerr << "No file specified" << std::endl;
                return -1;
            }
            fsrc = new fs_video( argv[2] );
            break;
        case 'c':
            DMESG( "Creating camera frame source" );
            fsrc = new fs_camera( );
            break;
        default:
            std::cerr << "Unknown argument \"" << argv[1] << '\"' << std::endl;
            return -1;
    }
    if ( fsrc == NULL ) {
        std::cerr << "Failed to allocate frame source" << std::endl;
        return -1;
    }
    if ( fsrc->is_valid() == false ) {
        std::cerr << "Frame source is invalid" << std::endl;
        delete fsrc;
        return -1;
    }
    DMESG( "Frame source is valid" );

    image_size.width = fsrc->frame_width();
    image_size.height = fsrc->frame_height();
    msac.init( MODE_NIETO, image_size );

    // Request and process frames until source indicates EOF
    while ( fsrc->get_frame( frame ) == 0 ) {
        //** Filter frame for gradient steps up/down horizontally -> lmf_frame
        lane_marker_filter( frame, lmf_frame );

        //** Perform Canny edge detection on lmf_frame -> lmf_frame
        ctimer.start();
        // src, dst, low threshold, high threshold, kernel size, accurate
        cv::Canny( lmf_frame, lmf_frame, 100, 200, 3, false );
        ctimer.stop();

        //** Perform Hough transform on lmf_frame, generating vector of lines
        htimer.start();
        std::vector<cv::Vec4i> hlines;
        // src, dst vec, rho, theta, threshold, min length, max gap
        // rho - distance resolution of accumulator in pixels
        // theta - angle resolution of accumulator in pixels
        cv::HoughLinesP( lmf_frame, hlines, 1, CV_PI / 180.0, 50, 50, 10 );
        htimer.stop();

        // Visualize Hough transform results
        cv::cvtColor( lmf_frame, hough_frame, CV_GRAY2BGR );
        //show_hough( hough_frame, hlines );

        // Show intersections of every point
        /*cv::Point intersect;
        size_t line_count = hlines.size();
        for( size_t i = 0; i < line_count; ++i )
            for ( size_t j = i + 1; j < line_count; ++j )
                if ( calc_intersect( hlines[i], hlines[j], intersect ) )
                    draw_cross( hough_frame, intersect,
                                cv::Scalar(0,255,0), 3 );*/

        //** RANSAC hough line intersections for vanishing point
        rtimer.start();
        cv::Mat vp;
        std::vector<cv::Point> aux;
        std::vector<std::vector<cv::Point> > lineSegments;
        std::vector<int> numInliers;
        std::vector<std::vector<std::vector<cv::Point> > > lineSegmentsClusters;
        bool vp_detected;
        for( size_t i = 0; i < hlines.size(); ++i )
        {
            cv::Point pt1, pt2;
            pt1.x = hlines[i][0];
            pt1.y = hlines[i][1];
            pt2.x = hlines[i][2];
            pt2.y = hlines[i][3];

            // Store into vector of pairs of Points for msac
            aux.clear();
            aux.push_back( pt1 );
            aux.push_back( pt2 );
            lineSegments.push_back( aux );
        }
        vp_detected = msac.VPEstimation( lineSegments, lineSegmentsClusters, numInliers, vp );
        rtimer.stop();
        if ( vp_detected ) msac.drawCS( hough_frame, lineSegmentsClusters, vp );

        //** Kalman filter RANSAC result
        //** homography on frame using filtered intersection -> i_frame
        //** Sobel gradient filter on i_frame -> mask_frame
        //** dilation of mask_frame -> mask_frame
        //** remove mask_frame from i_frame -> ip_frame
        //** ip_mu and ip_sigma of ip_frame pixels values calculated
        //** threshold i_frame above ip_mu + ( 3 * ip_sigma ) -> il_frame
        //** threshold i_frame below ip_mu - ( 3 * ip_sigma ) -> io_frame
        //** lane filter applied to i_frame -> l_frame
        //** l_mu and l_sigma of l_frame pixel values calculated
        //** threshold l_frame above l_mu + l_sigma -> ll_frame
        //** threshold l_frame below l_mu - l_sigma -> lpo_frame
        //** ll_mu and ll_sigma of ll_frame pixels values calculated
        //** lpo_mu and lpo_sigma of lpo_frame pixels values calculated
        // EM stuff using calculated parameters ... ?

        // Update frame displays
        win_frame.display_frame( frame );
        win_hough.display_frame( hough_frame );

        // Print timer results
        ctimer.printu();
        htimer.printu();
        rtimer.printu();

        // Check for key presses and allow highgui to process events
        if ( SINGLE_STEP ) {
            do { key = cv::waitKey( 1 ); } while( key < 0 );
            if ( key == 27 || key == 'q' ) break;
        }
        else if( ( key = cv::waitKey( 1 ) ) >= 0 ) break;
    }
    DMESG( "Done processing frames" );

    // Print average timer results
    ctimer.aprintu();
    htimer.aprintu();
    rtimer.aprintu();

    // Pause if no key was pressed during processing loop
    if ( !SINGLE_STEP ) while( key < 0 ) key = cv::waitKey( 1 );

    delete fsrc;
    return 0;
}

inline void lane_marker_filter( const cv::Mat &src, cv::Mat &dst ) {
    int aux;
    int tau_cnt = 0;
    int tau = ROTATE_TAU ? MIN_TAU : TAU;

    dst = cv::Mat::zeros( src.rows, src.cols, CV_8UC1 );

    for ( int row = 0; row < src.rows; ++row ) {
        const uchar *s = src.ptr<uchar>(row);
        uchar *d = dst.ptr<uchar>(row);

        for ( int col = 0; col < src.cols; ++col ) {
            // Check that we're within kernel size
            if ( ( col >= tau ) && ( col <  (src.cols - tau ) ) &&
                ( row > ( ( src.rows / 2 ) + LANE_FILTER_ROW_OFFSET ) ) ) {
                // Filter from Nietos 2010
                aux = 2 * s[col];
                aux -= s[col - tau];
                aux -= s[col + tau];
                aux -= abs( ( s[col - tau] - s[col + tau] ) );
                aux = ( aux < 0   ) ? 0   : aux; // Could apply custom threshold
                aux = ( aux > 255 ) ? 255 : aux; // Could apply custom threshold
                d[col] = (uchar)aux;
            }
            else d[col] = 0;
        }
        if ( row > ( ( src.rows / 2 ) + LANE_FILTER_ROW_OFFSET ) ) {
            ++tau_cnt;
            if ( ( tau_cnt % ( ( src.rows / 2 ) / TAU_DELTA ) ) == 0 ) ++tau;
        }
    }
}


inline void show_hough( cv::Mat &dst, const std::vector<cv::Vec4i> lines ) {
    size_t line_count = lines.size();
    for ( size_t i = 0; i < line_count; ++i ) {
        const cv::Vec4i l = lines[i];
        line( dst, cv::Point( l[0], l[1] ),
                   cv::Point( l[2], l[3] ),
                   cv::Scalar( 0, 0, 255 ), 1, CV_AA, 0 );
    }
}

inline bool calc_intersect( const cv::Vec4i l1, const cv::Vec4i l2,
                                                cv::Point &intersect ) {
    float x1, x2, y1, y2, m1, c1, m2, c2;

    x1 = l1[0]; y1 = l1[1]; x2 = l1[2]; y2 = l1[3];
    m1 = ( y2 - y1 ) / ( x2 - x1 );
    c1 = y1 - ( m1 * x1 );

    x1 = l2[0]; y1 = l2[1]; x2 = l2[2]; y2 = l2[3];
    m2 = ( y2 - y1 ) / ( x2 - x1 );
    c2 = y1 - ( m2 * x1 );

    if ( m1 == m2 ) return false; // Parallel
    intersect.x = ( c2 - c1 ) / ( m1 - m2 );
    intersect.y = ( m1 * intersect.x ) + c1;
    return true;
}



