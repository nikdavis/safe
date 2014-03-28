// FILE: main.cpp

#include "defs.hpp"
#include "fsource.hpp"
#include "cvwin.hpp"
#include "timer.hpp"
#include "MSAC.hpp"
#include "homography.hpp"
#include "bayesSeg.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <cmath>

// Pause after processing each frame
#define SINGLE_STEP             true

#define PRINT_TIMES             true
#define PRINT_VP                false
#define PRINT_ANGLES            false
#define PRINT_STATS             true

inline void lane_marker_filter( const cv::Mat &src, cv::Mat &dst );
void init_vp_kalman( cv::KalmanFilter &KF );
void mean_stddev( const cv::Mat &src, float &mean, float &stddev );
inline void show_hough( cv::Mat &dst, const std::vector<cv::Vec4i> lines );
inline bool calc_intersect( const cv::Vec4i l1, const cv::Vec4i l2,
                                                cv::Point &intersect );

int main( int argc, char* argv[] ) {
    cvwin win_a( "i_frame" );
    cvwin win_b( "hough_frame" );
    cvwin win_c( "obj_frame" );
    timer ltimer( "Lane filter         " );
    timer ctimer( "Canny edge detection" );
    timer htimer( "Hough transform     " );
    timer rtimer( "RANSAC              " );
    timer ktimer( "Kalman filter VP    " );
    timer hmtimer( "Homography          " );
    timer etimer( "EM update           " );
    timer itimer( "EM initialization   " );
    timer ptimer( "Process frame       " );
    frame_source* fsrc = NULL;
    cv::Mat frame, lmf_frame, hough_frame, i_frame, obj_frame;
    MSAC msac;
    cv::Size image_size;
    cv::KalmanFilter vpkf( 4, 2, 0 );// 4 dynamic, 2 measurement, and no control
    float prev_mu, prev_sigma;
    BayesianSegmentation bayes_seg;

    // Force update on first frame
    prev_mu = -1000.0;
    prev_sigma = -1000.0;

    cv::Mat_<float> vp = cv::Mat::zeros( 2, 1, CV_32FC1 );
    int key = -1;

    DMESG( "Parsing arguments" );
    if ( argc < 2 ) {
        std::cerr << "Usage is " << argv[0] << " -i/-f/-c [file]\n"
                     "\t-i image\n\t-v video\n\t-c camera" << std::endl;
        return -1;
    }
    if ( argv[1][0] != '-' ) {
        std::cerr << "Must specify source type -i/-v/-c" << std::endl;
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

    srand( 0 );   // For repeatable testing, always seed RNG with zero
    init_vp_kalman( vpkf );

    // Set dst frame size for lane marker filter once
    lmf_frame = cv::Mat::zeros( image_size.height, image_size.width, CV_8UC1 );

    // Request and process frames until source indicates EOF
    while ( fsrc->get_frame( frame ) == 0 ) {
        ptimer.start();

        //** Filter frame for gradient steps up/down horizontally -> lmf_frame
        ltimer.start();
        lane_marker_filter( frame, lmf_frame );
        cv::normalize( lmf_frame, lmf_frame, 0, 255, cv::NORM_MINMAX, CV_8UC1 );
        ltimer.stop();

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
        cv::Mat _vp;    // Temporary holder
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

            // Store into vector of pairs of Points for MSAC
            aux.clear();
            aux.push_back( pt1 );
            aux.push_back( pt2 );
            lineSegments.push_back( aux );
        }
        vp_detected = msac.VPEstimation( lineSegments, lineSegmentsClusters, numInliers, _vp );
        rtimer.stop();
        if ( vp_detected ) msac.drawCS( hough_frame, lineSegmentsClusters, _vp );

        //** Kalman filter RANSAC result
        ktimer.start();
        cv::Mat_<float> pvp(2,1);
        vp = vpkf.predict();
        if( vp_detected ) {
            // Convert _vp from RANSAC to something Kalman filter likes, 2x1 Mat
            pvp(0) = _vp.at<float>(0,0);
            pvp(1) = _vp.at<float>(1,0);
            // Dont update unless VP detected AND it was within frame dimensions
            if ( ( pvp(0) >= 0 ) && ( pvp(0) < fsrc->frame_width() ) &&
                 ( pvp(1) >= 0 ) && ( pvp(1) < fsrc->frame_height() ) ) {
                        vp = vpkf.correct( pvp );
                        if ( PRINT_VP ) cout << "VP: " << pvp(0) << ","
                                             << pvp(1) << endl;
            } else {
                        if ( PRINT_VP ) cout << "VP: ND, ND" << endl;
            }
        }
        ktimer.stop();
        draw_cross( hough_frame, cv::Point( vp(0,0), vp(1,0) ),
                                 cv::Scalar( 0, 255, 0 ), 4 );

        //** homography on frame using filtered intersection -> i_frame
        float theta, gamma;
        cv::Mat H;
        hmtimer.start();
        calcAnglesFromVP( vp, theta, gamma );
        generateHomogMat( H, -theta, -gamma );
        planeToPlaneHomog( frame, i_frame, H, 400 );
        hmtimer.stop();
        if ( PRINT_ANGLES ) cout << "ANGLE: " << theta << "," << gamma << endl;

        //** Calculate homog. intensity feature frame mu and sigma
        float mu, sigma;
        mean_stddev( i_frame, mu, sigma );
        if ( PRINT_STATS )
            std::cout << "MU: " << mu << " SIGMA: " << sigma << std::endl;

        //** If stats significantly different from last frame, reseed EM algor.
        itimer.start();
        if ( ( std::abs( mu    - prev_mu    ) > MU_DELTA    ) || 
             ( std::abs( sigma - prev_sigma ) > SIGMA_DELTA ) ) {
            DMESG( "Significant stat. deltas, reseeding EM algorithm" );
            cv::Mat mask_frame, ip_frame, il_frame, io_frame, l_frame;
            float ip_mu, ip_sigma, il_mu, il_sigma, io_mu, io_sigma;

            //** Sobel gradient filter on i_frame -> mask_frame
            cv::Sobel( i_frame, mask_frame, CV_16S, 1, 0 );
            cv::convertScaleAbs( mask_frame, mask_frame );
            cv::threshold( mask_frame, mask_frame, 80, 255, CV_THRESH_BINARY_INV );

            //** Dilation (erode because of inver.) of mask_frame -> mask_frame
            cv::erode( mask_frame, mask_frame, cv::Mat(), cv::Point(-1,-1), 4 );

            //** Remove mask_frame from i_frame -> ip_frame
            i_frame.copyTo( ip_frame, mask_frame );

            //** Calculate ip_mu and ip_sigma of ip_frame pixels values
            mean_stddev( ip_frame, ip_mu, ip_sigma );

            //** Threshold i_frame above ip_mu + ( 3 * ip_sigma ) -> il_frame
            cv::threshold( i_frame, il_frame, ip_mu + ( 3.0 * ip_sigma ), 255, CV_THRESH_TOZERO );
            mean_stddev( il_frame, il_mu, il_sigma );

            //** Threshold i_frame below ip_mu - ( 3 * ip_sigma ) -> io_frame
            cv::threshold( i_frame, io_frame, ip_mu - ( 3.0 * ip_sigma ), 255, CV_THRESH_TOZERO_INV );
            mean_stddev( io_frame, io_mu, io_sigma );

            //** Reseed (init) EM
            bayes_seg.sigmaInit( 10, 10, 10, UNDEF_DEFAULT_SIGMA );
			bayes_seg.miuInit( 100, 210, 30, UNDEF_DEFAULT_MIU );
            //bayes_seg.sigmaInit( ip_sigma, il_sigma, io_sigma, UNDEF_DEFAULT_SIGMA );
			//bayes_seg.miuInit( ip_mu, il_mu, io_mu, UNDEF_DEFAULT_MIU );
			bayes_seg.probPLOUInit( 0.45, 0.10, 0.40, 0.5 );
			bayes_seg.calcProb();

            if ( PRINT_STATS )
                std::cout << "IP_MU: " << ip_mu << " IP_SIGMA: " << ip_sigma << std::endl;
                std::cout << "IL_MU: " << il_mu << " IL_SIGMA: " << il_sigma << std::endl;
                std::cout << "IO_MU: " << io_mu << " IO_SIGMA: " << io_sigma << std::endl;
        }
        prev_mu = mu;
        prev_sigma = sigma;
        itimer.stop();

        //** Update EM
        etimer.start();
        bayes_seg.EM_Bayes( i_frame );
        etimer.stop();

        //** Create object image
        bayes_seg.classSeg( i_frame, obj_frame, OBJ );

        //** Perform opening
        cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        morphologyEx(obj_frame, obj_frame, cv::MORPH_OPEN, kernel);

        //** Perform blob detection
        //** Generate distance value
        //** tracking stuff...
        //** (new car addition, old car removal, correlation, etc)

        ptimer.stop();

        // Update frame displays
        win_a.display_frame( i_frame );
        win_b.display_frame( hough_frame );
        win_c.display_frame( obj_frame );

        if ( PRINT_TIMES ) {
            // Print timer results
            ltimer.printu();
            ctimer.printu();
            htimer.printu();
            rtimer.printu();
            ktimer.printu();
            hmtimer.printu();
            itimer.printu();
            etimer.printu();
            ptimer.printm();
            std::cout << std::endl;
        }

        // Check for key presses and allow highgui to process events
        if ( SINGLE_STEP ) {
            do { key = cv::waitKey( 1 ); } while( key < 0 );
            if ( key == 27 || key == 'q' ) break;
        }
        else if( ( key = cv::waitKey( 1 ) ) >= 0 ) break;
    }
    DMESG( "Done processing frames" );

    if ( PRINT_TIMES ) {
        // Print average timer results
        ltimer.aprintu();
        ctimer.aprintu();
        htimer.aprintu();
        rtimer.aprintu();
        ktimer.aprintu();
        hmtimer.aprintu();
        itimer.aprintu();
        etimer.aprintu();
        ptimer.aprintm();
    }

    // Pause if no key was pressed during processing loop
    if ( !SINGLE_STEP ) while( key < 0 ) key = cv::waitKey( 1 );

    delete fsrc;
    return 0;
}

inline void lane_marker_filter( const cv::Mat &src, cv::Mat &dst ) {
    int aux;
    int tau_cnt = 0;
    int tau = ROTATE_TAU ? MIN_TAU : TAU;

    for ( int row = src.rows / 2; row < src.rows; ++row ) {
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

void init_vp_kalman( cv::KalmanFilter &KF )
{
    KF.statePre.at<float>(0) = 0;
    KF.statePre.at<float>(1) = 0;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;
    KF.transitionMatrix = *(cv::Mat_<float>(4, 4) <<
                    1,      0,      1/kfdt,		0,
                    0,      1,      0,      1/kfdt,
                    0,      0,      1,      0,
                    0,      0,      0,      1);

    cv::setIdentity( KF.measurementMatrix );
    //cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4));

    KF.processNoiseCov = *(cv::Mat_<float>(4, 4) <<
        pow((float)kfdt, 4)/4.0,    0,  							pow((float)kfdt, 3)/3.0,    0,
        0,  						pow((float)kfdt, 4)/4.0,    	0,  						pow((float)kfdt, 3)/3.0,
        pow((float)kfdt, 3)/3.0,    0,                      		pow((float)kfdt, 2)/2.0,    0,
        0,  						pow((float)kfdt, 3)/3.0,   		0,  						pow((float)kfdt, 2)/2.0);
    KF.processNoiseCov = KF.processNoiseCov*( PROCESS_NOISE * PROCESS_NOISE );

    cv::setIdentity( KF.measurementNoiseCov, cv::Scalar::all( MEAS_NOISE * MEAS_NOISE ) );
    cv::setIdentity( KF.errorCovPost, cv::Scalar::all( 0.00001 ) );
}

// Assumes unsigned byte (uchar) elements, doesn't count zero elements
void mean_stddev( const cv::Mat &src, float &mean, float &stddev ) {
    int hist[256] = {0};
    int accum = 0;
    int size = src.rows * src.cols; // Number of elements
    uchar *pelements = src.data;
    for ( uchar *endp = pelements + size; pelements < endp; ++pelements ) {
        ++hist[*pelements];
        if ( *pelements == 0 ) --size;
        else accum += *pelements;
    }
    if ( size == 0 ) { // If image was empty, zero params and return
        mean = 0;
        stddev = 0;
        return;
    }
    mean = accum / ( float ) size;
    for ( int i = 1, accum = 0; i < 256; ++i ) {
        float delta = ( float ) i - mean;
        accum += ( float ) hist[i] * ( delta * delta );
    }
    stddev = std::sqrt( accum / ( float ) size );
}

inline void show_hough( cv::Mat &dst, const std::vector<cv::Vec4i> lines ) {
    size_t line_count = lines.size();
    for ( size_t i = 0; i < line_count; ++i ) {
        const cv::Vec4i l = lines[i];
        cv::line( dst, cv::Point( l[0], l[1] ),
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



