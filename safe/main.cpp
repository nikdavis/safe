// FILE: main.cpp

#include "defs.hpp"
#include "fsource.hpp"
#include "cvwin.hpp"
#include "timer.hpp"
#include "MSAC.hpp"
#include "kalman.hpp"
#include "homography.hpp"
#include "bayesSeg.hpp"
#include "carTracking.hpp"
#include "sdla.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <cmath>
#include <fstream>
#include <string>

// Pause after processing each frame
#define SINGLE_STEP             false
#define MOTORCYCLE              true

#define PRINT_TIMES             true
#define PRINT_VP                false
#define PRINT_ANGLES            true
#define PRINT_STATS             true

inline bool saveImg(const cv::Mat &img, std::string fileNameFormat, int fileNum);
inline void lane_marker_filter( const cv::Mat &src, cv::Mat &dst );
inline void show_hough( cv::Mat &dst, const std::vector<cv::Vec4i> lines );
inline bool calc_intersect( const cv::Vec4f l1, const cv::Vec4f l2,
                                                cv::Point2f &intersect );

/* Calibration data --- This is static and we need to include it as
 * a header because other libs (ie homography - ie Mat K) need it as well.*/
double cameraData[] = {  6.3117175205641183e+02, 0., 3.1950000000000000e+02,
                        0., 6.3117175205641183e+02, 2.3950000000000000e+02,
                        0., 0., 1.};
double distCoeffData[] = {  -4.1605062165297507e-01, 2.6505676737778694e-01,
                            -5.2493360426022302e-03, -2.5224864678663654e-03,
                            -1.4925040070852708e-01 };
cv::Mat cameraMat = cv::Mat(3, 3, CV_64F, cameraData).clone();
cv::Mat distCoeffMat = cv::Mat(5, 1, CV_64F, distCoeffData).clone();


int main( int argc, char* argv[] ) {
    bool motorcycle = MOTORCYCLE;
    srand(0); // Force consistent results on reruns

    sdla alarm( "boop.wav" );

    int undist = 1;
    cvwin win_a( "frame" );
    cvwin win_b( "bird_frame" );
    cvwin win_c( "hough_frame" );
    //cvwin win_d( "obj_frame" );
    cvwin win_e( "blob display" );
    timer utimer( "Undistort           " );
    timer ltimer( "Lane filter         " );
    timer ctimer( "Canny edge detection" );
    timer htimer( "Hough transform     " );
    timer rtimer( "RANSAC              " );
    timer ktimer( "Kalman filter VP    " );
    timer hmtimer( "Homography          " );
    timer etimer( "EM update           " );
    timer itimer( "EM initialization   " );
    timer btimer( "Blob detection      " );
    timer ptimer( "Process frame       " );
    frame_source* fsrc = NULL;

    cv::Mat frame_raw, frame, lmf_frame, hough_frame, bird_frame, obj_frame;
    MSAC msac;
    cv::Size image_size;
    double gammaInit, thetaInit;
    if(motorcycle) {
        thetaInit = -2.635;
        gammaInit = 1.124;
    } else {
        thetaInit = -7.0;
        gammaInit = 1.5;
    }

    Kalman1D theta(thetaInit, 0.05f, 0.00005f);       /* Kalman filters for Theta, Gamma */
    Kalman1D gamma(gammaInit, 0.05f, 0.00005f);        /* init val, measure var, proc var */
    float prev_mu, prev_sigma;
    BayesianSegmentation    bayes_seg;
    CarTracking             car_track;

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

    float prev_x = fsrc->frame_width() / 2.0;
    float prev_y = fsrc->frame_height() / 2.0;

    // Set dst frame size for lane marker filter once
    lmf_frame = cv::Mat::zeros( image_size.height, image_size.width, CV_8UC1 );

    // Request and process frames until source indicates EOF
    int frame_count = 0;
    while ( fsrc->get_frame( frame_raw ) == 0 ) {
        ptimer.start();
		frame_count++;
        /* Explicitly undistorting our FireflyMV camera */

        utimer.start();
        if ( undist ) {
            cv::undistort(frame_raw, frame, cameraMat, distCoeffMat);
        } else {
            frame = frame_raw;
        }
        utimer.stop();
        
        //cv::flip( frame, frame, 1);

        /* Gaussian helps preprocess noise out for LMF/Canny/Hough */
        cv::GaussianBlur( frame, frame, cv::Size(3, 3), 0, 0 );
        cv::GaussianBlur( frame, frame, cv::Size(3, 3), 0, 0 );

        //** Filter frame for gradient steps up/down horizontally -> lmf_frame
        ltimer.start();
        lane_marker_filter( frame, lmf_frame );
        //cv::normalize( lmf_frame, lmf_frame, 0, 255, cv::NORM_MINMAX, CV_8UC1 );
        ltimer.stop();

        //** Perform Canny edge detection on lmf_frame -> lmf_frame
        ctimer.start();
        // src, dst, low threshold, high threshold, kernel size, accurate
        cv::Canny( lmf_frame, lmf_frame, 60, 170, 3, false );
        ctimer.stop();

        //** Perform Hough transform on lmf_frame, generating vector of lines
        htimer.start();
        std::vector<cv::Vec4i> hlines;
        // src, dst vec, rho, theta, threshold, min length, max gap
        // rho - distance resolution of accumulator in pixels
        // theta - angle resolution of accumulator in pixels
        cv::HoughLinesP( lmf_frame, hlines, 3, CV_PI / 60.0, 40, 40, 10 );
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

        /* Process Kalman */
        ktimer.start();
        cv::Mat_<float> pvp(2,1);
        float thetaAng, gammaAng, thetaDelta, gammaDelta, xdelta, ydelta;
        if( vp_detected ) {
            /* Ensure VP values are not NaN! Default to center of frame */
            if ( IS_NAN(_vp.at<float>(0,0)) || IS_NAN(_vp.at<float>(1,0)) ) {
                pvp(0) = fsrc->frame_width() / 2.0;
                pvp(1) = fsrc->frame_height() / 2.0;
            }
            else {
                // Convert _vp from RANSAC to something Kalman filter likes, 2x1 Mat
                pvp(0) = _vp.at<float>(0,0);
                pvp(1) = _vp.at<float>(1,0);
            }
            
            // If the vanishing point is out of the frame, the point is not good.
            if ((pvp(0) < 2 * frame.cols / 7) || (pvp(0) > 5 * frame.cols / 7) || 
            	(pvp(1) < 2 * frame.rows / 5) || (pvp(1) > 3 * frame.rows / 5)) {
            	theta.skipMeas();
            	gamma.skipMeas();
            } else {
		        /* Convert to units of Kalman filter */
		        calcAnglesFromVP(pvp, thetaAng, gammaAng);
		        /* Only process if the delta is sane */
		        thetaDelta = fabs(thetaAng - theta.xHat);
		        gammaDelta = fabs(gammaAng - gamma.xHat);
		        xdelta = fabs(pvp(0) - prev_x); prev_x = pvp(0);
		        ydelta = fabs(pvp(1) - prev_y); prev_y = pvp(1);
		        if ( thetaDelta < 15.0 && ydelta < 10.0 ) {
		            theta.addMeas(thetaAng);
		        } else {
		            theta.skipMeas();
		        }
		        if ( gammaDelta < 25.0 && xdelta < 10.0 ) {
		            gamma.addMeas(gammaAng);
		        } else {
		            gamma.skipMeas();
		        }
            }
        } else {
            theta.skipMeas();
            gamma.skipMeas();
        }
        ktimer.stop();
        draw_cross( hough_frame, cv::Point( vp(0,0), vp(1,0) ),
                                 cv::Scalar( 0, 255, 0 ), 4 );
		cv::Point filter_vp;
		calcVpFromAngles(theta.xHat, gamma.xHat, filter_vp); 
		cv::circle( hough_frame, filter_vp, 3, cv::Scalar(0, 255, 255 ), 4 );
                        
        /* Generate IPM or BIRDS-EYE view with plane-to-plane homography */
        hmtimer.start();
        cv::Mat H;
        generateHomogMat(H, -theta.xHat, -gamma.xHat);
        planeToPlaneHomog(frame, bird_frame, H, 400);
        hmtimer.stop();

        /* Printing prints estimates of the angles, not raw */
        if ( PRINT_ANGLES ) std::cout << "ANGLE: " << theta.xHat << "," << gamma.xHat << std::endl;

        //** Calculate homog. intensity feature frame mu and sigma
        float mu, sigma;        
        bayes_seg.mean_stddev( bird_frame, mu, sigma );
        if ( PRINT_STATS )
            std::cout << "MU: " << mu << " SIGMA: " << sigma << std::endl;
        
        //** If stats significantly different from last frame, reseed EM algor.
        itimer.start();
        if ( ( abs( mu    - prev_mu    ) > MU_DELTA    ) ||
             ( abs( sigma - prev_sigma ) > SIGMA_DELTA ) ) {
            DMESG( "Significant stat. deltas, reseeding EM algorithm" );
            bayes_seg.autoInitEM( bird_frame );
        }
        prev_mu = mu;
        prev_sigma = sigma;
        itimer.stop();

        //** Update EM
        etimer.start();
        bayes_seg.EM_Bayes( bird_frame );
        etimer.stop();

        //** Create object image
        bayes_seg.classSeg( bird_frame, obj_frame, OBJ );
        
        //saveImg( obj_frame, "./frame/obj_frame", frame_count);

        //** Perform opening
        cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(obj_frame, obj_frame, cv::MORPH_OPEN, kernel);
		//cv::erode(obj_frame, obj_frame, kernel);
		//saveImg( obj_frame, "./frame/erode_frame", frame_count);
		//cv::dilate(obj_frame, obj_frame, kernel);
		//saveImg( obj_frame, "./frame/dilate_frame", frame_count);
		
        //** Perform blob detection then passing object candidates through
        // temporal filter. Only blobs that show up in a certain consecutive
        // frame are considered as cars. And only cars are passed through 
        // the extended Kalman filter.
        btimer.start();
        car_track.detect_filter( obj_frame );
        btimer.stop();

        //** Find the blob bounding boxes then display them.
        car_track.findBoundContourBox( obj_frame );

        cv::Mat blob_disp = obj_frame.clone();
        cv::cvtColor( blob_disp, blob_disp, CV_GRAY2RGB);
        for (unsigned int i = 0; i < car_track.boundRect.size(); i++) {
            cv::rectangle( blob_disp, car_track.boundRect[i], cv::Scalar(255, 0, 0), 2, 4, 0 );
        }

        bool alarming = false;
        for (unsigned int i = 0; i < car_track.objCands.size(); i++)
        {
            // show the center point of all blobs detected
            cv::circle( blob_disp, car_track.objCands[i].Pos, 3, cv::Scalar(0, 0, 255), -1);
            
            // Only cars will be calculated position and direction
            if (car_track.objCands[i].inFilter)
            {
                // Currently ignoreing EKF, it clearly need absolute velocity to function properly
                // As is, the EKF is very slow to follow, and exibits odd behavior as it models
                // a car on he road turning around, etc., as it's relative velocity changes
                cv::circle( blob_disp, car_track.objCands[i].Pos, 3, cv::Scalar(255, 0, 0), -1);
                cv::circle( blob_disp, car_track.objCands[i].filterPos, 3, cv::Scalar(0, 255, 0), -1);

                cv::Point2f cvtP;
                car_track.cvtCoord(car_track.objCands[i].filterPos, cvtP, obj_frame);
                cv::Point2f feetPos((float)cvtP.x*PX_FEET_SCALE, (float)cvtP.y*PX_FEET_SCALE);

                cv::Point2f lstart = car_track.objCands[i].filterPos;
                cv::Point2f lend = car_track.objCands[i].filterPos + 100 * car_track.objCands[i].filterVelo;
                cv::line( blob_disp, lstart, lend, cv::Scalar(0, 128, 0), 4);

                std::cout << "Object[" << i << "] x: " << feetPos.x << " y: " << feetPos.y
                          << " xv: " << car_track.objCands[i].filterVelo.x
                          << " yv: " << car_track.objCands[i].filterVelo.y << std::endl;

                // http://www.michigan.gov/documents/msp/BrakeTesting-MSP_VehicleEval08_Web_221473_7.pdf
                // Average was 26.86ft/s^2 or about 8 m/s^2 braking acceleration
                // For safety, assume max braking of 6 m/s
                cv::Point2f intersection;
                cv::Vec4f candVec( lstart.x, lstart.y, lend.x, lend.y );
                cv::Vec4f hitSeg( 100, 479, 300, 479 );
                if ( calc_intersect( candVec, hitSeg, intersection ) ) {
                    // Intersected with rear of vehicle, check stopping distance
                    // Ignore x velocity, y should be very very dominant
                    float vy = car_track.objCands[i].filterVelo.y;
                    //TODO: convert velocity from pixels per frame to meters per second
                    //vy *= some conversion factor;
                    float stopdist = ( vy * vy ) / ( 2.0 * 6.0 );
                    float dist = 480 - car_track.objCands[i].filterPos.y;
                    if ( stopdist > dist ) {
                        // Alert user of potential hazard
                        // TODO: Setup leveled response, deal with priority properly
                        std::cout << "\033[22;31mAERT!\e[m" << std::endl;
                        alarming = true;
                        alarm.set_interval( 0, 100 );
                        alarm.play_WAV();
                    }
                }
            }
        }
        if ( !alarming ) alarm.set_interval( 0, 0 ); // Turn off alarm

        ptimer.stop();

        // Update frame displaysnad box
        win_a.display_frame( frame );
        win_b.display_frame( bird_frame );
        win_c.display_frame( hough_frame );
        //win_d.display_frame( obj_frame );
        win_e.display_frame( blob_disp );

        if ( PRINT_TIMES ) {
            // Print timer results
            utimer.printu();
            ltimer.printu();
            ctimer.printu();
            htimer.printu();
            rtimer.printu();
            ktimer.printu();
            hmtimer.printu();
            itimer.printu();
            etimer.printu();
            btimer.printu();
            ptimer.printm();
            std::cout << std::endl;
        }

        // Check for key presses and allow highgui to process events
        key = cv::waitKey(1);
        if ( SINGLE_STEP ) {
            while( key < 0 ) {
                key = cv::waitKey(1);
            }
        }
    /* Quit if key is ESC or q */
        if(key == 27 || key == 'q') break;
        else if(key == 'u') {
            /* u key switches undistortion on/off. default is on. */
            undist = (undist + 1) % 2;
        }
    }
    DMESG( "Done processing frames" );

    if ( PRINT_TIMES ) {
        // Print average timer results
        utimer.aprintu();
        ltimer.aprintu();
        ctimer.aprintu();
        htimer.aprintu();
        rtimer.aprintu();
        ktimer.aprintu();
        hmtimer.aprintu();
        itimer.aprintu();
        etimer.aprintu();
        btimer.aprintu();
        ptimer.aprintm();
    }

    // Pause if no key was pressed during processing loop
    if ( !SINGLE_STEP ) while( key < 0 ) key = cv::waitKey( 30 );

    std::cout << "Cleaning up..." << std::endl;

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
                // Filter from Nieto 2010
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
        cv::line( dst, cv::Point( l[0], l[1] ),
                       cv::Point( l[2], l[3] ),
                       cv::Scalar( 0, 0, 255 ), 1, CV_AA, 0 );
    }
}

inline bool calc_intersect( const cv::Vec4f l1, const cv::Vec4f l2,
                                                cv::Point2f &intersect ) {
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

inline bool saveImg(const cv::Mat &img, std::string fileNameFormat, int fileNum)
{
	cv::Mat save;
	if (img.channels() == 3)
		cv::cvtColor(img, save, CV_RGB2GRAY);
	else if (img.channels() == 4)
		cv::cvtColor(img, save, CV_RGBA2GRAY);
	else
		save = img.clone();
	// Save image as pgm
	std::vector< int > compression_params;			//vector that stores the compression parameters of the image
	compression_params.push_back(CV_IMWRITE_PXM_BINARY);
	compression_params.push_back(9);
	std::string sFileNum = static_cast<std::ostringstream*>(&(std::ostringstream() << fileNum))->str();
	std::string sFileName = fileNameFormat;
	if (fileNum < 10)
		sFileName += ("_000" + sFileNum + ".pgm");
	else if ((fileNum >= 10) && (fileNum < 100))
		sFileName += ("_00" + sFileNum + ".pgm");
	else if ((fileNum >= 100) && (fileNum < 1000))
		sFileName += ("_0" + sFileNum + ".pgm");
	else
		sFileName += ("_" + sFileNum + ".pgm");
	return cv::imwrite(sFileName, save, compression_params);
}



