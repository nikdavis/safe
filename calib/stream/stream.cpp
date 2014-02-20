/* This file takes static calibration data and loads an undistorted camera frame */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <stdlib.h>
#include "fireflymv_camera.hpp"

#define WIN_NAME    "FireFlyMV Camera Driver Test"

using namespace cv;
using namespace std;

double calibData[] = {  665.5959261296912, 0, 298.4703622855924,
                        0, 664.2752435636575, 218.5832590919033,
                        0, 0, 1};
double distCoeffData[] = {  -0.4543018569873565, 0.6033947908983965, 0.00006727659791911125,
                            0.0003047149111478155, 1.242552873032454 };

Mat calibMat = Mat(3, 3, CV_64F, calibData).clone();
Mat distCoeffMat = Mat(5, 1, CV_64F, distCoeffData).clone();

int main(int argc, char ** argv) {
    Mat frame, raw, map1, map2;; //current frame
    FireflyMVCamera camera;
    namedWindow(WIN_NAME, 1);                  // Create a window for display.
    if(!camera.ready()) {
        printf("Camera did not initialize properly!\n");
        return -1;
    }

    /* Start our app */
    for(;;) {
        /* Grab frame */
        camera.grabFrame(raw);
        frame = raw.clone();
/*
        initUndistortRectifyMap(calibMat, 
							    distCoeffMat, 
							    Mat(),
							    getOptimalNewCameraMatrix(calibMat, distCoeffMat, frame.size(), 1, frame.size(), 0),
							    frame.size(), 
							    CV_16SC2, 
							    map1, map2);

        
	    remap(raw, frame, map1, map2, INTER_LINEAR);
*/
        /* Reduce noise with a kernel 3x3 */
        //GaussianBlur( frame, frame, Size(3, 3), 1.0 );
        /* Canny detector */
        //Canny( frame, frame, 30, 150, 3);
        undistort( raw, frame, calibMat, distCoeffMat );

        //show the current frame and the fg masks
        imshow(WIN_NAME, frame);

        char c = waitKey(10);
        if( c == 27 ) break;

    }

    waitKey(0);

	return 0;
}
