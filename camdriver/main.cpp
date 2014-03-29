#include "fireflymv_camera.hpp"
//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//C
#include <stdio.h>
//C++
#include <iostream>
#include <sstream>


#define WIN_NAME    "FireFlyMV Camera Driver Test"

using namespace cv;
using namespace std;

double cameraData[] = {  6.3117175205641183e+02, 0., 3.1950000000000000e+02,
                        0., 6.3117175205641183e+02, 2.3950000000000000e+02,
                        0., 0., 1.};
double distCoeffData[] = {  -4.1605062165297507e-01, 2.6505676737778694e-01,
                            -5.2493360426022302e-03, -2.5224864678663654e-03,
                            -1.4925040070852708e-01 };
cv::Mat cameraMat = cv::Mat(3, 3, CV_64F, cameraData).clone();
cv::Mat distCoeffMat = cv::Mat(5, 1, CV_64F, distCoeffData).clone();

int main(int argc, char** argv)
{
    Mat frame, temp; //current frame
    FireflyMVCamera camera;
    int undist = 0;

    namedWindow(WIN_NAME, 1);                  // Create a window for display.
    if(!camera.ready()) {
        printf("Camera did not initialize properly!\n");
        return -1;
    }

    cout << "Press u to toggle between distorted, undistorted.\n";

    /* Start our app */
    for(;;) {
        /* Grab frame */
        camera.grabFrame(frame);
        /* Reduce noise with a kernel 3x3 */
        //GaussianBlur( frame, frame, Size(3, 3), 1.0 );
        /* Canny detector */
        //Canny( frame, frame, 30, 150, 3);

        if(undist) {
            temp = frame.clone();
            undistort(temp, frame, cameraMat, distCoeffMat);
        }
        //show the current frame and the fg masks
        imshow(WIN_NAME, frame);

        char c = waitKey(10);
        
        if( c == 27 ) break;
        else if (c == 117) {
            undist = (undist + 1) % 2;
            cout << "Switched undistort mode to: " << undist << endl;
        }

    }

    waitKey(0);

	return 0;
}
