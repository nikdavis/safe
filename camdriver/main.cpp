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
#if 0
    unsigned int shutter, oldShutter, frameNum;
    shutter = oldShutter = frameNum = 0;
    float shutterFl, integError, derivError, dt, Ku, Pu, Kp, Ki, Kd, setPoint, p, d, i;
    float error[3] = {0, 0, 0};
    setPoint = 90.0;
    integError = 0;
    derivError = 0;
    dt = 1.0 / 30.0;
    Ku = 0.14;
    Pu = 0.45;
    Kp = 0.45 * Ku;
    Ki = 1.7 * Kp / Pu;
    Kd = Kp * Pu / 8.0;
    p = i = d = 0.0;
#endif
    namedWindow(WIN_NAME, 1);                  // Create a window for display.
    if(!camera.ready()) {
        printf("Camera did not initialize properly!\n");
        return -1;
    }

    //cout << "Press u to toggle between distorted, undistorted.\n";

    /* Start our app */
    for(;;) {
        /* Grab frame */
        camera.grabFrame(frame);
#if 0
        Mat roadFrame = frame(Range(270, 480), Range(0,640));
        Scalar roadMu = mean(roadFrame);
        error[2] = error[1];
        error[1] = error[0];
        error[0] = setPoint - roadMu(0);
        /* Use trap method to compute error */
        integError += 0.5 * (error[0] + error[1]) * dt;
        derivError = (error[0] - error[1]) / (dt * 1);     
        p = Kp * error[0];
        i = Ki * integError;
        d = Kd * derivError;
        //i = 0;
        //d = 0;
        /* Note: to make this work I had to disable auto-exposure, -brightness, and -gain.
         * I also had to set ABSOLUTE mode OFF on each */
        camera.readShutter(&shutter);
        oldShutter = shutter;
        /* Update shutter w/ basic PID */
        shutterFl = (float)shutter + p + i + d;
        if(shutterFl > 500) shutterFl = 500;
        if(shutterFl < 0) shutterFl = 0;
        shutter = (unsigned int) shutterFl;
        camera.setShutter(shutter);
        printf("frame: %d\n", frameNum);
        printf("setpoint: %f\n", setPoint);
        printf("mean of road: %f\n", roadMu(0));
        printf("perr: %f\n", error[0]);
        printf("Kp: %f\n", Kp);
        printf("p: %f\n", p);
        printf("ierr: %f\n", integError);
        printf("Ki: %f\n", Ki);
        printf("i: %f\n", i);
        printf("derr: %f\n", derivError);
        printf("Kd: %f\n", Kd);
        printf("d: %f\n", d);
        printf("pid: %f\n", p + i + d);
        printf("old shutter: %u\n", oldShutter);
        printf("new shutter: %u\n", shutter);
#endif
        //return 0;
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
        else if (c == 'u') {
            undist = (undist + 1) % 2;
            cout << "Switched undistort mode to: " << undist << endl;
        }
    }

    waitKey(0);

	return 0;
}
