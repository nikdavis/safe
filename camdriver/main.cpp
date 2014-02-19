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

int main(int argc, char** argv)
{
    Mat frame; //current frame
    FireflyMVCamera camera;
    namedWindow(WIN_NAME, 1);                  // Create a window for display.
    if(!camera.ready()) {
        printf("Camera did not initialize properly!\n");
        return -1;
    }

    /* Start our app */
    for(;;) {
        /* Grab frame */
        camera.grabFrame(frame);
        /* Reduce noise with a kernel 3x3 */
        GaussianBlur( frame, frame, Size(3, 3), 1.0 );
        /* Canny detector */
        Canny( frame, frame, 30, 150, 3);


        //show the current frame and the fg masks
        imshow(WIN_NAME, frame);

        char c = waitKey(10);
        if( c == 27 ) break;

    }

    waitKey(0);

	return 0;
}
