/*
 * main.cpp
 *
 *  Created on: Feb 27, 2014
 *      Author: nik
 */

#include <stdlib.h>
#include <iostream> // for standard I/O
#include <string>   // for strings
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write
#include "fireflymv_camera.hpp"

using namespace std;
using namespace cv;

#define INPUT_FPS		30
#define CAM_WIDTH		640
#define CAM_HEIGHT		480
#define FOURCC			CV_FOURCC('F','M','P','4')


int main(int argc, char *argv[])
{
    FireflyMVCamera camera;

    /* Parse arguments */
    if(argc != 3) {
    	cout << "Need to call ./<exec> <outputname>.avi <secondsToRecord>" << endl;
    	return -1;
    }
    char * outName = argv[1];
    int secondsToCapture = atoi(argv[2]);

    /* Check camera status */
    if (!camera.ready())
    {
        cout  << "Could not open FireflyMV camera, exiting... " << endl;
        return -1;
    }

    /* Get output video ready, will overwrite other videos */
    VideoWriter outputVideo;
    Size camSize = Size(CAM_WIDTH, CAM_HEIGHT);
    /* FMP4 is the default codec of mencoder */
    outputVideo.open(outName, FOURCC, INPUT_FPS, camSize, false);
    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << endl;
        return -1;
    }

    /* Print some fancy info */
    int ex = FOURCC;
    char EXT[] = {(char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};
    cout << "Output frame resolution: Width=" << camSize.width << "  Height=" << camSize.height
         << " of nr#: " << INPUT_FPS << endl;
    cout << "Output codec type: " << EXT << endl;
    cout << "Output name: " << outName << endl;

    /* Start capturing frames */
    Mat src, res;
    int frameCount = 0;
    int seconds = 0;
    for(;;) //Show the image captured in the window and repeat
    {
    	camera.grabFrame(src);
    	outputVideo << src;
    	frameCount++;
    	if(frameCount % INPUT_FPS == 0) {
    		seconds++;
    		if(seconds % 10 == 0)
    			cout << seconds << " second(s)" << endl;
    		if(seconds >= secondsToCapture)
    			break;
    	}
    }

    cout << "Successfully captured " << secondsToCapture << " seconds(s)!" << endl;
    return 0;
}
