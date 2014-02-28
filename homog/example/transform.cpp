#include "fireflymv_camera.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <math.h>
#include "homography.hpp"

#define SINGLE_FRAME    0
#define OUTPUT_SOBEL    0
#define TUANS           1
#define STEP_THROUGH    0
using namespace cv;
using namespace std;



Mat frame;
Mat frame1;


int pitch_int = 90;
int yaw_int = 90;
int tz_int = 5300;
int f_int = 50;


double w;
double h; 
double w1;
double h1; 
double pitch; 
double yaw;
double roll; 
//double dist;
double tx;
double ty;
double tz;
double f;
Point vanishing_point;

void redraw(int sobel = 0)
{
	float theta = (pitch_int - 90);
	float gamma = (yaw_int - 90);
	Mat H;
	generateHomogMat(H, theta, gamma);
	planeToPlaneHomog(frame, frame1, H, 400);
    imshow("Frame1", frame1); //show birds-eye view
}


/* Sobel info
    if(sobel) {
        Sobel(frame1, sobelx, frame1.depth(), 1, 0, 5);
        Sobel(frame1, sobely, frame1.depth(), 0, 1, 5);
        mask = sobelx + sobely;
        threshold(mask, mask, 180, 255, THRESH_BINARY);
        erode(mask, mask, Mat());
        dilate(mask, mask, Mat(), Point(-1,-1), 5);
        bitwise_and(frame1, ~mask, frame1);
    }
*/

void callback(int, void* ) 
{
    redraw(OUTPUT_SOBEL);
}


int main() 
{
    FireflyMVCamera camera;
	namedWindow( "Frame" ,CV_WINDOW_AUTOSIZE);
	namedWindow( "Frame1" ,CV_WINDOW_AUTOSIZE);

	createTrackbar("pitch", "Frame", &pitch_int, 180, &callback);
	createTrackbar("yaw", "Frame", &yaw_int, 180, &callback);
    //createTrackbar("tz", "Frame", &tz_int, 10000, &callback);
    //createTrackbar("f", "Frame", &f_int, 500, &callback);
	VideoCapture cap("../../MVI_5248_bw_480p.mp4"); // open the video file for reading

    if(!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		while (waitKey(0) != 27);
		return 0;
	}

	w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    cout << "CamWidth: " << w << endl;
    cout << "CamHeight: " << h << endl;
	//resize(frame, frame1, Size(), 1.5, 1.5);
	double fps = 30;
#if !SINGLE_FRAME
    while(1) {
        bool ret = cap.read(frame);

        if (!ret) //if not success, break loop
        {
            cap.set(CV_CAP_PROP_POS_AVI_RATIO , 0);
            ret = cap.read(frame);
                if(!ret) {
                    cout << "Video invalid, exiting!" << endl;
                    break;
                } else {
                    cout << "No more frames, restarting video cap..." << endl;
                }
            continue;
        }
        if(frame.channels() >= 3)
            cvtColor(frame, frame, CV_RGB2GRAY);
        frame1 = frame.clone();
        imshow("Frame", frame); //show the frame in "MyVideo" window
        redraw(OUTPUT_SOBEL);

#else

    bool ret = cap.read(frame);
    redraw();
    while(1) {
        int histSize = 256;
        float range[] = {0, 256};
        const float*histRange = {range};
        Mat hist;
        //calcHist(&frame1, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, true);
        imshow("Frame", frame);
#endif
        if(waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
	        cout << "esc key is pressed by user" << endl;
	        break;
        }
    }
	return 0;
}
