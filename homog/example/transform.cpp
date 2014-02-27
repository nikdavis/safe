#include "fireflymv_camera.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>

#define CAM_HEIGHT  (480)
#define CAM_WIDTH   (640)

using namespace cv;
using namespace std;

Mat frame;
Mat frame1;


int pitch_int = 90;
int yaw_int = 90;
int roll_int = 90;
int dist_int;
int f_int;

double w;
double h; 
double w1;
double h1; 
double pitch; 
double yaw;
double roll; 
double dist; 
double f;
Point vanishing_point;

void redraw() 
{
	f = (double)(f_int+1)/10;
    pitch = (double)(pitch_int - 90) * CV_PI/180.;
	//pitch = -atan((double)(vanishing_point.y - frame.size().height/2)/(f*10));
	//cout << (double)(vanishing_point.x - frame.size().height/2)/f << endl;
	roll = (double)(roll_int  - 90) * CV_PI/180.;
	yaw = (double)(yaw_int  - 90) * CV_PI/180.;
    //dist = 1./(dist_int+1);
    //dist = dist_int+1;
    dist = dist_int-50;
    

    /*cout << "pitch = " << pitch*180/CV_PI << endl;
	cout << "yaw = " << yaw*180/CV_PI << endl;
	cout << "roll = " << roll*180/CV_PI << endl;
    cout << "dist = " << dist << endl;
    cout << "f = " << f << endl;*/

    // Projection 2D -> 3D matrix (2D homogeneous coordinate to 3D homogeneous coordinate)
    Mat A1 = (Mat_<double>(4,3) <<
        1,              0,              -w/2,
        0,              1,              -h/2,
        0,              0,              1,
        0,              0,              1);

	// Rotation matrices around the Z axis
    Mat R_roll = (Mat_<double>(4, 4) <<
        cos(roll),     -sin(roll),    	0,              0,
        sin(roll),     cos(roll),     	0,              0,
        0,              0,              1,              0,
        0,              0,              0,              1);
	
	// Rotation matrices around the Y axis
    Mat R_yaw = (Mat_<double>(4, 4) <<
        cos(yaw),       0,             	-sin(yaw),      0,
        0,              1,     			0,    			0,
        sin(yaw),       0,     			cos(yaw),     	0,
        0,              0,              0,              1);

    // Rotation matrices around the X axis
    Mat R_pitch = (Mat_<double>(4, 4) <<
        1,              0,              0,              0,
        0,              cos(pitch),     -sin(pitch),    0,
        0,              sin(pitch),     cos(pitch),     0,
        0,              0,              0,              1);

    // Translation matrix on the Z axis 
    Mat T = (Mat_<double>(4, 4) <<
        1,              0,              0,              0,
        0,              1,              0,              0,
        0,              0,              1,              dist,
        0,              0,              0,              1);

    // Camera Intrisecs matrix 3D -> 2D
    Mat A2 = (Mat_<double>(3,4) <<
        f,              0,              w/2,            0,
        0,              f,              h/2,            0,
        0,              0,              1,              0);

	Mat v_real_world = (Mat_<double>(4,1) << 
		0,				0,				1,				0);

	// matRotationTotal = matRotationX * matRotationY * matRotationZ
    Mat m = A2 * (T * (R_pitch * R_yaw * R_roll * A1));

    /*cout << "R=" << endl << R << endl;
    cout << "A1=" << endl << A1 << endl;
    cout << "R*A1=" << endl << (R*A1) << endl;
    cout << "T=" << endl << T << endl;
    cout << "T * (R * A1)=" << endl << (T * (R * A1)) << endl;
    cout << "A2=" << endl << A2 << endl;
    cout << "A2 * (T * (R * A1))=" << endl << (A2 * (T * (R * A1))) << endl;*/
    /*cout << "m=" << endl << m << endl;*/

	warpPerspective( frame, frame, m, frame.size(), INTER_CUBIC | WARP_INVERSE_MAP);
	imshow("Frame1", frame);
}

void callback(int, void* ) 
{
    redraw();
}


int main() 
{
    FireflyMVCamera camera;
	namedWindow( "Frame" ,CV_WINDOW_AUTOSIZE);		//create a window called "MyVideo"
	namedWindow( "Frame1" ,CV_WINDOW_AUTOSIZE);		//create a window called "MyVideo"
	namedWindow( "Frame2" ,CV_WINDOW_AUTOSIZE);		//create a window called "MyVideo"
	//namedWindow( "Frame3" ,CV_WINDOW_AUTOSIZE);	//create a window called "MyVideo"

	pitch_int = 86;
	createTrackbar("pitch", "Frame", &pitch_int, 180, &callback);
	createTrackbar("yaw", "Frame", &yaw_int, 180, &callback);
	createTrackbar("roll", "Frame", &roll_int, 180, &callback);
    dist_int = 55;
    createTrackbar("dist", "Frame", &dist_int, 100, &callback);
	f_int = 33;
    createTrackbar("f", "Frame", &f_int, 100, &callback);


	//cap.set(CV_CAP_PROP_POS_MSEC, 20000); //start the video at 300ms
	w = CAM_WIDTH;
	h = CAM_HEIGHT;
	//resize(frame, frame1, Size(), 1.5, 1.5);
	double fps = 30;

	cout << "Frame per seconds : " << fps << endl;

	while(1)
	{
		int ret = camera.grabFrame(frame);
		
		if (ret < 0) //if not success, break loop
		{
			cout << "Cannot read a frame from video file" << endl;
			waitKey(0);
			break;
		}
        if(frame.channels() >= 3)
		    cvtColor(frame, frame, CV_RGB2GRAY);
		imshow("Frame", frame); //show the frame in "MyVideo" window
		redraw();
		

        if(waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			//break; 
		}
	}
	return 0;
}
