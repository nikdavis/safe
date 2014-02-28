#include "fireflymv_camera.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <math.h>

#define HAVE_IPP 1

#define SINGLE_FRAME    0
#define OUTPUT_SOBEL    0
#define TUANS           1
#define STEP_THROUGH    0
using namespace cv;
using namespace std;



Mat frame;
Mat frame1;

/* Good ones (original tuan)
int pitch_int = 8;
int yaw_int = 90;
int roll_int = 90;
int dist_int = 282;
int f_int = 70;
*/

/* for mine 
int pitch_int = 96;
int yaw_int = 88;
int roll_int = 90;
int tx_int = 879;
int ty_int = 527;
int tz_int = 5721;
int f_int = 50;
*/



int pitch_int = 96;
int yaw_int = 90;
int roll_int = 90;
int tx_int = -427 + 500;
int ty_int = 542;
int tz_int = 4485;
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
    Mat sobelx, sobely, mask;
	f = (double)(f_int) + 500;
    pitch = (double)(pitch_int - 180) * CV_PI/180.;
	//pitch = -atan((double)(vanishing_point.y - frame.size().height/2)/(f*10));
	//cout << (double)(vanishing_point.x - frame.size().height/2)/f << endl;
	roll = (double)(roll_int  - 90) * CV_PI/180.;
	yaw = (double)(yaw_int  - 90) * CV_PI/180.;
    //dist = 1./(dist_int+1);
    //dist = dist_int+1;
    //dist = dist_int;
    tx = tx_int - 500;
    ty = ty_int - 500;
    tz = tz_int - 5000;
    

    /*cout << "pitch = " << pitch*180/CV_PI << endl;
	cout << "yaw = " << yaw*180/CV_PI << endl;
	cout << "roll = " << roll*180/CV_PI << endl;
    cout << "dist = " << dist << endl;
    cout << "f = " << f << endl;*/
#if TUANS
    // Projection 2D -> 3D matrix
    Mat A1 = (Mat_<double>(4,3) <<
        1, 0, -w/2,
        0, 1, -h/2,
        0, 0,    0,
        0, 0,    1);

    // Rotation matrices around the X,Y,Z axis
    Mat RX = (Mat_<double>(4, 4) <<
        1,          0,           0, 0,
        0, cos(pitch), -sin(pitch), 0,
        0, sin(pitch),  cos(pitch), 0,
        0,          0,           0, 1);

    Mat RY = (Mat_<double>(4, 4) <<
        cos(yaw), 0, -sin(yaw), 0,
        0, 1,          0, 0,
        sin(yaw), 0,  cos(yaw), 0,
        0, 0,          0, 1);

    Mat RZ = (Mat_<double>(4, 4) <<
        cos(roll), -sin(roll), 0, 0,
        sin(roll),  cos(roll), 0, 0,
        0,          0,           1, 0,
        0,          0,           0, 1);

    // Composed rotation matrix with (RX,RY,RZ)
    Mat R = RX * RY * RZ;

    // Translation matrix on the Z axis change dist will change the height
    Mat T = (Mat_<double>(4, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, tz,
        0, 0, 0, 1);

    // Camera Intrisecs matrix 3D -> 2D
    Mat A2 = (Mat_<double>(3,4) <<
        f, 0, w/2, 0,
        0, f, h/2, 0,
        0, 0,   1, 0);

	// matRotationTotal = matRotationX * matRotationY * matRotationZ
/* Mult K after downsize 
    Mat P(4, 4, R_pitch.type());
    Mat mlarge(4, 3, R_pitch.type());
    Mat m(3, 3, R_pitch.type());
    P = (T * R_pitch * R_yaw * R_roll);
    P.col(0).copyTo(mlarge.col(0));
    P.col(2).copyTo(mlarge.col(1));
    P.col(3).copyTo(mlarge.col(2));
    m = mlarge( Range(0,3), Range(0,3) );
    m = A2 * m;
    m = m.inv();
*/

    Mat m = A2 * (T * (R * A1));
#else

	Mat Tvec = (Mat_<double>(3,1) << 
		tx,
        ty,
        tz);

    Mat R_eul = (Mat_<double>(3,1) <<
        pitch,
        yaw,
        roll);

    Mat R;
    Rodrigues(R_eul, R);
/*
    Mat RT(4, 4, R.type()); // RT is 4x4
    RT( Range(0,3), Range(0,3) ) = R * 1; // copies R into R|T
    RT( Range(0,3), Range(3,4) ) = Tvec * 1; // copies T into R|T
    double *p = RT.ptr<double>(3);
    p[0] = p[1] = p[2] = 0; p[3] = 1;
    //RTinv = RTinv.inv();
*/

    Mat RT(4, 4, R.type()); // RT is 4x4
    RT( Range(0,3), Range(0,3) ) = R * 1; // copies R into R|T
    RT( Range(0,3), Range(3,4) ) = Tvec * 1; // copies T into R|T
    double *p = RT.ptr<double>(3);
    p[0] = p[1] = p[2] = 0; p[3] = 1;
    //RTinv = RTinv.inv();
    // Camera Intrisecs matrix 3D -> 2D
    Mat K = (Mat_<double>(3, 4) <<
        f,              0,              w/2,            0,
        0,              f,              h/2,            0,
        0,              0,              1,              0);

    Mat P(3, 4, R.type());
    Mat m(3, 3, R.type());
    P = K * RT;

    P.col(0).copyTo(m.col(0));
    P.col(2).copyTo(m.col(1));
    P.col(3).copyTo(m.col(2));
    m = m.inv();

    cout << "x, y, z: " << tx << " " << ty << " " << tz << " " << endl;
#endif


    /*cout << "R=" << endl << R << endl;
    cout << "A1=" << endl << A1 << endl;
    cout << "R*A1=" << endl << (R*A1) << endl;
    cout << "T=" << endl << T << endl;
    cout << "T * (R * A1)=" << endl << (T * (R * A1)) << endl;
    cout << "A2=" << endl << A2 << endl;
    cout << "A2 * (T * (R * A1))=" << endl << (A2 * (T * (R * A1))) << endl;*/
    /*cout << "m=" << endl << m << endl;*/

	//warpPerspective( frame, frame1, m, Size(854, 480), INTER_CUBIC | WARP_INVERSE_MAP);
    m = m.inv();
	warpPerspective( frame, frame1, m, frame.size());
    if(sobel) {
        Sobel(frame1, sobelx, frame1.depth(), 1, 0, 5);
        Sobel(frame1, sobely, frame1.depth(), 0, 1, 5);
        mask = sobelx + sobely;
        threshold(mask, mask, 180, 255, THRESH_BINARY);
        erode(mask, mask, Mat());
        dilate(mask, mask, Mat(), Point(-1,-1), 5);
        bitwise_and(frame1, ~mask, frame1);
    }
	imshow("Frame1", frame1);

}

void callback(int, void* ) 
{
    redraw(OUTPUT_SOBEL);
}


int main() 
{
    FireflyMVCamera camera;
	namedWindow( "Frame" ,CV_WINDOW_AUTOSIZE);		//create a window called "MyVideo"
	namedWindow( "Frame1" ,CV_WINDOW_AUTOSIZE);		//create a window called "MyVideo"
	//namedWindow( "Frame3" ,CV_WINDOW_AUTOSIZE);	//create a window called "MyVideo"

	createTrackbar("pitch", "Frame", &pitch_int, 360, &callback);
	createTrackbar("yaw", "Frame", &yaw_int, 180, &callback);
	createTrackbar("roll", "Frame", &roll_int, 180, &callback);
    createTrackbar("tx", "Frame", &tx_int, 1000, &callback);
    createTrackbar("ty", "Frame", &ty_int, 1000, &callback);
    createTrackbar("tz", "Frame", &tz_int, 10000, &callback);
    createTrackbar("f", "Frame", &f_int, 500, &callback);
	VideoCapture cap("../../../MVI_5248_bw_480p.mp4"); // open the video file for reading

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
