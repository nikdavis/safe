#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <math.h>
#include "homography.hpp"
#include "timer.hpp"
#include "bayesSeg.hpp"

#define SINGLE_FRAME    0
#define OUTPUT_SOBEL    0
#define TUANS           1
#define STEP_THROUGH    0

using namespace cv;
using namespace std;

Mat frame;
Mat frame1;

int pitch_int = 96;
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
	Mat H, invH;
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

void callback(int, void*)
{
	redraw(OUTPUT_SOBEL);
}


int main(int argc, char** argv)
{
	
	
	if (argc < 4)
	{
		cout << "Usage: main video theta(degree) gamma(degree)" << endl;
		return -1;
	}
	//VideoCapture cap("E:/OpenCV Images/sample2.avi");	// open the video file for reading
	cout << argv[1] << endl;
	VideoCapture cap(argv[1]);	// open the video file for reading

	if (!cap.isOpened())				// if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		return -1;
	}
	
	BayesianSegmentation BayesSeg;
	timer frametimer( 	"Frame:				" );
	timer emtimer( 		"EM:				" );
	timer homogtimer( 	"Homog:				" );
	timer morphotimer( 	"Morpho:				" );

	//frame = imread("E:\\OpenCV Images\\frame1_regular.png", CV_LOAD_IMAGE_GRAYSCALE);
	//frame = imread("E:\\OpenCV Images\\frame1_regular.png", CV_LOAD_IMAGE_GRAYSCALE);
	//resize(frame, frame1, Size(), 1, 1);
	//imshow("Frame", frame);
	/*createTrackbar("pitch", "Frame", &pitch_int, 180, &callback);
	createTrackbar("yaw", "Frame", &yaw_int, 180, &callback);*/
	//redraw(OUTPUT_SOBEL);

	/*float theta = (pitch_int - 90);
	float gamma = (yaw_int - 90);*/
	float theta = ((int)atoi(argv[2]) - 90);
	float gamma = ((int)atoi(argv[3]) - 90);
	int iKernelSize = 7;
	Mat H, invH;
	Mat frame;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(iKernelSize, iKernelSize));
	
	BayesSeg.sigmaInit(20, 20, 20, UNDEF_DEFAULT_SIGMA);
	BayesSeg.miuInit(120, 200, 40, UNDEF_DEFAULT_MIU);
	BayesSeg.probPLOUInit(0.25, 0.25, 0.25, 0.25);

	BayesSeg.calcProb();
	while (1)
	{

		frametimer.start();
		bool bSuccess = cap.read(frame); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video file" << endl;
			break;
		}		
		if (frame.channels() == 3)
			cvtColor(frame, frame, CV_RGB2GRAY);
		else if (frame.channels() == 4)
			cvtColor(frame, frame, CV_RGBA2GRAY);
		equalizeHist(frame, frame);
		
		// Homography
		homogtimer.start();
		genHomogMat(H, theta, gamma);
		//generateHomogMat(H, theta, gamma);
		planeToPlaneHomog(frame, frame, H, 400);		
		homogtimer.stop();
		homogtimer.printm();
		
		//Update EM
		emtimer.start();
		BayesSeg.EM_Bayes(frame);
        emtimer.stop();
        emtimer.printm();
		
		
		/*cout <<" 	P  -  L  -  O  -  U" << endl;
		cout << "Sigma: " << BayesSeg.sigma.sigmaP << " - " << BayesSeg.sigma.sigmaL << " - " << BayesSeg.sigma.sigmaO << " - " << BayesSeg.sigma.sigmaU << endl;
		cout << "Miu:	" << BayesSeg.miu.miuP << " - " << BayesSeg.miu.miuL << " - " << BayesSeg.miu.miuO << " - " << BayesSeg.miu.miuU << endl;
		cout << "Omega: " << BayesSeg.omega.omegaP << " - " << BayesSeg.omega.omegaL << " - " << BayesSeg.omega.omegaO << " - " << BayesSeg.omega.omegaU << endl << endl;*/
		imshow("MyVideo", frame); //show the frame in "MyVideo" window
		
		morphotimer.start();
		// Display class
		Mat obj, closeObj, openObj;
		LUT(frame, BayesSeg.probPLOU_X.probO_X, obj);
		threshold(obj, obj, 0.7	, 1, CV_THRESH_BINARY);
		//imshow("Object", obj);
		

		//apply morphology filter to the image
		//morphologyEx(obj, closeObj, MORPH_CLOSE, kernel);
		morphologyEx(obj, openObj, MORPH_OPEN, kernel);
		//subtract(openObj, closeObj, obj);
		morphotimer.stop();
		morphotimer.printm();
				
		imshow("Obj morphology", openObj);
		frametimer.stop();
		frametimer.printm();
		cout << endl << endl;
		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
		
	}
	homogtimer.aprintm();
	emtimer.aprintm();
	morphotimer.aprintm();
	frametimer.aprintm();
	
	/*Point A(100, 200);
	cvtColor(frame1, frame1, CV_GRAY2RGB);
	circle(frame1, A, 3, Scalar(0, 0, 255), -1, 8);
	imshow("Frame1", frame1);
	cvtColor(frame1, frame1, CV_RGB2GRAY);
	

	Mat frame2;
	planeToPlaneHomog(frame1, frame2, invH, 840);
	Point B;
	pointHomogToPointOrig(invH, A, B);
	cout << "x: " << B.x << " - y: " << B.y << endl;
	cvtColor(frame1, frame1, CV_GRAY2RGB);
	circle(frame2, B, 3, Scalar(255, 0, 0), -1, 8);
	imshow("Frame2", frame2);*/

	
	waitKey(0);
	return 1;
}
