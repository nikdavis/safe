#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <math.h>
#include "homography.hpp"
#include "timer.hpp"
#include "bayesSeg.hpp"
#include "carTracking.hpp"
#include "helpFn.hpp"
#include "EKF.hpp"


#define SINGLE_FRAME    ( 0 )
#define OUTPUT_SOBEL    ( 0 )
#define TUANS           ( 1 )
#define STEP_THROUGH    ( 0 )

using namespace cv;
using namespace std;
RNG rng(12345);

Mat frame, frame1;
int boxSize[17] = {15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95 };

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
	CarTracking carTrack;
	
	timer frametimer( 	"Frame:				" );
	timer emtimer( 		"EM:				" );
	timer homogtimer( 	"Homog:				" );
	timer morphotimer( 	"Morpho:				" );
	timer blobtimer( 	"Blob detection:			" );

	/*float theta = (pitch_int - 90);
	float gamma = (yaw_int - 90);*/
	float theta = (float)(atoi(argv[2]) - 90);
	float gamma = (float)(atoi(argv[3]) - 90);
	int iKernelSize = 7;
	Mat H, invH;
	Mat frame;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(iKernelSize, iKernelSize));
	Mat kernel1 = Mat(13, 3, CV_8U);
	kernel1.setTo(Scalar(1));
	
	BayesSeg.sigmaInit(20, 20, 20, UNDEF_DEFAULT_SIGMA);
	BayesSeg.miuInit(120, 200, 40, UNDEF_DEFAULT_MIU);
	BayesSeg.probPLOUInit(0.25, 0.25, 0.25, 0.25);

	BayesSeg.calcProb();
	while (1)
	{
		frametimer.start();
		cap.read(frame); // read a new frame from video

		if (frame.empty()) //if not success, break loop
		{
			cout << "Cannot read a frame from video file" << endl;
			break;
		}		
		Mat orig = frame.clone();
		
		if (frame.channels() == 3)
			cvtColor(frame, frame, CV_RGB2GRAY);
		else if (frame.channels() == 4)
			cvtColor(frame, frame, CV_RGBA2GRAY);
			
		equalizeHist(frame, frame);
		
		// Homography
		homogtimer.start();
		genHomogMat(&H, &invH, theta, gamma);
		//generateHomogMat(H, theta, gamma);
		planeToPlaneHomog(frame, frame, H, 400);		
		homogtimer.stop();
		//homogtimer.printm();
		
		//Update EM
		emtimer.start();
		BayesSeg.EM_Bayes(frame);
        emtimer.stop();
        //emtimer.printm();
		
		imshow("MyVideo", frame); //show the frame in "MyVideo" window
		

		// Segmentation class: object
		Mat obj;
		morphotimer.start();		
		BayesSeg.classSeg(&frame, &obj, BayesSeg.OBJ);
		GaussianBlur(obj, obj, Size(15, 15), 10, 10);
		threshold(obj, obj, 100, 255, CV_THRESH_BINARY);
		
		morphotimer.stop();
		//morphotimer.printm();
		
		// blob detection
		blobtimer.start();
		
		// Blob contour bounding box
		carTrack.findBoundContourBox(&obj);
		
		// rotated rectangle
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		Mat dst = obj.clone();
		cvtColor(dst, dst, CV_GRAY2RGB);
		for (unsigned int i = 0; i < carTrack.boundRect.size(); i++) {
			rectangle(dst, carTrack.boundRect[i], Scalar(255, 0, 0), 2, 4, 0);
		}
		
		// Blob detection
		carTrack.detect_filter(&obj);	

		char zBuffer[35];
		for (unsigned int i = 0; i < carTrack.objCands.size(); i++)
		{
			circle(dst, carTrack.objCands[i].Pos, 3, Scalar(0, 0, 255), -1);
			if (carTrack.objCands[i].inFilter)
			{
				Point filterPos;
				circle(dst, carTrack.objCands[i].Pos, 3, Scalar(255, 0, 0), -1);
				pointHomogToPointOrig(&invH, &carTrack.objCands[i].filterPos, &filterPos);
				circle(dst, carTrack.objCands[i].filterPos, 2, Scalar(0, 255, 0), -1);
				circle(orig, filterPos, 2, Scalar(0, 0, 255), -1);
				
				Point2f direction;
				carTrack.calAngle(&carTrack.objCands[i].filterPos, &obj, &direction);

				/*cout << "Vx: " << direction.x << "-" << carTrack.objCands[i].direction(0) << endl;
				cout << "Vy: " << direction.y << "-" << carTrack.objCands[i].direction(1) << endl;
				cout << "angle: " << atan(carTrack.objCands[i].direction(0) / carTrack.objCands[i].direction(1)) << endl;
				cout << SQUARE_ERROR(direction.x, direction.y, carTrack.objCands[i].direction(0), carTrack.objCands[i].direction(1)) << endl;*/

				Point p1(carTrack.objCands[i].filterPos.x, carTrack.objCands[i].filterPos.y);
				Point p2(carTrack.objCands[i].filterPos.x + 1000 * carTrack.objCands[i].direction(0),
						 carTrack.objCands[i].filterPos.y + 1000 * carTrack.objCands[i].direction(1));

				/*cout << p1.x << "-" << p1.y << endl;
				cout << p2.x << "-" << p2.y << endl;*/
				line(dst, p1, p2, Scalar(255, 255, 0), 5);

				Point cvtP;
				carTrack.cvtCoord(&p1, &cvtP, &obj);
				Point2f feetPos(cvtP.x*PX_FEET_SCALE, cvtP.y*PX_FEET_SCALE);

				//float velocity = carTrack.objCands[i].EKF.statePost.at<float>(3) * PX_FEET_SCALE * SAMPLE_FREQ * (0.682f);

				cout << feetPos.x << "x" << feetPos.y << endl;
				//cout << "velocity: " << velocity << endl;


				float a = 8.0f;
				float b = 25.0f;

				if ((powf(feetPos.x, 2) / (a*a) + powf(feetPos.y, 2) / (b*b)) < 1)
				{
					if (abs(carTrack.objCands[i].direction(0)*direction.x + carTrack.objCands[i].direction(1)*direction.y) > 0.5)
						cout << "\033[22;31mALARM\e[m" << endl;	
				}
			}
		}
			
		blobtimer.stop();
		//blobtimer.printm();
		
		imshow("dst", dst);
		imshow("Original", orig);
		
		frametimer.stop();
		//frametimer.printm();
		cout << endl << endl;
		if (waitKey(DELAY_MS) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}
	homogtimer.aprintm();
	emtimer.aprintm();
	morphotimer.aprintm();
	blobtimer.aprintm();
	frametimer.aprintm();
	
	waitKey(0);
	return 1;
}
