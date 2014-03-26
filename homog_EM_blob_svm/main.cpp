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
#include "CarSVM.hpp"
#include "svm.hpp"

#define SINGLE_FRAME    ( 0 )
#define OUTPUT_SOBEL    ( 0 )
#define TUANS           ( 1 )
#define STEP_THROUGH    ( 0 )

using namespace cv;
using namespace std;

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

/*void redraw(int sobel = 0)
{
	float theta = (pitch_int - 90);
	float gamma = (yaw_int - 90);
	Mat H, invH;
	generateHomogMat(H, theta, gamma);
	planeToPlaneHomog(frame, frame1, H, 400);
	imshow("Frame1", frame1); //show birds-eye view
}


// Sobel info
if(sobel) {
Sobel(frame1, sobelx, frame1.depth(), 1, 0, 5);
Sobel(frame1, sobely, frame1.depth(), 0, 1, 5);
mask = sobelx + sobely;
threshold(mask, mask, 180, 255, THRESH_BINARY);
erode(mask, mask, Mat());
dilate(mask, mask, Mat(), Point(-1,-1), 5);
bitwise_and(frame1, ~mask, frame1);
}


void callback(int, void*)
{
	redraw(OUTPUT_SOBEL);
}*/


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
	CarSVM carsvm;
	
	timer frametimer( 	"Frame:				" );
	timer emtimer( 		"EM:				" );
	timer homogtimer( 	"Homog:				" );
	timer morphotimer( 	"Morpho:				" );
	timer blobtimer( 	"Blob detection:			" );
	
	
	// Load car model for SVM predict
	const svm_model *carModel = new svm_model[1];
	carModel = svm_load_model("car.model");
	cout << "number of class: " << carModel->nr_class << endl;
	cout << "Kernel type: " << carModel->param.kernel_type << endl;
	cout << "SVM type: " << svm_get_svm_type(carModel) << endl << endl;

	/*float theta = (pitch_int - 90);
	float gamma = (yaw_int - 90);*/
	float theta = ((int)atoi(argv[2]) - 90);
	float gamma = ((int)atoi(argv[3]) - 90);
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
		

		// Segmentation class: object
		Mat obj;
		morphotimer.start();		
		BayesSeg.classSeg(&frame, &obj, BayesSeg.OBJ);
		morphotimer.stop();
		morphotimer.printm();
		
		// Blob contour bounding box
		//carTrack.findBoundContourBox(&obj);
		
		// Blob detection
		carTrack.detect_filter(&obj);
				
		
		// Blob detection
		blobtimer.start();
		Mat objTmp = obj.clone();
		cvtColor(objTmp, objTmp, CV_GRAY2RGB);
		char zBuffer[35];
		for (unsigned int i = 0; i < carTrack.objCands.size(); i++)
		{
			//circle(obj, carTrack.objCands[i].Pt, 3, Scalar(0, 0, 255), -1);
			if (carTrack.objCands[i].inFilter)
			{
				Mat highProbCarImg;
				Rect bigbox, smallbox;
				Point filterPos;

				
				circle(objTmp, carTrack.objCands[i].Pos, 3, Scalar(255, 0, 0), -1);
				circle(objTmp, carTrack.objCands[i].filterPos, 3, Scalar(0, 255, 0), -1);
				pointHomogToPointOrig(&invH, &carTrack.objCands[i].filterPos, &filterPos);
				circle(orig, filterPos, 2, Scalar(0, 0, 255), -1);
				carTrack.cropBoundObj(&orig, &highProbCarImg, &invH, &bigbox, i);
				//imshow("highProbCarImg", highProbCarImg);
				rectangle(orig, bigbox, Scalar(0, 255, 0), 3);
				// copy the text to the "zBuffer"
				//_snprintf_s(zBuffer, 35, "l: %d", cvRound(bigbox.height/1.5));

				//put the text in the "zBuffer" to the "dst" image
				//putText(orig, zBuffer, bigbox.tl(), CV_FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 0, 0), 2);
				if (carTrack.carSVMpredict(&highProbCarImg, &smallbox, carsvm.POS, carModel, i))
				{
					//cout << "car detected" << endl;
					Rect carBox(bigbox.tl().x + smallbox.tl().x, bigbox.tl().y + smallbox.tl().y, smallbox.width, smallbox.height);
					rectangle(orig, carBox, Scalar(0, 0, 255), 3);
					/*Mat grayOrig;
					cvtColor(orig, grayOrig, CV_RGB2GRAY);
					if (saveBoxImg(&grayOrig, &carBox, "E:\\pos32\\no_car", &fileNum))
						fileNum++;*/

					continue;
				}
			}
		}
	 	
		blobtimer.stop();
		blobtimer.printm();
		
		//imshow("Obj morphology", obj);
		imshow("Object", objTmp);
		imshow("Original", orig);
		
		frametimer.stop();
		frametimer.printm();
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
