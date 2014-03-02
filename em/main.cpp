#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "bayesSeg.hpp"
#include "timer.hpp"


using namespace std;
using namespace cv;


void callBack(int threshValue, void *userData)
{
	Mat src = *(static_cast<Mat*>(userData));
	Mat img;
	src.copyTo(img);
	Mat objMask;
	threshold(img, objMask, threshValue, 255,  THRESH_BINARY);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));

	//apply morphology filter to the image
	morphologyEx(objMask, objMask, MORPH_OPEN, kernel);
	imshow("My Window", objMask);
}



int main(int argc, char** argv)
{
	BayesianSegmentation BayesSeg;

    timer histtimer("Histogram:			" );	
	timer btimer( 	"Calcbayes:			" );
    timer emtimer( 	"EM:				" );
	timer objtimer( "Object seg:			" );

	if (argc != 2)
	{
		cout << " Usage: main a_sample_image" << endl;
		return -1;
	}

	Mat src = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);   // Read the file

	if (!src.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}
	
	// Create a window
	int threshValue = 40;
	namedWindow("My Window", 1);
	createTrackbar("Threshold", "My Window", &threshValue, 255, callBack, &src);

	BayesSeg.sigmaInit(10, 10, 10, 20);
	BayesSeg.miuInit(50, 130, 10, 10);
	BayesSeg.probPLOUInit(0.2, 0.25, 0.25, 0.3);
	
	// In Linux, pgm image is loaded as color image????
	Mat src1;
	if (src.channels() == 3)
		cvtColor(src, src1, CV_RGB2GRAY);
	else if (src.channels() == 4)
		cvtColor(src, src1, CV_RGBA2GRAY);
	else
		src.copyTo(src1);
		
	for (int i = 0; i < 10; i++)
	{
		histtimer.start();
		BayesSeg.calcHistogram(&src1);
		histtimer.stop();
		btimer.start();
		BayesSeg.calcBayesian(src1);
		btimer.stop();
		emtimer.start();
		BayesSeg.EM_update(src1);
        emtimer.stop();
		BayesSeg.Prior();
		objtimer.start();
		BayesSeg.ObjectSeg(src1, 40);
		objtimer.stop();
		histtimer.printm();
        btimer.printm();
        emtimer.printm();
   		objtimer.printm();
	}
	histtimer.aprintm();
    btimer.aprintm();
    emtimer.aprintm();
	objtimer.aprintm();

	/*cout << "Sum omega = " << ((BayesSeg.omega.omegaP + BayesSeg.omega.omegaL + BayesSeg.omega.omegaO + BayesSeg.omega.omegaU)) << endl;
	
	cout << BayesSeg.omega.omegaP << endl;
	cout << BayesSeg.omega.omegaL << endl;
	cout << BayesSeg.omega.omegaO << endl;
	cout << BayesSeg.omega.omegaU << endl;*/
	
	waitKey(0);
	return 0;
}
