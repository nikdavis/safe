#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "bayesSeg.hpp"
#include "timer.hpp"


using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	BayesianSegmentation BayesSeg;

    timer histtimer("Histogram:			" );	
	timer btimer( 	"Calcbayes:			" );
    timer emtimer( 	"EM:				" );

	/*if (argc != 2)
	{
		cout << " Usage: main first_image" << endl;
		return -1;
	}

	Mat src = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);   // Read the file

	if (!src.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}*/
	Mat src, oldSrc;
	string pathToData("./em_dataset/frame%d.png");
	VideoCapture sequence(pathToData);

	BayesSeg.sigmaInit(20, 20, 20, 20);
	BayesSeg.miuInit(60, 100, 40, 210);
	BayesSeg.probPLOUInit(0.1, 0.2, 0.3, 0.4);
	
	sequence >> src;
	cvtColor(src, src, CV_RGBA2GRAY);
	BayesSeg.calcHistogram(&src);
	BayesSeg.calcBayesian(src);
	// In Linux, pgm image is loaded as color image????
	/*Mat src1;
	if (src.channels() == 3)
		cvtColor(src, src1, CV_RGB2GRAY);
	else if (src.channels() == 4)
		cvtColor(src, src1, CV_RGBA2GRAY);
	else
		src.copyTo(src1);*/
		
	for (int i = 0; i < 10; i++)
	{
		oldSrc = src.clone();
		sequence >> src;
		cvtColor(src, src, CV_RGBA2GRAY);
		if (src.empty())
		{
			cout << "End of Sequence" << endl;
			waitKey(0);
			break;
		}
		
		//Update EM
		emtimer.start();
		BayesSeg.EM_update(oldSrc, src);
        emtimer.stop();
		
		// Calculate histogram
		histtimer.start();
		BayesSeg.calcHistogram(&src);
		histtimer.stop();
		
		// Calculate posterior prob
		btimer.start();
		BayesSeg.Prior();
		BayesSeg.sigma.sigmaU = 20;
		BayesSeg.miu.miuU = 210;
		BayesSeg.calcBayesian(src);
		btimer.stop();
		
		cout << "Sigma: " << BayesSeg.sigma.sigmaP << " - " << BayesSeg.sigma.sigmaL << " - " << BayesSeg.sigma.sigmaO << " - " << BayesSeg.sigma.sigmaU << endl;
		cout << "Miu:	" << BayesSeg.miu.miuP << " - " << BayesSeg.miu.miuL << " - " << BayesSeg.miu.miuO << " - " << BayesSeg.miu.miuU << endl;
		cout << "Omega: " << BayesSeg.omega.omegaP << " - " << BayesSeg.omega.omegaL << " - " << BayesSeg.omega.omegaO << " - " << BayesSeg.omega.omegaU << endl << endl;
		
		histtimer.printm();
        btimer.printm();
        emtimer.printm();
	}
	histtimer.aprintm();
    btimer.aprintm();
    emtimer.aprintm();

	/*cout << "Sum omega = " << ((BayesSeg.omega.omegaP + BayesSeg.omega.omegaL + BayesSeg.omega.omegaO + BayesSeg.omega.omegaU)) << endl;
	
	cout << BayesSeg.omega.omegaP << endl;
	cout << BayesSeg.omega.omegaL << endl;
	cout << BayesSeg.omega.omegaO << endl;
	cout << BayesSeg.omega.omegaU << endl;*/
	
	//waitKey(0);
	return 0;
}
