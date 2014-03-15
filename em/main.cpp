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
	Mat src;
	string pathToData("./em_dataset/frame%d.png");
	VideoCapture sequence(pathToData);

	BayesSeg.sigmaInit(20, 20, 20, 20);
	BayesSeg.miuInit(60, 100, 40, 210);
	BayesSeg.probPLOUInit(0.1, 0.2, 0.3, 0.4);
	
	sequence >> src;
	cvtColor(src, src, CV_RGBA2GRAY);
	// This function will be called only at the start of program or when the process is restarted.
	BayesSeg.calcProb();
	
	for (;;)
	{
		//Update EM: everything will be calculated in the "BayesSeg.EM_Bayes(src)" function.
		emtimer.start();
		BayesSeg.EM_Bayes(src);
        emtimer.stop();
        emtimer.printm();	
        
		cout <<" 	P  -  L  -  O  -  U" << endl;
		cout << "Sigma: " << BayesSeg.sigma.sigmaP << " - " << BayesSeg.sigma.sigmaL << " - " << BayesSeg.sigma.sigmaO << " - " << BayesSeg.sigma.sigmaU << endl;
		cout << "Miu:	" << BayesSeg.miu.miuP << " - " << BayesSeg.miu.miuL << " - " << BayesSeg.miu.miuO << " - " << BayesSeg.miu.miuU << endl;
		cout << "Omega: " << BayesSeg.omega.omegaP << " - " << BayesSeg.omega.omegaL << " - " << BayesSeg.omega.omegaO << " - " << BayesSeg.omega.omegaU << endl << endl;
		cout <<  "\033[22;31m" << (((BayesSeg.miu.miuO < BayesSeg.miu.miuP) && (BayesSeg.miu.miuP < BayesSeg.miu.miuL))?"true":"false") << "\e[m" << endl;
        
        
        // Display class
		/*Mat obj;
		LUT(src, BayesSeg.probPLOU_X.probO_X, obj);
		threshold(obj, obj, 0.5	, 1, CV_THRESH_BINARY);
		imshow("Object", obj);
		waitKey(1000);*/
		
		sequence >> src;
		if (src.empty())
		{
			cout << "End of Sequence" << endl;
			waitKey(0);
			break;
		}
		cvtColor(src, src, CV_RGBA2GRAY);
	}
    emtimer.aprintm();

	/*cout << "Sum omega = " << ((BayesSeg.omega.omegaP + BayesSeg.omega.omegaL + BayesSeg.omega.omegaO + BayesSeg.omega.omegaU)) << endl;
	
	cout << BayesSeg.omega.omegaP << endl;
	cout << BayesSeg.omega.omegaL << endl;
	cout << BayesSeg.omega.omegaO << endl;
	cout << BayesSeg.omega.omegaU << endl;*/
	
	//waitKey(0);
	return 0;
}
