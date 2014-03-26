
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <pthread.h>

#include "bayesSeg.hpp"
#include "timer.hpp"


using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
	BayesianSegmentation BayesSeg;

    timer btimer( "Calcbayes:         " );
    timer emtimer( "EM:              " );
	
	if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

	Mat src = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);   // Read the file

    if(! src.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
	Mat src1;
	src.convertTo(src1, CV_32F);
	cout << src.cols << "x" << src.rows << endl;
	//Mat p = BayesSeg.calProb(src, 10, 50);
	//cout << p.row(0) << endl;
	BayesSeg.sigmaInit(10, 10, 10, 20);
	BayesSeg.miuInit(50, 130, 10, 10);
	BayesSeg.probPLOUInit(0.25, 0.25, 0.25, 0.25);




	for (int i = 0; i < 10; i++)
	{
		btimer.start();
		BayesSeg.calBayesian(src1);
		btimer.stop();
		emtimer.start();
		BayesSeg.EM_update(src1);
        emtimer.stop();

        btimer.printm();
        emtimer.printm();
	}
    btimer.aprintm();
    emtimer.aprintm();

	cout << BayesSeg.omega.omegaP << endl;
	cout << BayesSeg.omega.omegaL << endl;
	cout << BayesSeg.omega.omegaO << endl;
	cout << BayesSeg.omega.omegaU << endl;


	/*LARGE_INTEGER frequency;
	LARGE_INTEGER start;
	LARGE_INTEGER end;
	double interval;

	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);

	BayesSeg.calBayesian(src1);
	BayesSeg.EM_update(src1);
	cout << BayesSeg.omega.omegaP << endl;
	cout << BayesSeg.omega.omegaL << endl;
	cout << BayesSeg.omega.omegaO << endl;
	cout << BayesSeg.omega.omegaU << endl;

	QueryPerformanceCounter(&end);
	interval = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;

	//printf("%f\n", interval);
	cout << "Computation time = " << interval << endl;*/
	
	waitKey(0);
}
