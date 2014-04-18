#ifndef _EKF_H_
#define _EKF_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <pthread.h>
#include <string>

#define LUT_LENGTH				( 1000 )
#define RADIAN_STEP				((double) CV_PI/(2 * LUT_LENGTH) )

#define x_hat(i)				( statePost.at<float>(i) )
#define x_hat_pre(i)			( statePre.at<float>(i) )

#define	EKF_MEAS_NOISE			( 5 )
#define EKF_PROCESS_NOISE		( 10 )

#define EKF_DYNAMIC				( 6 )
#define EKF_MEAS				( 2 )
#define EKF_L					( (float)100 )

// Define for Kalman filter
#define EKF_DELAY_MS			( 30 )
#define EKF_dt					( (float) EKF_DELAY_MS/1000 )
#define EKF_SAMPLE_FREQ			( 30 )
#ifndef	EKF_dt
#define EKF_dt					( 1/ EKF_SAMPLE_FREQ )
#endif


using namespace cv;
using namespace std;



class ExtendedKalmanFilter
{
public:
	ExtendedKalmanFilter();

	~ExtendedKalmanFilter();

	Mat processNoiseCov;

	Mat measurementNoiseCov;

	Mat transitionMatrix;

	Mat measurementMatrix;

	Mat statePre;

	Mat statePost;

	Mat errorCovPost;

	Mat errorCovPre;

	Mat Kgain;

	Mat predict();

	Mat correct(Mat measure);

	void calJacobian();

private:
	static float sineLUT[];
	
	float calSin(double angleRad);

	float calCos(double angleRad);

	float calTan(double angleRad);
};

#endif /* _EKF_H_ */
