// FILE: EKF.cpp

#include "EKF.hpp"

using namespace cv;
using namespace std;

ExtendedKalmanFilter::ExtendedKalmanFilter(void)
{
    measurementMatrix = Mat(EKF_MEAS, EKF_DYNAMIC, CV_32FC1);

    processNoiseCov = Mat(EKF_DYNAMIC, EKF_DYNAMIC, CV_32FC1);

    measurementNoiseCov = Mat(EKF_MEAS, EKF_MEAS, CV_32FC1);

    errorCovPost = Mat(EKF_DYNAMIC, EKF_DYNAMIC, CV_32FC1);

    errorCovPre = Mat(EKF_DYNAMIC, EKF_DYNAMIC, CV_32FC1);

    transitionMatrix = Mat(EKF_DYNAMIC, EKF_DYNAMIC, CV_32FC1);

    statePre = Mat(EKF_DYNAMIC, 1, CV_32FC1);

    statePost = Mat(EKF_DYNAMIC, 1, CV_32FC1);

    Kgain = Mat(EKF_DYNAMIC, EKF_MEAS, CV_32FC1);
}

ExtendedKalmanFilter::~ExtendedKalmanFilter(void)
{
}

void ExtendedKalmanFilter::calJacobian()
{
	transitionMatrix = (Mat_<float>(EKF_DYNAMIC, EKF_DYNAMIC) <<
		1, 0, -x_hat(3)*sin(x_hat(2))*EKF_dt - x_hat(5)*sin(x_hat(2))*EKF_dt*EKF_dt / 2,	cos(x_hat(2))*EKF_dt,	0,					cos(x_hat(2))*EKF_dt*EKF_dt / 2,
		0, 1, x_hat(3)*cos(x_hat(2))*EKF_dt + x_hat(5)*cos(x_hat(2))*EKF_dt*EKF_dt / 2,		sin(x_hat(2))*EKF_dt,	0,					sin(x_hat(2))*EKF_dt*EKF_dt / 2,
		0, 0, 1,																			tan(x_hat(4)) / EKF_L,	x_hat(3) / (cos(x_hat(2))*cos(x_hat(2))*EKF_L),		0,
		0, 0, 0, 1, 0, EKF_dt,
		0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 1);			
}

Mat ExtendedKalmanFilter::predict()
{
	//Project the state ahead
	x_hat_pre(0) = x_hat(0) + x_hat(3)*cos(x_hat(2))*EKF_dt + (x_hat(5)*cos(x_hat(2))*EKF_dt*EKF_dt) / 2;
	x_hat_pre(1) = x_hat(1) + x_hat(3)*sin(x_hat(2))*EKF_dt + (x_hat(5)*sin(x_hat(2))*EKF_dt*EKF_dt) / 2;
	x_hat_pre(2) = fmod(x_hat(2) + (x_hat(3)*tan(x_hat(4))*EKF_dt) / EKF_L, (float) CV_PI * 2);
	x_hat_pre(3) = x_hat(3) + x_hat(5)*EKF_dt;
	x_hat_pre(4) = fmod(x_hat(4), (float)CV_PI * 2);
	x_hat_pre(5) = x_hat(5);

	//
	(x_hat_pre(2) < 0) ? x_hat_pre(2) = x_hat_pre(2) + (float)CV_PI * 2 : x_hat_pre(2);
	(x_hat_pre(4) < 0) ? x_hat_pre(4) = x_hat_pre(4) + (float)CV_PI * 2 : x_hat_pre(4);

	// Project the error covariance ahead
	Mat transitionMatrixTranspose;
	calJacobian();
	transpose(transitionMatrix, transitionMatrixTranspose);
	errorCovPre = transitionMatrix*errorCovPost*transitionMatrixTranspose + processNoiseCov;
	return statePre;
}

Mat ExtendedKalmanFilter::correct(Mat measure)
{
	// Compute the Kalman gain
	Mat measurementMatrixTranspose;
	transpose(measurementMatrix, measurementMatrixTranspose);
	Kgain = errorCovPre*measurementMatrixTranspose*((measurementMatrix*errorCovPre*measurementMatrixTranspose + measurementNoiseCov).inv());

	// Update estimate with measurement
	statePost = statePre + Kgain*(measure - measurementMatrix*statePre);
	x_hat(2) = fmod(x_hat(2), (float)CV_PI * 2);
	x_hat(4) = fmod(x_hat(4), (float)CV_PI * 2);
	(x_hat(2) < 0) ? x_hat(2) = x_hat(2) + (float)CV_PI * 2 : x_hat(2);
	(x_hat(4) < 0) ? x_hat(4) = x_hat(4) + (float)CV_PI * 2 : x_hat(4);

	// Update the error covariance
	Mat eye = Mat(EKF_DYNAMIC, EKF_DYNAMIC, CV_32FC1);
	setIdentity(eye);
	errorCovPost = (eye - Kgain*measurementMatrix)*errorCovPre;
	return statePost;
}



