#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include "bayesSeg.hpp"
#include <iostream>
#include <pthread.h>
#include <string>
#include <sstream> 
#include <fstream>

using namespace cv;
using namespace std;

Mat BayesianSegmentation::GRAY_RANGE = (Mat_<float>(256, 1) <<	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
																16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
																32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
																48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
																64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
																80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
																96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
																112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
																128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
																144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
																160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
																176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
																192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
																208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
																224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
																240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255);

void BayesianSegmentation::writeCSV(Mat data, string fileName)
{
	ofstream fout(fileName);
	for (int r = 0; r < data.rows; r++)
	{
		for (int c = 0; c < data.cols; c++)
		{
			fout << data.at<float>(r, c) << ';';
		}
		fout << endl;
	}
	fout.close();
}

void BayesianSegmentation::calcHistogram(Mat* img)
{
	N = img->size().area();

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true; 
	bool accumulate = false;

	/// Compute the histograms:
	calcHist(img, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	//writeCSV(hist, "hist.csv");
}

void* BayesianSegmentation::calcSingleProb(void* arg)
{
	PassArg* args = (static_cast<PassArg*>(arg));
	double p;
	//out = (1/sigma*sqrt(2*pi))*exp(-(X-miu).^2/(2*sigma^2));
	// P_CONST: 1/sqrt(2*pi)
	Mat subMat, powMat, divMat, expMat, finalMat, outMat;
	subtract(GRAY_RANGE, args->miu, subMat);				// X - miu
	pow(subMat, 2, powMat);									// (X - miu).^2
	divide(powMat, 2 * args->sigma * args->sigma, divMat);	// (X-miu).^2/(2*sigma^2)
	exp(-divMat, expMat);									// exp(-(X-miu).^2/(2*sigma^2));
	p = P_CONST / args->sigma;								// (1/(sigma*sqrt(2*pi)))
	finalMat = expMat*p;									// (1/sigma*sqrt(2*pi))*exp(-(X-miu).^2/sigma.^2);
	args->probX_PLOU = finalMat*args->probPLOU;

	return nullptr;
}

void BayesianSegmentation::calcProbThread(Mat img)
{
	pthread_t thread1, thread2, thread3, thread4;
	PassArg passArg1, passArg2, passArg3, passArg4;

	passArg1.src = img;
	passArg1.miu = miu.miuP;
	passArg1.sigma = sigma.sigmaP;
	passArg1.probPLOU = probPLOU.probP;
	pthread_create(&thread1, NULL, calcSingleProb, static_cast<void*>(&passArg1));

	passArg2.src = img;
	passArg2.miu = miu.miuL;
	passArg2.sigma = sigma.sigmaL;
	passArg2.probPLOU = probPLOU.probL;
	pthread_create(&thread2, NULL, calcSingleProb, static_cast<void*>(&passArg2));

	passArg3.src = img;
	passArg3.miu = miu.miuO;
	passArg3.sigma = sigma.sigmaO;
	passArg3.probPLOU = probPLOU.probO;
	pthread_create(&thread3, NULL, calcSingleProb, static_cast<void*>(&passArg3));

	passArg4.src = img;
	passArg4.miu = miu.miuU;
	passArg4.sigma = sigma.sigmaU;
	passArg4.probPLOU = probPLOU.probU;
	pthread_create(&thread4, NULL, calcSingleProb, static_cast<void*>(&passArg4));

	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
	pthread_join(thread3, NULL);
	pthread_join(thread4, NULL);

	probX_PLOU.probX_P = passArg1.probX_PLOU;
	probX_PLOU.probX_L = passArg2.probX_PLOU;
	probX_PLOU.probX_O = passArg3.probX_PLOU;
	probX_PLOU.probX_U = passArg4.probX_PLOU;
}

void BayesianSegmentation::calcProb(Mat img)
{
	double p;
	Mat subMat, powMat, divMat, expMat, finalMat;
	//out = (1/sigma*sqrt(2*pi))*exp(-(X-miu).^2/(2*sigma^2));
	// P_CONST: 1/sqrt(2*pi)

	// P
	subtract(GRAY_RANGE, miu.miuP, subMat);			// X - miu
	pow(subMat, 2, powMat);							// (X - miu).^2
	divide(powMat, 2 * sigma.sigmaP*sigma.sigmaP, divMat);	// (X-miu).^2/(2*sigma^2)
	exp(-divMat, expMat);							// exp(-(X-miu).^2/(2*sigma^2));
	p = P_CONST / sigma.sigmaP;						// (1/(sigma*sqrt(2*pi)))
	probX_PLOU.probX_P = expMat*p;							// (1/sigma*sqrt(2*pi))*exp(-(X-miu).^2/sigma.^2);
	//LUT(img, finalMat, probX_PLOU.probX_P);

	// L
	subtract(GRAY_RANGE, miu.miuL, subMat);			// X - miu
	pow(subMat, 2, powMat);							// (X - miu).^2
	divide(powMat, 2 * sigma.sigmaL*sigma.sigmaL, divMat);	// (X-miu).^2/(2*sigma^2)
	exp(-divMat, expMat);							// exp(-(X-miu).^2/(2*sigma^2));
	p = P_CONST / sigma.sigmaL;						// (1/(sigma*sqrt(2*pi)))
	probX_PLOU.probX_L = expMat*p;							// (1/sigma*sqrt(2*pi))*exp(-(X-miu).^2/sigma.^2);
	//LUT(img, finalMat, probX_PLOU.probX_L);


	// O
	subtract(GRAY_RANGE, miu.miuO, subMat);			// X - miu
	pow(subMat, 2, powMat);							// (X - miu).^2
	divide(powMat, 2 * sigma.sigmaO*sigma.sigmaO, divMat);	// (X-miu).^2/(2*sigma^2)
	exp(-divMat, expMat);							// exp(-(X-miu).^2/(2*sigma^2));
	p = P_CONST / sigma.sigmaO;						// (1/(sigma*sqrt(2*pi)))
	probX_PLOU.probX_O = expMat*p;							// (1/sigma*sqrt(2*pi))*exp(-(X-miu).^2/sigma.^2);
	//LUT(img, finalMat, probX_PLOU.probX_O);


	// U
	subtract(GRAY_RANGE, miu.miuU, subMat);			// X - miu
	pow(subMat, 2, powMat);							// (X - miu).^2
	divide(powMat, 2 * sigma.sigmaU*sigma.sigmaU, divMat);	// (X-miu).^2/(2*sigma^2)
	exp(-divMat, expMat);							// exp(-(X-miu).^2/(2*sigma^2));
	p = P_CONST / sigma.sigmaU;						// (1/(sigma*sqrt(2*pi)))
	probX_PLOU.probX_U = expMat*p;							// (1/sigma*sqrt(2*pi))*exp(-(X-miu).^2/sigma.^2);
	//LUT(img, finalMat, probX_PLOU.probX_U);
}

void* BayesianSegmentation::calSingleProbPLOU_X(void* arg)
{
	PassArg* args = (static_cast<PassArg*>(arg));
	//divide(probX_PLOU.probX_O*probPLOU.probO, ProbX, probPLOU_X.probO_X);
	//divide(args->probX_PLOU*args->probPLOU, args->probX, args->probPLOU_X);
	divide(args->probX_PLOU, args->probX, args->probPLOU_X);
	return nullptr;
}

void BayesianSegmentation::calProbPLOU_XThread(void)
{
	pthread_t thread1, thread2, thread3, thread4;
	PassArg arg1, arg2, arg3, arg4;
	arg1.probPLOU = probPLOU.probP;
	arg1.probX = ProbX;
	arg1.probX_PLOU = probX_PLOU.probX_P;
	pthread_create(&thread1, NULL, calSingleProbPLOU_X, static_cast<void*>(&arg1));

	arg2.probPLOU = probPLOU.probL;
	arg2.probX = ProbX;
	arg2.probX_PLOU = probX_PLOU.probX_L;
	pthread_create(&thread2, NULL, calSingleProbPLOU_X, static_cast<void*>(&arg2));

	arg3.probPLOU = probPLOU.probO;
	arg3.probX = ProbX;
	arg3.probX_PLOU = probX_PLOU.probX_O;
	pthread_create(&thread3, NULL, calSingleProbPLOU_X, static_cast<void*>(&arg3));

	arg4.probPLOU = probPLOU.probU;
	arg4.probX = ProbX;
	arg4.probX_PLOU = probX_PLOU.probX_U;
	pthread_create(&thread4, NULL, calSingleProbPLOU_X, static_cast<void*>(&arg4));

	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
	pthread_join(thread3, NULL);
	pthread_join(thread4, NULL);

	probPLOU_X.probP_X = arg1.probPLOU_X;
	probPLOU_X.probL_X = arg2.probPLOU_X;
	probPLOU_X.probO_X = arg3.probPLOU_X;
	probPLOU_X.probU_X = arg4.probPLOU_X;
}

void BayesianSegmentation::calcBayesian(Mat img)
{
	
#if !USE_THREAD
	// calculate the probability of X given P / L / O / U
	calcProb(img);

	// calculate the probability of X
	//ProbX = probX_P*probP + probX_L*probL + probX_O*probO + probX_U*probU;
	ProbX = probX_PLOU.probX_P*probPLOU.probP + probX_PLOU.probX_L*probPLOU.probL + probX_PLOU.probX_O*probPLOU.probO + probX_PLOU.probX_U*probPLOU.probU;

	
	// calculate the propability of P/L/O/U given X
	divide(probX_PLOU.probX_P*probPLOU.probP, ProbX, probPLOU_X.probP_X);
	divide(probX_PLOU.probX_L*probPLOU.probL, ProbX, probPLOU_X.probL_X);
	divide(probX_PLOU.probX_O*probPLOU.probO, ProbX, probPLOU_X.probO_X);
	divide(probX_PLOU.probX_U*probPLOU.probU, ProbX, probPLOU_X.probU_X);

	/*writeCSV(probPLOU_X.probP_X, "probP_X.csv");
	writeCSV(probPLOU_X.probL_X, "probL_X.csv");*/
#else
	// calculate the probability of X given P / L / O / U
	calcProbThread(img);

	// calculate the probability of X
	//ProbX = probX_P*probP + probX_L*probL + probX_O*probO + probX_U*probU;
	add(probX_PLOU.probX_P, probX_PLOU.probX_L, ProbX);
	add(probX_PLOU.probX_O, ProbX, ProbX);
	add(ProbX, probX_PLOU.probX_U, ProbX);
	
	// calculate the propability of P/L/O/U given X
	calProbPLOU_XThread();

	/*writeCSV(probPLOU_X.probP_X, "probP_X_Thread.csv");
	writeCSV(probPLOU_X.probL_X, "probL_X_Thread.csv");*/
#endif
}

void BayesianSegmentation::calcSigma(Mat img)
{
	Mat subMat, powMat, mulMat;

	// P
	subtract(GRAY_RANGE, miu.miuP, subMat);
	pow(subMat, 2, powMat);
	multiply(probPLOU_X.probP_X, powMat, mulMat);
	sigma.sigmaP = sqrt((sum(mulMat)[0]) / (N * omega.omegaP)) + 1;

	// L
	subtract(GRAY_RANGE, miu.miuL, subMat);
	pow(subMat, 2, powMat);
	multiply(probPLOU_X.probL_X, powMat, mulMat);
	sigma.sigmaL = sqrt((sum(mulMat)[0]) / (N * omega.omegaL)) + 1;

	// O
	subtract(GRAY_RANGE, miu.miuO, subMat);
	pow(subMat, 2, powMat);
	multiply(probPLOU_X.probO_X, powMat, mulMat);
	sigma.sigmaO = sqrt((sum(mulMat)[0]) / (N * omega.omegaO)) + 1;

	// P
	subtract(GRAY_RANGE, miu.miuU, subMat);
	pow(subMat, 2, powMat);
	multiply(probPLOU_X.probU_X, powMat, mulMat);
	sigma.sigmaU = sqrt((sum(mulMat)[0]) / (N * omega.omegaU)) + 1;
}

void* BayesianSegmentation::EM_updateSingleClass(void* arg)
{
	PassArg* args = (static_cast<PassArg*>(arg));
	Mat img = args->src.clone();
	Mat subMat, powMat, mulMat;

	// Calculate omega
	Mat temp, temp1, temp2;
	multiply(args->probPLOU_X, args->hist, temp);
	args->omega = sum(temp)[0] / args->N;

	// Calculate miu
	LUT(args->oldSrc, args->probPLOU_X, temp);
	if ((img.type() != CV_32F) && (img.type() != CV_64F))
		img.convertTo(img, CV_32F);
	multiply(img, temp, temp1);
	args->miu = (sum(temp1)[0]) / (args->N*args->omega);

	// Calculate sigma
	//subtract(args->src, args->miu, subMat);
	//pow(subMat, 2, powMat);
	
	subtract(GRAY_RANGE, args->miu, subMat);	
	pow(subMat, 2, powMat);
	LUT(args->src, powMat, temp2);
	
	multiply(temp, temp2, mulMat);
	args->sigma = sqrt((sum(mulMat)[0]) / (args->N * args->omega)) + 1;
	return nullptr;
}

void BayesianSegmentation::EM_updateThread(Mat oldImg, Mat img)
{
	/*if ((img.type() != CV_32F) && (img.type() != CV_64F))
		img.convertTo(img, CV_32F);*/

	pthread_t thread1, thread2, thread3, thread4;
	PassArg arg1, arg2, arg3, arg4;

	arg1.N = N;
	arg1.src = img;
	arg1.oldSrc = oldImg;
	arg1.hist = hist;
	arg1.probPLOU_X = probPLOU_X.probP_X;
	pthread_create(&thread1, NULL, EM_updateSingleClass, static_cast<void*>(&arg1));

	arg2.N = N;
	arg2.src = img;
	arg2.oldSrc = oldImg;
	arg2.hist = hist;
	arg2.probPLOU_X = probPLOU_X.probL_X;
	pthread_create(&thread2, NULL, EM_updateSingleClass, static_cast<void*>(&arg2));

	arg3.N = N;
	arg3.src = img;
	arg3.oldSrc = oldImg;
	arg3.hist = hist;
	arg3.probPLOU_X = probPLOU_X.probO_X;
	pthread_create(&thread3, NULL, EM_updateSingleClass, static_cast<void*>(&arg3));

	arg4.N = N;
	arg4.src = img;
	arg4.oldSrc = oldImg;
	arg4.hist = hist;
	arg4.probPLOU_X = probPLOU_X.probU_X;
	pthread_create(&thread4, NULL, EM_updateSingleClass, static_cast<void*>(&arg4));

	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
	pthread_join(thread3, NULL);
	pthread_join(thread4, NULL);

	miu.miuP = arg1.miu;
	miu.miuL = arg2.miu;
	miu.miuO = arg3.miu;
	miu.miuU = arg4.miu;

	omega.omegaP = arg1.omega;
	omega.omegaL = arg2.omega;
	omega.omegaO = arg3.omega;
	omega.omegaU = arg4.omega;

	sigma.sigmaP = arg1.sigma;
	sigma.sigmaL = arg2.sigma;
	sigma.sigmaO = arg3.sigma;
	sigma.sigmaU = arg4.sigma;
}

void BayesianSegmentation::EM_update(Mat oldImg, Mat img)
{
	
#if !USE_THREAD
	Mat tmp1, tmp2, tmp3, tmp4, temp;

	/*LUT(img, probPLOU_X.probP_X, temp);
	writeCSV(temp, "probP_X1.csv");*/
	multiply(probPLOU_X.probP_X, hist, tmp1);
	omega.omegaP = sum(tmp1)[0] / N;

	multiply(probPLOU_X.probL_X, hist, tmp2);
	omega.omegaL = sum(tmp2)[0] / N;

	multiply(probPLOU_X.probO_X, hist, tmp3);
	omega.omegaO = sum(tmp3)[0] / N;

	multiply(probPLOU_X.probU_X, hist, tmp4);
	omega.omegaU = sum(tmp4)[0] / N;

	LUT(oldImg, probPLOU_X.probP_X, tmp1);
	LUT(oldImg, probPLOU_X.probL_X, tmp2);
	LUT(oldImg, probPLOU_X.probO_X, tmp3);
	LUT(oldImg, probPLOU_X.probU_X, tmp4);

	if ((img.type() != CV_32F) && (img.type() != CV_64F))
		img.convertTo(img, CV_32F);

	Mat temp1, temp2, temp3, temp4;
	
	multiply(img, tmp1, temp1);
	miu.miuP = (sum(temp1)[0]) / (N*omega.omegaP);

	
	multiply(img, tmp2, temp2);
	miu.miuL = (sum(temp2)[0]) / (N*omega.omegaL);

	
	multiply(img, tmp3, temp3);
	miu.miuO = (sum(temp3)[0]) / (N*omega.omegaO);

	
	multiply(img, tmp4, temp4);
	miu.miuU = (sum(temp4)[0]) / (N*omega.omegaU);

	Mat subMat, powMat, mulMat;

	// P
	subtract(img, miu.miuP, subMat);
	pow(subMat, 2, powMat);
	multiply(tmp1, powMat, mulMat);
	sigma.sigmaP = sqrt((sum(mulMat)[0]) / (N * omega.omegaP)) + 1;

	// L
	subtract(img, miu.miuL, subMat);
	pow(subMat, 2, powMat);
	multiply(tmp2, powMat, mulMat);
	sigma.sigmaL = sqrt((sum(mulMat)[0]) / (N * omega.omegaL)) + 1;

	// O
	subtract(img, miu.miuO, subMat);
	pow(subMat, 2, powMat);
	multiply(tmp3, powMat, mulMat);
	sigma.sigmaO = sqrt((sum(mulMat)[0]) / (N * omega.omegaO)) + 1;

	// P
	subtract(img, miu.miuU, subMat);
	pow(subMat, 2, powMat);
	multiply(tmp4, powMat, mulMat);
	sigma.sigmaU = sqrt((sum(mulMat)[0]) / (N * omega.omegaU)) + 1;
#else
	EM_updateThread(oldImg, img);
#endif

}

void BayesianSegmentation::Prior(void)
{
	// One way to compute the prior probability
	probPLOU.probP = omega.omegaP;
	probPLOU.probL = omega.omegaL;
	probPLOU.probO = omega.omegaO;
	probPLOU.probU = omega.omegaU;
}

void BayesianSegmentation::sigmaInit(double sigmaP, double sigmaL, double sigmaO, double sigmaU)
{
	sigma.sigmaP = sigmaP;
	sigma.sigmaL = sigmaL;
	sigma.sigmaO = sigmaO;
	sigma.sigmaU = sigmaU;
}

void BayesianSegmentation::miuInit(double miuP, double miuL, double miuO, double miuU)
{
	miu.miuP = miuP;
	miu.miuL = miuL;
	miu.miuO = miuO;
	miu.miuU = miuU;
}

void BayesianSegmentation::probPLOUInit(double probP, double probL, double probO, double probU)
{
	probPLOU.probP = probP;
	probPLOU.probL = probL;
	probPLOU.probO = probO;
	probPLOU.probU = probU;
}

void BayesianSegmentation::ObjectSeg(Mat img, double threshValue)
{
	Mat objMask;
	threshold(img, objMask, threshValue, 255, THRESH_BINARY_INV);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

	//apply morphology filter to the image
	morphologyEx(objMask, objMask, MORPH_OPEN, kernel);
	imshow("My Window", objMask);
}

