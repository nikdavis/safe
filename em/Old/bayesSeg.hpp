/* bayesSeg.hpp */
#ifndef __BAYES_SEG_HPP__
#define __BAYES_SEG_HPP__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <pthread.h>

#ifndef USE_THREAD
#define USE_THREAD 1
#endif
#define P_CONST ((double) 0.3989422804)

using namespace cv;
using namespace std;

class BayesianSegmentation
{
public:
	struct ProbX_PLOU
	{
		Mat probX_P;
		Mat probX_L;
		Mat probX_O;
		Mat probX_U;
	} probX_PLOU;

	struct ProbPLOU_X
	{
		Mat probP_X;
		Mat probL_X;
		Mat probO_X;
		Mat probU_X;
	} probPLOU_X;

	struct ProbPLOU
	{
		double probP;
		double probL;
		double probO;
		double probU;
	} probPLOU;

	Mat ProbX;

	struct  Sigma
	{
		double sigmaP;
		double sigmaL;
		double sigmaO;
		double sigmaU;
	} sigma;
	struct Miu
	{
		double miuP;
		double miuL;
		double miuO;
		double miuU;
	} miu;

	struct Omega
	{
		double omegaP;
		double omegaL;
		double omegaO;
		double omegaU;
	} omega;

	struct PassArg
	{
		Mat src;
		double miu;
		double sigma;
		double omega;
		Mat probX_PLOU;
		Mat probX;
		double probPLOU;
		Mat probPLOU_X;
		int N;
	};

	//BayesianSegmentation();
	double singleMiu, singleSigma;

	static void* calProbThread(void* arg);

	static void* calProbPLOU_X(void* arg);

	static void* EM_update_class(void* arg);

	Mat calProb(Mat src, double sigma, double miu);

	void calBayesian(Mat input);

	void calBayesianThread(Mat input);

	static double calSigma(Mat input, double miu, double omega, int N);

	void EM_update(Mat input);

	void sigmaInit(double sigmaP, double sigmaL, double sigmaO, double sigmaU);

	void miuInit(double miuP, double miuL, double miuO, double miuU);

	void probPLOUInit(double probP, double probL, double probO, double probU);

	void priorProb(void);
};

#endif /*__BAYES_SEG_HPP__*/
