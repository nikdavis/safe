/* bayesSeg.hpp */
#ifndef __BAYES_SEG_HPP__
#define __BAYES_SEG_HPP__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>

#ifndef USE_THREAD
#define USE_THREAD 1
#endif
#define P_CONST ((double) 0.3989422804)

using namespace cv;
using namespace std;

class BayesianSegmentation
{
public:
	static Mat GRAY_RANGE;
	
	struct InterVar
	{
		Mat interP;
		Mat interL;
		Mat interO;
		Mat interU;
	} interVarPLOU;

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

	Mat hist;

	struct PassArg
	{
		Mat src;
		Mat probX_PLOU;
		Mat probX;
		Mat probPLOU_X;
		Mat hist;
		Mat interVar;
		double probPLOU;
		double miu;
		double sigma;
		double omega;
		int N;
	};

	int N;

	void calcHistogram(Mat* img);

	static void* calcSingleProb(void* arg);

	void calcProbThread(Mat img);

	void calcProb(Mat img);

	static void* calSingleProbPLOU_X(void* arg);

	void calProbPLOU_XThread(void);

	void calcBayesian(Mat img);

	void calcSigma(void);

	static void* EM_updateSingleClass(void* arg);

	void EM_updateThread(Mat img);

	void EM_update(Mat img);

	void Prior(void);

	void sigmaInit(double sigmaP, double sigmaL, double sigmaO, double sigmaU);

	void miuInit(double miuP, double miuL, double miuO, double miuU);

	void probPLOUInit(double probP, double probL, double probO, double probU);

	void ObjectSeg(Mat img, double thresValue);
};

#endif /*__BAYES_SEG_HPP__*/
