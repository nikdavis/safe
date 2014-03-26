/* bayesSeg.hpp */
#ifndef __BAYES_SEG_HPP__
#define __BAYES_SEG_HPP__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define P_CONST					((double) 0.3989422804)
#define UNDEF_DEFAULT_SIGMA		((double) 50)
#define UNDEF_DEFAULT_MIU		((double) 250)

using namespace cv;
using namespace std;

class BayesianSegmentation
{
private:
	static Mat GRAY_RANGE;

	struct ProbX_PLOU
	{
		Mat probX_P;
		Mat probX_L;
		Mat probX_O;
		Mat probX_U;
	} probX_PLOU;

	struct ProbPLOU
	{
		double probP;
		double probL;
		double probO;
		double probU;
	} probPLOU;

	Mat ProbX;

	Mat hist;

	struct PassArg
	{
		Mat oldSrc;
		Mat src;
		Mat probX_PLOU;
		Mat probX;
		Mat probPLOU_X;
		Mat hist;
		double probPLOU;
		double miu;
		double sigma;
		double omega;
		int N;
		int plouClass;
		
	};

	enum ELEMENT_CLASSESS : int
	{
		PAVE = 0,
		LANE,
		OBJ,
		UNDEF
	};

	int N;

	void calcHistogram(Mat* img);

	static void* calcProbThread(void* arg);

	static void* EM_BayesThread(void* arg);

public:
	struct ProbPLOU_X
	{
		Mat probP_X;
		Mat probL_X;
		Mat probO_X;
		Mat probU_X;
	} probPLOU_X;

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

	void writeCSV(Mat data, string fileName);

	void calcProb(void);

	void EM_Bayes(Mat img);

	void sigmaInit(double sigmaP, double sigmaL, double sigmaO, double sigmaU);

	void miuInit(double miuP, double miuL, double miuO, double miuU);

	void probPLOUInit(double probP, double probL, double probO, double probU);

	void ObjectSeg(Mat img, double thresValue);
};

#endif /*__BAYES_SEG_HPP__*/
