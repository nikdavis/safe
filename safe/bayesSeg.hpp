/* bayesSeg.hpp */
#ifndef __BAYES_SEG_HPP__
#define __BAYES_SEG_HPP__

//#include <opencv2/core/core.hpp>

#ifndef USE_THREAD
#define USE_THREAD 1
#endif

// 1/sqrt(2*pi)
#define P_CONST ((double) 0.3989422804)

class BayesianSegmentation
{
public:
	static cv::Mat GRAY_RANGE;
	
	struct InterVar
	{
		cv::Mat interP;
		cv::Mat interL;
		cv::Mat interO;
		cv::Mat interU;
	} interVarPLOU;

	struct ProbX_PLOU
	{
		cv::Mat probX_P;
		cv::Mat probX_L;
		cv::Mat probX_O;
		cv::Mat probX_U;
	} probX_PLOU;

	struct ProbPLOU_X
	{
		cv::Mat probP_X;
		cv::Mat probL_X;
		cv::Mat probO_X;
		cv::Mat probU_X;
	} probPLOU_X;

	struct ProbPLOU
	{
		double probP;
		double probL;
		double probO;
		double probU;
	} probPLOU;

	cv::Mat ProbX;
	
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

	cv::Mat hist;

	struct PassArg
	{
		cv::Mat src;
		cv::Mat probX_PLOU;
		cv::Mat probX;
		cv::Mat probPLOU_X;
		cv::Mat hist;
		cv::Mat interVar;
		double probPLOU;
		double miu;
		double sigma;
		double omega;
		int N;
	};

	int N;

	void calcHistogram(cv::Mat &img);

	static void* calcSingleProb(void* arg);

	void calcProbThread(cv::Mat &img);

	void calcProb(cv::Mat &img);

	static void* calSingleProbPLOU_X(void* arg);

	void calProbPLOU_XThread(void);

	void calcBayesian(cv::Mat &img);

	void calcSigma(void);

	static void* EM_updateSingleClass(void* arg);

	void EM_updateThread(cv::Mat &img);

	void EM_update(cv::Mat img);

	void Prior(void);

	void sigmaInit(double sigmaP, double sigmaL, double sigmaO, double sigmaU);

	void miuInit(double miuP, double miuL, double miuO, double miuU);

	void probPLOUInit(double probP, double probL, double probO, double probU);

	void ObjectSeg(const cv::Mat &img, double thresValue, cv::Mat &objMask) const;
};

#endif /*__BAYES_SEG_HPP__*/



