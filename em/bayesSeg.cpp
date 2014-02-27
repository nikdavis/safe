#include "bayesSeg.hpp"

using namespace cv;
using namespace std;


void BayesianSegmentation::calBayesianThread(Mat input)
{
	pthread_t thread1, thread2, thread3, thread4;
	int  iret1, iret2, iret3, iret4;
	PassArg passArg1, passArg2, passArg3, passArg4 ;
	passArg1.src = input;
	passArg1.miu = miu.miuP;
	passArg1.sigma = sigma.sigmaP;
	iret1 = pthread_create(&thread1, NULL, calProbThread, static_cast<void*>(&passArg1));

	passArg2.src = input;
	passArg2.miu = miu.miuL;
	passArg2.sigma = sigma.sigmaL;
	iret2 = pthread_create(&thread2, NULL, calProbThread, static_cast<void*>(&passArg2));

	passArg3.src = input;
	passArg3.miu = miu.miuO;
	passArg3.sigma = sigma.sigmaO;
	iret3 = pthread_create(&thread3, NULL, calProbThread, static_cast<void*>(&passArg3));

	passArg4.src = input;
	passArg4.miu = miu.miuU;
	passArg4.sigma = sigma.sigmaU;
	iret4 = pthread_create(&thread4, NULL, calProbThread, static_cast<void*>(&passArg4));

	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
	pthread_join(thread3, NULL);
	pthread_join(thread4, NULL);

	probX_PLOU.probX_P = passArg1.probX_PLOU;
	probX_PLOU.probX_L = passArg2.probX_PLOU;
	probX_PLOU.probX_O = passArg3.probX_PLOU;
	probX_PLOU.probX_U = passArg4.probX_PLOU;
}

void* BayesianSegmentation::calProbThread(void *arg)
{
	PassArg* args = (static_cast<PassArg*>(arg));
	if ((args->src.type() != CV_32F) && (args->src.type() != CV_64F))
		args->src.convertTo(args->src, CV_32F);
	double p;
	//out = (1/sigma*sqrt(2*pi))*exp(-(X-miu).^2/(2*sigma^2));
	// P_CONST: 1/sqrt(2*pi)
	Mat subMat, powMat, divMat, negDivMat, expMat, outMat;
	subtract(args->src, args->miu, subMat);		// X - miu
	pow(subMat, 2, powMat);					// (X - miu).^2
	divide(powMat, 2 * args->sigma * args->sigma, divMat);	// (X-miu).^2/(2*sigma^2)
	subtract(0, divMat, negDivMat);			// -(X-miu).^2/(2*sigma^2)
	exp(negDivMat, expMat);					// exp(-(X-miu).^2/(2*sigma^2));
	p = P_CONST / args->sigma;					// (1/(sigma*sqrt(2*pi)))
	args->probX_PLOU = expMat*p;						// (1/sigma*sqrt(2*pi))*exp(-(X-miu).^2/sigma.^2);
	//multiply(expMat, Mat::ones(expMat.size(), expMat.type()), outMat, p);			

	//return nullptr;
}

Mat BayesianSegmentation::calProb(Mat src, double sigma, double miu)
{
	if ((src.type() != CV_32F) && (src.type() != CV_64F))
		src.convertTo(src, CV_32F);
	double p;
	//out = (1/sigma*sqrt(2*pi))*exp(-(X-miu).^2/(2*sigma^2));
	// P_CONST: 1/sqrt(2*pi)
	Mat subMat, powMat, divMat, negDivMat, expMat, outMat;
	subtract(src, Scalar(miu), subMat);		// X - miu
	pow(subMat, 2, powMat);					// (X - miu).^2
	divide(powMat, 2 * sigma*sigma, divMat);	// (X-miu).^2/(2*sigma^2)
	subtract(0, divMat, negDivMat);			// -(X-miu).^2/(2*sigma^2)
	exp(negDivMat, expMat);					// exp(-(X-miu).^2/(2*sigma^2));
	p = P_CONST / sigma;					// (1/(sigma*sqrt(2*pi)))
	outMat = expMat*p;						// (1/sigma*sqrt(2*pi))*exp(-(X-miu).^2/sigma.^2);
	//multiply(expMat, Mat::ones(expMat.size(), expMat.type()), outMat, p);			
	return outMat;
}



void BayesianSegmentation::calBayesian(Mat input)
{
#if !USE_THREAD
	// calculate the probability of X given P / L / O / U
	probX_PLOU.probX_P = calProb(input, sigma.sigmaP, miu.miuP);
	probX_PLOU.probX_L = calProb(input, sigma.sigmaL, miu.miuL);
	probX_PLOU.probX_O = calProb(input, sigma.sigmaO, miu.miuO);
	probX_PLOU.probX_U = calProb(input, sigma.sigmaU, miu.miuU);
#else
	calBayesianThread(input);
#endif

	// calculate the probability of X
	// ProbX = probX_P*probP + probX_L*probL + probX_O*probO + probX_U*probU;
	ProbX = probX_PLOU.probX_P*probPLOU.probP + probX_PLOU.probX_L*probPLOU.probL + probX_PLOU.probX_O*probPLOU.probO + probX_PLOU.probX_U*probPLOU.probU;

#if !USE_THREAD
	// calculate the propability of P/L/O/U given X
	divide(probX_PLOU.probX_P*probPLOU.probP, ProbX, probPLOU_X.probP_X);
	divide(probX_PLOU.probX_L*probPLOU.probL, ProbX, probPLOU_X.probL_X);
	divide(probX_PLOU.probX_O*probPLOU.probO, ProbX, probPLOU_X.probO_X);
	divide(probX_PLOU.probX_U*probPLOU.probU, ProbX, probPLOU_X.probU_X);
#else
	pthread_t thread1, thread2, thread3, thread4;
	int  iret1, iret2, iret3, iret4;
	PassArg arg1, arg2, arg3, arg4;
	arg1.probPLOU = probPLOU.probP;
	arg1.probX = ProbX;
	arg1.probX_PLOU = probX_PLOU.probX_P;
	iret1 = pthread_create(&thread1, NULL, calProbPLOU_X, static_cast<void*>(&arg1));

	arg2.probPLOU = probPLOU.probL;
	arg2.probX = ProbX;
	arg2.probX_PLOU = probX_PLOU.probX_L;
	iret2 = pthread_create(&thread2, NULL, calProbPLOU_X, static_cast<void*>(&arg2));

	arg3.probPLOU = probPLOU.probO;
	arg3.probX = ProbX;
	arg3.probX_PLOU = probX_PLOU.probX_O;
	iret3 = pthread_create(&thread3, NULL, calProbPLOU_X, static_cast<void*>(&arg3));

	arg4.probPLOU = probPLOU.probU;
	arg4.probX = ProbX;
	arg4.probX_PLOU = probX_PLOU.probX_U;
	iret4 = pthread_create(&thread4, NULL, calProbPLOU_X, static_cast<void*>(&arg4));

	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
	pthread_join(thread3, NULL);
	pthread_join(thread4, NULL);

	probPLOU_X.probP_X = arg1.probPLOU_X;
	probPLOU_X.probL_X = arg2.probPLOU_X;
	probPLOU_X.probO_X = arg3.probPLOU_X;
	probPLOU_X.probU_X = arg4.probPLOU_X;
#endif
}

void* BayesianSegmentation::calProbPLOU_X(void* arg)
{
	PassArg* args = (static_cast<PassArg*>(arg));
	//divide(probX_PLOU.probX_O*probPLOU.probO, ProbX, probPLOU_X.probO_X);
	divide(args->probX_PLOU*args->probPLOU, args->probX, args->probPLOU_X);
	//return nullptr;
}

double BayesianSegmentation::calSigma(Mat input, double miu, double omega, int N)
{
	Mat subMat, powMat;
	subtract(input, Scalar(miu), subMat);
	pow(subMat, 2, powMat);
	return sqrt((sum(powMat)[0]) / (N*omega));
}

void* BayesianSegmentation::EM_update_class(void* arg)
{
	PassArg* args = (static_cast<PassArg*>(arg));
	args->omega = sum(args->probPLOU_X)[0] / args->N;
	Mat temp;
	cv::multiply(args->src, args->probPLOU_X, temp);
	args->miu = (sum(temp)[0]) / (args->N*args->omega);
	args->sigma = calSigma(args->src, args->miu, args->omega, args->N);
	//return nullptr;
}

void BayesianSegmentation::EM_update(Mat input)
{
	int N = input.size().area();
#if !USE_THREAD
	omega.omegaP = sum(probPLOU_X.probP_X)[0] / N;
	omega.omegaL = sum(probPLOU_X.probL_X)[0] / N;
	omega.omegaO = sum(probPLOU_X.probO_X)[0] / N;
	omega.omegaU = sum(probPLOU_X.probU_X)[0] / N;

	Mat temp1, temp2, temp3, temp4;
	multiply(input, probPLOU_X.probP_X, temp1);
	miu.miuP = (sum(temp1)[0]) / (N*omega.omegaP);

	multiply(input, probPLOU_X.probL_X, temp2);
	miu.miuL = (sum(temp2)[0]) / (N*omega.omegaL);

	multiply(input, probPLOU_X.probO_X, temp3);
	miu.miuO = (sum(temp3)[0]) / (N*omega.omegaO);

	multiply(input, probPLOU_X.probU_X, temp4);
	miu.miuU = (sum(temp4)[0]) / (N*omega.omegaU);

	// update sigma
	sigma.sigmaP = calSigma(input, miu.miuP, omega.omegaP, N);
	sigma.sigmaL = calSigma(input, miu.miuL, omega.omegaL, N);
	sigma.sigmaO = calSigma(input, miu.miuO, omega.omegaO, N);
	sigma.sigmaU = calSigma(input, miu.miuU, omega.omegaU, N);
#else
	pthread_t thread1, thread2, thread3, thread4;
	int  iret1, iret2, iret3, iret4;
	PassArg arg1, arg2, arg3, arg4;

	arg1.N = N;
	arg1.src = input;
	arg1.probPLOU_X = probPLOU_X.probP_X;
	iret1 = pthread_create(&thread1, NULL, EM_update_class, static_cast<void*>(&arg1));

	arg2.N = N;
	arg2.src = input;
	arg2.probPLOU_X = probPLOU_X.probL_X;
	iret2 = pthread_create(&thread2, NULL, EM_update_class, static_cast<void*>(&arg2));

	arg3.N = N;
	arg3.src = input;
	arg3.probPLOU_X = probPLOU_X.probO_X;
	iret3 = pthread_create(&thread3, NULL, EM_update_class, static_cast<void*>(&arg3));

	arg4.N = N;
	arg4.src = input;
	arg4.probPLOU_X = probPLOU_X.probU_X;
	iret4 = pthread_create(&thread4, NULL, EM_update_class, static_cast<void*>(&arg4));

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

#endif
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

void BayesianSegmentation::priorProb(void)
{
	probPLOU.probP = omega.omegaP;
	probPLOU.probL = omega.omegaL;
	probPLOU.probO = omega.omegaO;
	probPLOU.probU = omega.omegaU;
}
