#include "bayesSeg.hpp"
#include <pthread.h>

using namespace cv;

Mat BayesianSegmentation::GRAY_RANGE = (Mat_<float>(256, 1) <<
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
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

/* This function will calculate the histogram of homography images.
 * Since the 0-value points in homography images are too many, that causes
 * the Expectation Maximization (EM) process for Gaussian Mixture Model (GMM)
 * incorrect. Therefore, the 0-value points causing by homography transformation
 * are assumed taking 90% of 0-value points in a whole image.
 */
void BayesianSegmentation::calcHistogram( const Mat &img )
{
    N = img.size().area();

    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;

    /// Compute the histograms:
    calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
    float before = hist.at<float>(0);
    hist.at<float>(0) = before * 0.1f;

    N -= cvRound( before * 0.9f );
}

void* BayesianSegmentation::calcProbThread( void* arg )
{
    PassArg* args = (static_cast<PassArg*>(arg));
    double p;
    //out = (1/sigma*sqrt(2*pi))*exp(-(X-miu).^2/(2*sigma^2));
    // P_CONST: 1/sqrt(2*pi)
    Mat subMat, powMat, divMat, expMat, finalMat, outMat;
    subtract(GRAY_RANGE, args->miu, subMat);                // X - miu
    pow(subMat, 2, powMat);                                 // (X - miu).^2
    divide(powMat, 2.0 * args->sigma * args->sigma, divMat);// (X-miu).^2/(2*sigma^2)
    exp(-divMat, expMat);                                   // exp(-(X-miu).^2/(2*sigma^2));
    p = P_CONST / args->sigma;                              // (1/(sigma*sqrt(2*pi)))
    finalMat = expMat * p;                                  // (1/sigma*sqrt(2*pi))*exp(-(X-miu).^2/sigma.^2);
    args->probX_PLOU = finalMat*args->probPLOU;
    return 0;
}

/* calculate the probability of X given P / L / O / U
 * + Each one will be calculated seperately in each thread.
 * + There is NO input argument. The function uses miu, sigma, probPLOU which is
 * contained in class structure.
 * + This function should be called only at the start of program and each time
 * the process is restarted.
 */
void BayesianSegmentation::calcProb( void )
{
    pthread_t thread1, thread2, thread3, thread4;
    PassArg passArg1, passArg2, passArg3, passArg4;

    passArg1.miu = miu.P;
    passArg1.sigma = sigma.P;
    passArg1.probPLOU = probPLOU.probP;
    pthread_create(&thread1, NULL, calcProbThread, static_cast<void*>(&passArg1));

    passArg2.miu = miu.L;
    passArg2.sigma = sigma.L;
    passArg2.probPLOU = probPLOU.probL;
    pthread_create(&thread2, NULL, calcProbThread, static_cast<void*>(&passArg2));

    passArg3.miu = miu.O;
    passArg3.sigma = sigma.O;
    passArg3.probPLOU = probPLOU.probO;
    pthread_create(&thread3, NULL, calcProbThread, static_cast<void*>(&passArg3));

    passArg4.miu = miu.U;
    passArg4.sigma = sigma.U;
    passArg4.probPLOU = probPLOU.probU;
    pthread_create(&thread4, NULL, calcProbThread, static_cast<void*>(&passArg4));

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    pthread_join(thread3, NULL);
    pthread_join(thread4, NULL);

    probX_PLOU.probX_P = passArg1.probX_PLOU;
    probX_PLOU.probX_L = passArg2.probX_PLOU;
    probX_PLOU.probX_O = passArg3.probX_PLOU;
    probX_PLOU.probX_U = passArg4.probX_PLOU;
}

/* This function includes updating miu, sigma, omega and calculating
 * probX_PLOU for one of P / L / O / U.
 */
void* BayesianSegmentation::EM_BayesThread( void* arg )
{
    PassArg* args = (static_cast<PassArg*>(arg));

    // calculate the propability of P/L/O/U given X
    divide(args->probX_PLOU, args->probX, args->probPLOU_X);

    // Calculate omega
    Mat m_omega;
    multiply(args->probPLOU_X, args->hist, m_omega);
    args->omega = sum(m_omega)[0] / args->N;

    // update probPLOU
    args->probPLOU = args->omega;

    // Calculate miu
    // If thread is for Undefine class, skip calculating miu and sigma.
    // Miu and sigma are assigned to fixed values
    if (args->plouClass != UNDEF)
    {
        Mat miu;
        multiply(GRAY_RANGE, args->probPLOU_X, miu);
        multiply(miu, args->hist, miu);
        args->miu = sum(miu)[0] / (args->N*args->omega);
    }
    else
        args->miu = UNDEF_DEFAULT_MIU;

    // Calculate sigma
    Mat subMat, powMat, mulMat, sigma;
    subtract(GRAY_RANGE, args->miu, subMat);                // X - miu
    pow(subMat, 2, powMat);                                 // (X - miu).^2
    // If thread is for Undefine class, skip calculating miu and sigma.
    // Miu and sigma are assigned to fixed values
    if (args->plouClass != UNDEF)
    {
        multiply(args->probPLOU_X, powMat, sigma);
        multiply(sigma, args->hist, sigma);
        args->sigma = sqrt((sum(sigma)[0]) / (args->N * args->omega)) + 1.0;
    }
    else
        args->sigma = UNDEF_DEFAULT_SIGMA;

    // Calculate new probX_PLOU for one of P / L / O / U
    Mat divMat, expMat, finalMat;
    divide(powMat, 2 * args->sigma * args->sigma, divMat);  // (X-miu).^2/(2*sigma^2)
    exp(-divMat, expMat);                                   // exp(-(X-miu).^2/(2*sigma^2));
    double p = P_CONST / args->sigma;                       // (1/(sigma*sqrt(2*pi)))
    finalMat = expMat*p;                                    // (1/sigma*sqrt(2*pi))*exp(-(X-miu).^2/sigma.^2);
    args->probX_PLOU = finalMat*args->probPLOU;
    return 0;
}

void BayesianSegmentation::EM_Bayes( const Mat &img )
{
    // calculate the probability of X
    //ProbX = probX_P*probP + probX_L*probL + probX_O*probO + probX_U*probU;
    add(probX_PLOU.probX_P, probX_PLOU.probX_L, ProbX);
    add(probX_PLOU.probX_O, ProbX, ProbX);
    add(ProbX, probX_PLOU.probX_U, ProbX);

    // Calculate the histogram of homography image
    calcHistogram(img);

    // EM update for P / L / O / U in four seperated threads
    pthread_t thread1, thread2, thread3, thread4;
    PassArg arg1, arg2, arg3, arg4;

    arg1.plouClass = PAVE;
    arg1.N = N;
    arg1.hist = hist;
    arg1.probX = ProbX;
    arg1.probX_PLOU = probX_PLOU.probX_P;
    pthread_create(&thread1, NULL, EM_BayesThread, static_cast<void*>(&arg1));

    arg2.plouClass = LANE;
    arg2.N = N;
    arg2.hist = hist;
    arg2.probX = ProbX;
    arg2.probX_PLOU = probX_PLOU.probX_L;
    pthread_create(&thread2, NULL, EM_BayesThread, static_cast<void*>(&arg2));

    arg3.plouClass = OBJ;
    arg3.N = N;
    arg3.hist = hist;
    arg3.probX = ProbX;
    arg3.probX_PLOU = probX_PLOU.probX_O;
    pthread_create(&thread3, NULL, EM_BayesThread, static_cast<void*>(&arg3));

    arg4.plouClass = UNDEF;
    arg4.N = N;
    arg4.hist = hist;
    arg4.probX = ProbX;
    arg4.probX_PLOU = probX_PLOU.probX_U;
    pthread_create(&thread4, NULL, EM_BayesThread, static_cast<void*>(&arg4));

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    pthread_join(thread3, NULL);
    pthread_join(thread4, NULL);

    miu.P = arg1.miu;
    miu.L = arg2.miu;
    miu.O = arg3.miu;
    miu.U = arg4.miu;

    omega.P = arg1.omega;
    omega.L = arg2.omega;
    omega.O = arg3.omega;
    omega.U = arg4.omega;

    sigma.P = arg1.sigma;
    sigma.L = arg2.sigma;
    sigma.O = arg3.sigma;
    sigma.U = arg4.sigma;

    probPLOU_X.probP_X = arg1.probPLOU_X;
    probPLOU_X.probL_X = arg2.probPLOU_X;
    probPLOU_X.probO_X = arg3.probPLOU_X;
    probPLOU_X.probU_X = arg4.probPLOU_X;

    probX_PLOU.probX_P = arg1.probX_PLOU;
    probX_PLOU.probX_L = arg2.probX_PLOU;
    probX_PLOU.probX_O = arg3.probX_PLOU;
    probX_PLOU.probX_U = arg4.probX_PLOU;

    probPLOU.probP = arg1.probPLOU;
    probPLOU.probL = arg2.probPLOU;
    probPLOU.probO = arg3.probPLOU;
    probPLOU.probU = arg4.probPLOU;

    probPLOU_X.probP_X.at<float>(0) = 0;
    probPLOU_X.probL_X.at<float>(0) = 0;
    probPLOU_X.probO_X.at<float>(0) = 0;
    probPLOU_X.probU_X.at<float>(0) = 0;
}

void BayesianSegmentation::sigmaInit( double sigmaP, double sigmaL, double sigmaO, double sigmaU )
{
    sigma.P = sigmaP;
    sigma.L = sigmaL;
    sigma.O = sigmaO;
    sigma.U = sigmaU;
}

void BayesianSegmentation::miuInit( double miuP, double miuL, double miuO, double miuU )
{
    miu.P = miuP;
    miu.L = miuL;
    miu.O = miuO;
    miu.U = miuU;
}

void BayesianSegmentation::probPLOUInit( double probP, double probL, double probO, double probU )
{
    probPLOU.probP = probP;
    probPLOU.probL = probL;
    probPLOU.probO = probO;
    probPLOU.probU = probU;
}

void BayesianSegmentation::classSeg( const Mat &img, Mat &obj, e_class cl ) const
{
    switch ( cl )
    {
    case PAVE:
        threshold( probPLOU_X.probP_X, probPLOU_X.probP_X, PROB_THRESHOLD, 255, CV_THRESH_BINARY );
        LUT( img, probPLOU_X.probP_X, obj );
        break;
    case LANE:
        threshold( probPLOU_X.probL_X, probPLOU_X.probL_X, PROB_THRESHOLD, 255, CV_THRESH_BINARY );
        LUT( img, probPLOU_X.probL_X, obj );
        break;
    case OBJ:
        threshold( probPLOU_X.probO_X, probPLOU_X.probO_X, PROB_THRESHOLD, 255, CV_THRESH_BINARY );
        LUT( img, probPLOU_X.probO_X, obj );
        break;
    case UNDEF:
        threshold( probPLOU_X.probU_X, probPLOU_X.probU_X, PROB_THRESHOLD, 255, CV_THRESH_BINARY );
        LUT( img, probPLOU_X.probU_X, obj );
        break;
    }

    // Normalize the obj image to grayscale
    obj.convertTo(obj, CV_8U);
}



