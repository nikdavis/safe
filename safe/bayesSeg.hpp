/* bayesSeg.hpp */
#ifndef __BAYES_SEG_HPP__
#define __BAYES_SEG_HPP__

#define P_CONST                 ((double) 0.3989422804)     // 1/sqrt(2*pi)
#define UNDEF_DEFAULT_SIGMA     ((double) 50)
#define UNDEF_DEFAULT_MIU       ((double) 250)
#define PROB_THRESHOLD          ((double) 0.7 )

enum e_class { PAVE, LANE, OBJ, UNDEF };

struct PassArg
{
    cv::Mat oldSrc;
    cv::Mat src;
    cv::Mat probX_PLOU;
    cv::Mat probX;
    cv::Mat probPLOU_X;
    cv::Mat hist;
    double probPLOU;
    double miu;
    double sigma;
    double omega;
    int N;
    int plouClass;
};

class BayesianSegmentation
{
public:
    struct ProbPLOU_X
    {
        cv::Mat probP_X;
        cv::Mat probL_X;
        cv::Mat probO_X;
        cv::Mat probU_X;
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

    void calcProb( void );
    void EM_Bayes( cv::Mat img );
    void sigmaInit( double sigmaP, double sigmaL, double sigmaO, double sigmaU );
    void miuInit( double miuP, double miuL, double miuO, double miuU );
    void probPLOUInit( double probP, double probL, double probO, double probU );
    void classSeg( cv::Mat &img, cv::Mat &obj, e_class cl );

private:
    static cv::Mat GRAY_RANGE;

    struct ProbX_PLOU
    {
        cv::Mat probX_P;
        cv::Mat probX_L;
        cv::Mat probX_O;
        cv::Mat probX_U;
    } probX_PLOU;

    struct ProbPLOU
    {
        double probP;
        double probL;
        double probO;
        double probU;
    } probPLOU;

    cv::Mat ProbX;

    cv::Mat hist;

    int N;

    void calcHistogram( cv::Mat* img );
    static void* calcProbThread( void* arg );
    static void* EM_BayesThread( void* arg );
};

#endif /*__BAYES_SEG_HPP__*/



