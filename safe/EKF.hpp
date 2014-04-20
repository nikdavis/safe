// FILE: EKF.hpp

#ifndef _EKF_H_
#define _EKF_H_

#include <opencv2/core/core.hpp>

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

class ExtendedKalmanFilter {
    public:
        ExtendedKalmanFilter( void );

        ~ExtendedKalmanFilter( void );

        cv::Mat processNoiseCov;

        cv::Mat measurementNoiseCov;

        cv::Mat transitionMatrix;

        cv::Mat measurementMatrix;

        cv::Mat statePre;

        cv::Mat statePost;

        cv::Mat errorCovPost;

        cv::Mat errorCovPre;

        cv::Mat Kgain;

        cv::Mat predict( void );

        cv::Mat correct( cv::Mat measure );

        void calJacobian( void );
};

#endif /* _EKF_H_ */



