#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
using namespace std;

#define DELAY_MS        62.5
#define dt              ( DELAY_MS / 1000.0 )
#define SAMPLE_FREQ     30.0
#ifndef dt
#define dt              ( 1.0 / SAMPLE_FREQ )
#endif

#define MEAS_NOISE      0.005
#define PROCESS_NOISE   0.5


struct mouse_info_struct { int x,y; };
struct mouse_info_struct mouse_info = {-1,-1}, last_mouse;

//vector<Point> mousev,kalmanv;

void on_mouse(int event, int x, int y, int flags, void* param) 
{
    last_mouse = mouse_info;
    mouse_info.x = x;
    mouse_info.y = y;
}

int main (int argc, char * const argv[]) 
{
    Mat img(700, 700, CV_8UC3);
    KalmanFilter KF(4, 2, 0);       // 4 dynamic parameters, 2 measurement parameters, and no control
    //Mat_<float> state(4, 1);        // (x, y, Vx, Vy)
    //Mat processNoise(4, 1, CV_32F);
    Mat_<float> measurement(2,1); 
    measurement.setTo(Scalar(0));
    char code = (char)-1;

    srand(time(NULL));

    namedWindow("mouse kalman");
    setMouseCallback("mouse kalman", on_mouse, 0);

    while ( mouse_info.x < 0 || mouse_info.y < 0) {
        imshow("mouse kalman", img);
        waitKey(30);
    }

    KF.statePre.at<float>(0) = mouse_info.x;
    KF.statePre.at<float>(1) = mouse_info.y;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;
    KF.transitionMatrix = *(Mat_<float>(4, 4) <<
                    1,      0,      dt,     0,
                    0,      1,      0,      dt,
                    0,      0,      1,      0,
                    0,      0,      0,      1);

    setIdentity(KF.measurementMatrix);
    //setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    KF.processNoiseCov = *(Mat_<float>(4, 4) <<
        pow((float)dt, 4)/4.0,    0,                      pow((float)dt, 3)/3.0,    0,
        0,                      pow((float)dt, 4)/4.0,    0,                      pow((float)dt, 3)/3.0,
        pow((float)dt, 3)/3.0,    0,                      pow((float)dt, 2)/2.0,    0,
        0,                      pow((float)dt, 3)/3.0,    0,                      pow((float)dt, 2)/2.0);
    KF.processNoiseCov = KF.processNoiseCov*(PROCESS_NOISE*PROCESS_NOISE);
    setIdentity(KF.measurementNoiseCov, Scalar::all(MEAS_NOISE*MEAS_NOISE));
    setIdentity(KF.errorCovPost, Scalar::all(0.1));

    cout << "dt = " << dt << endl;
    cout << "measurement matrix = " << KF.measurementMatrix << endl;
    cout << "Process noise Cov = " << KF.processNoiseCov << endl;
    cout << "Measurement Noise Cov = " << KF.measurementNoiseCov << endl;

    //mousev.clear();
    //kalmanv.clear();
    //randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));
    Mat estimated;
    while( true )
    {
        Mat prediction = KF.predict();
        Point predictPt(prediction.at<float>(0),prediction.at<float>(1));

        measurement(0) = mouse_info.x;//+(((rand()/(float)RAND_MAX)-0.5)*100.0);
        measurement(1) = mouse_info.y;//+(((rand()/(float)RAND_MAX)-0.5)*100.0);

        Point measPt(measurement(0),measurement(1));
        //mousev.push_back(measPt);
        // generate measurement
        //measurement += KF.measurementMatrix*state;

        if( code != 'h' ) estimated = KF.correct(measurement);
        Point statePt(estimated.at<float>(0),estimated.at<float>(1));
        //kalmanv.push_back(statePt);

        // plot points
        #define drawCross( center, color, d )                           \
        line( img, Point( center.x - d, center.y - d ),                 \
        Point( center.x + d, center.y + d ), color, 2, CV_AA, 0);       \
        line( img, Point( center.x + d, center.y - d ),                 \
        Point( center.x - d, center.y + d ), color, 2, CV_AA, 0 )

        img = Scalar::all(0);
        drawCross( statePt, Scalar(255,255,255), 5 );
        drawCross( measPt, Scalar(0,0,255), 5 );
        drawCross( predictPt, Scalar(0,255,0), 3 );
        //line( img, statePt, measPt, Scalar(0,0,255), 3, CV_AA, 0 );
        //line( img, statePt, predictPt, Scalar(0,255,255), 3, CV_AA, 0 );

        //for (unsigned int i = 0; i < mousev.size()-1; i++)
        //    line(img, mousev[i], mousev[i+1], Scalar(255,255,0), 1);

        //for (unsigned int i = 0; i < kalmanv.size()-1; i++)
        //    line(img, kalmanv[i], kalmanv[i+1], Scalar(0,255,0), 1);

        //randn( processNoise, Scalar(0), Scalar::all(sqrt(KF.processNoiseCov.at<float>(0, 0))));
        //state = KF.transitionMatrix*state + processNoise;

        imshow( "mouse kalman", img );
        code = (char)waitKey(DELAY_MS);

        if( code == 27 || code == 'q' || code == 'Q' )
            break;
    }

    return 0;
}



