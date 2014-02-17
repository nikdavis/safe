#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv/highgui.h"
#include "opencv/cv.h"
#include <iostream>
#include <sys/time.h>

#define ROTATE_TAU               0
#define MIN_TAU                  5
#define MAX_TAU                  15
#define TAU                      7
#define TAU_DELTA                10
#define LANE_FILTER_ROW_OFFSET   0

using namespace cv;
using namespace std;

/** @function main */
int main ( int argc, char** argv )
{
    /// Declare variables
    Mat img, src, dst, output, cdst, kernel;
    VideoCapture video;
    Point anchor;
    double delta;
    int ddepth;
    int kernel_size;
    const char* window_name = "filter2D Demo";
    struct timeval start, finish;
    double elapsed;
    int frameNum = 0;
    int timeElapsed = 0;

    int c;
    /*
    /// Load an image
    img = imread( argv[1], 0 );
    normalize(img, img, 0, 255, NORM_MINMAX, CV_8UC1);
    img.convertTo(src, CV_32F);
    dst = Mat::zeros(src.rows, src.cols, CV_32F);
    */
    video.open(argv[1]);
    if( !video.isOpened() )
    {
       cout << "ERROR: can not open camera or video file\n";
       return -1;
    }

    /// Create window
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );
    /// Press 'ESC' to exit the program
    //kernel = (Mat_<double>(1,35) << -1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1);
    //cout << "kernel: " << kernel << endl;
    //cout << "anchor: " << kernel.at<double>(0,1) << endl;
    while(1) {
      frameNum++;

    /* Get next image. Video is grayscale so don't interpret as
     * BGR image. I.e. only grab first channel! */
      video >> img;
    if(img.ptr(0) == 0) {
        cout << "No more frames, exiting...\n";
        return 0;
    }
    if(img.channels() > 1) {
        cvtColor(img, img, CV_BGR2GRAY);
    } 
    //cout << "chan: " << img.channels() << endl;
    //normalize(img, img, 0, 255, NORM_MINMAX, CV_8UC1);
    img.convertTo(img, CV_32F);
    img.convertTo(src, CV_32F);
    dst = Mat::zeros(src.rows, src.cols, CV_32F);
    cout << "rows: " << img.rows << endl;
    cout << "cols: " << img.cols << endl;
    cout << "type: " << img.type() << ' ' << CV_8UC3 << endl;
    cout << "chan: " << img.channels() << endl;
    c = waitKey(1);
    if(c == 27) {
        break;
    }
    gettimeofday(&start, NULL);
    int aux = 0;
    int tau_cnt = 0;
#if ROTATE_TAU
    int tau = MIN_TAU;
#else
    int tau = TAU;
#endif
    for (int row=0;row<src.rows;row++) {
        float * data = src.ptr<float>(row);
        float * out = dst.ptr<float>(row);

        for (int col=0;col<src.cols;col++) {
            // Check that we're within kernel size
            if((col >= tau) && (col < (src.cols - tau)) &&
                        (row > ((src.rows / 2) + LANE_FILTER_ROW_OFFSET))) {   
                // Filter from Nietos 2010
                aux = 2 * data[col];
                aux -= data[col-tau];
                aux -= data[col+tau];
                aux -= abs((int)(data[col-tau] - data[col+tau]));
                aux = (aux < 0) ? 0 : aux;
                aux = (aux > 255) ? 255 : aux;
                out[col] = (unsigned char)aux;
                //cout << out[col] << endl;
                //col++;col++;
            } else {
                out[col] = 0;
            }
        }
        if(row > ((src.rows / 2) + LANE_FILTER_ROW_OFFSET)) {
            tau_cnt++;
            if((tau_cnt % ((src.rows / 2) / TAU_DELTA)) == 0) {
                //cout << "row: " << row << endl;
                //cout << "tau: " << tau << endl;
                tau++;
            }
        }
    }      
    dst.convertTo(dst, CV_8UC1);
    output = dst.clone();

    normalize(output, output, 0, 255, NORM_MINMAX, CV_8UC1);
    //erode(output, output, Mat());
    //dilate(output, output, Mat());
    Canny(output, output, 100, 200, 3);
    cvtColor(output, cdst, CV_GRAY2BGR);
    vector<Vec4i> lines;
    HoughLinesP(output, lines, 1, CV_PI/180, 50, 50, 10 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
    }
    cout << "lines: " << lines.size() << endl;

    gettimeofday(&finish, NULL);

    if (finish.tv_usec < start.tv_usec) {
        timeElapsed = ((finish.tv_usec + 1000000) - start.tv_usec) / 1000.0;
        cout << "NEGATIVE \n\n\n";
    } else {
        timeElapsed = (finish.tv_usec - start.tv_usec) / 1000.0;
    }
    /// Apply filter
    //filter2D(src, dst, ddepth , kernel, anchor, delta, BORDER_DEFAULT );
    cout << "elapsed ms: " << (finish.tv_usec - start.tv_usec) / 1000.0 << endl;
    //cout << "start: " << start.tv_usec << endl;
    //cout << "finish: " << finish.tv_usec << endl;
    //dst.convertTo(output, CV_8UC1);
    imshow(window_name, cdst);
#if ROTATE_TAU
    tau++;
    if((tau % MAX_TAU) == 0) {
        tau = MIN_TAU;
    }
#endif
    }

    return 0;
}



