/* calib.hpp */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <stdlib.h>
#include <vector>

class CameraCalibrator
{
public:
	cv::Mat map1, map2;
	cv::Mat cameraMatrix;
	cv::Mat distCoeffs;
	vector<cv::Mat> rvecs,tvecs;
	vector<vector<cv::Point3f> > objectPoints;
	vector<vector<cv::Point2f> > imagePoints;		
	Size checkerboardSize;	
	float squareLength;		
    int flag;	
	string patternToUse;
	CameraCalibrator(): checkerboardSize(cv::Size(8, 6)), 
						squareLength(26), 
						flag(0),
						patternToUse("CHECKER_BOARD"),
						cameraMatrix(cv::Mat::eye(3, 3, CV_64F)),
						distCoeffs(cv::Mat::zeros(5, 1, CV_64F)){};

	int addCalibrationData(string imgPath, cv::Size& checkerboardSize = cv::Size(8, 6));
	
	double doCalibrate(cv::Size& imageSize = cv::Size(640, 480));

	void getUndistorMapping(const cv::Mat image);

	void getUndistorMapping(cv::Size& imageSize = cv::Size(640, 480));

	void writeCameraParams(string outputFileName, string imgPath);

    void readCameraParams(string inputFileName);
};

