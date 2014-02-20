/* main.cpp */


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "calib.hpp"
#include <vector>
 
using namespace cv;
using namespace std;


int main() 
{
	CameraCalibrator CamCalibrate;

	Mat src, undistortImg, map1, map2;
	namedWindow("Source", CV_WINDOW_AUTOSIZE);
	namedWindow("Undistorted image", CV_WINDOW_AUTOSIZE);
	string pathToData("./checkerboard 1.pgm");
	CamCalibrate.checkerboardSize = Size(8, 6);
	CamCalibrate.addCalibrationData(pathToData);
	CamCalibrate.doCalibrate();
	cout << "Done" << endl;
	cout << "Camera matrix = " << endl << CamCalibrate.cameraMatrix << endl;
	cout << "Distortion coefficients = " << endl << CamCalibrate.distCoeffs << endl;
	// Save camera parameters to .xml file for using later
	CamCalibrate.writeCameraParams("./CameraParams.xml", "./checkerboard 1.pgm");

	// Load the camera parameters from .xml file
	/*CamCalibrate.readCameraParams("E:\\OpenCV Data\\CameraParams.xml");
	cout << "Camera matrix = " << endl << CamCalibrate.cameraMatrix << endl;
	cout << "Distortion coefficient = " << endl << CamCalibrate.distCoeffs << endl;*/
	
	// Do remap to get undistored image
	src = imread("./checkerboard 1.pgm", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Source", src);
	CamCalibrate.getUndistorMapping(src);
	CamCalibrate.map1.copyTo(map1);
	CamCalibrate.map2.copyTo(map2);
	remap(src, undistortImg, map1, map2, INTER_LINEAR);
	imshow("Undistorted image", undistortImg);
	
	waitKey(0);
}


