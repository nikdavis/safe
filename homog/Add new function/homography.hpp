/*
 * homography.hpp
 *
 *  Created on: Feb 27, 2014
 *      Author: nik
 */

#ifndef HOMOGRAPHY_HPP_
#define HOMOGRAPHY_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

using namespace std;

void generateHomogMat(cv::Mat &H, float theta, float gamma);
void pointHomogToPointOrig(cv::Mat H, cv::Point &input, cv::Point &output);
void planeToPlaneHomog(cv::Mat &in, cv::Mat &out, cv::Mat &H, int outputWidth);

#endif /* HOMOGRAPHY_HPP_ */
