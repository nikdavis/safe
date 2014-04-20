/*
 * homography.hpp
 *
 *  Created on: Feb 27, 2014
 *      Author: nik
 */

#ifndef HOMOGRAPHY_HPP_
#define HOMOGRAPHY_HPP_

#include <opencv2/core/core.hpp>

void calcAnglesFromVP(cv::Mat &vp, float &theta, float &gamma);
void generateHomogMat(cv::Mat &H, float theta, float gamma);
void planeToPlaneHomog(cv::Mat &in, cv::Mat &out, cv::Mat &H, int outputWidth);

#endif /* HOMOGRAPHY_HPP_ */



