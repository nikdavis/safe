/*
 * homography.hpp
 *
 *  Created on: Feb 27, 2014
 *      Author: nik
 */

#ifndef HOMOGRAPHY_HPP_
#define HOMOGRAPHY_HPP_

#include <opencv2/core/core.hpp>


/* We want to move up away from the road, so intuition says
 * we want a positive Y translation. But Y gets mapped to
 * Z, so we translate positive Z. Or something like that. */
/* This is "in pixels" which are relative to the image sensor size */
#define X_TRANSLATION	(0)
#define Y_TRANSLATION	(20)
#define Z_TRANSLATION	(255)
#define CAM_RES_Y		(480)
#define CAM_RES_X		(640)
#define FOCAL_IN_PX		(378)
#define OUTPUT_SIZE_X	(416)
#define OUTPUT_SIZE_Y	(480)


void calcAnglesFromVP(cv::Mat &vp, float &theta, float &gamma);
void calcVPFromAngles(int &x, int &y, float gamma, float theta);
void calcVpFromAngles(const float &theta, const float &gamma, cv::Point &vp);
void generateHomogMat(cv::Mat &H, float theta, float gamma);
void planeToPlaneHomog(cv::Mat &in, cv::Mat &out, cv::Mat &H, int outputWidth);

#endif /* HOMOGRAPHY_HPP_ */



