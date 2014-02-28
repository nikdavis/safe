/*
 * homography.cpp
 *
 *  Created on: Feb 27, 2014
 *      Author: nik
 */

#include "homography.hpp"



/* We want to move up away from the road, so intuition says
 * we want a positive Y translation. But Y gets mapped to
 * Z, so we translate positive Z. Or something like that. */
/* This is "in pixels" which are relative to the image sensor size */
#define Z_TRANSLATION	(309)
#define CAM_RES_Y		(480)
#define CAM_RES_X		(854)
#define FOCAL_IN_PX		(550)
#define OUTPUT_SIZE_X	(416)
#define OUTPUT_SIZE_Y	(480)

/* NOTE: should really substitute camera calibration matrix in for
 * focal length, and center (CAM_RES_X, CAM_RES_Y). Will do that soon.
 */



void planeToPlaneHomog(cv::Mat &in, cv::Mat &out, cv::Mat &H, int outputWidth) {
	warpPerspective(in, out, H, cv::Size(outputWidth, OUTPUT_SIZE_Y));
}


/* Will generate a 2D to 2D homography matrix that can be used in
 * cv::warpPerspective to generate a bird-eye view transformation.
 * The arguments are passed in units of DEGREES. X and y represent
 * the new image size desired. */
void generateHomogMat(cv::Mat &H, float theta, float gamma) {
	float beta = gamma;		/* Due to the order of our rotations gamma (Y-axis) gets mapped to beta (Z-axis) */
	theta = theta - 90;		/* Turn camera downward */

	/* Convert to rads */
	theta = theta * CV_PI / 180.0;
	beta = beta * CV_PI / 180.0;

    cv::Mat A1 = (cv::Mat_<double>(4,3) <<
        1, 0, -OUTPUT_SIZE_X/2,
        0, 1, -CAM_RES_Y/2,
        0, 0,    0,
        0, 0,    1);

    // Rotation cv::Matrices around the X,Y,Z axis
    cv::Mat RX = (cv::Mat_<double>(4, 4) <<
        1,          0,           0, 0,
        0, cos(theta), -sin(theta), 0,
        0, sin(theta),  cos(theta), 0,
        0,          0,           0, 1);

    /* Normally we need to adjust for gamma (left/right camera deviation) but
     * since X rotation is applied first our original gamma (Y) is then mapped to the Z
     * axis */

    cv::Mat RY = (cv::Mat_<double>(4, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1);

    /* Gamma (y rotation) gets mapped into Beta (z) after x rotation */
    cv::Mat RZ = (cv::Mat_<double>(4, 4) <<
        cos(beta), -sin(beta), 0, 0,
        sin(beta),  cos(beta), 0, 0,
        0,          0,           1, 0,
        0,          0,           0, 1);

    // Composed rotation cv::Matrix with (RX,RY,RZ)
    cv::Mat R = RX * RY * RZ;

    // Translation cv::Matrix on the Z axis change dist will change the height
    cv::Mat T = (cv::Mat_<double>(4, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, Z_TRANSLATION,
        0, 0, 0, 1);

    // Camera Intrisecs cv::Matrix 3D -> 2D
    cv::Mat A2 = (cv::Mat_<double>(3,4) <<
        FOCAL_IN_PX,	0,				CAM_RES_X/2,	0,
        0,				FOCAL_IN_PX,	CAM_RES_Y/2,	0,
        0,				0,				1,				0);

    H = A2 * (T * (R * A1));
    H = H.inv();
    cout << "theta: " << theta << endl;
    cout << "gamma: " << gamma << endl;
}
