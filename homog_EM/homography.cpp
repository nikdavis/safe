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
#define Y_TRANSLATION	( 20 )
#define Z_TRANSLATION	( 180 )
#define CAM_RES_Y		( 480 )
#define CAM_RES_X		( 640 )
#define FOCAL_IN_PX		( 550 )
#define OUTPUT_SIZE_X	( 400 )
#define OUTPUT_SIZE_Y	( 330 )

/* NOTE: should really substitute camera calibration matrix in for
 * focal length, and center (CAM_RES_X, CAM_RES_Y). Will do that soon.
 */



void planeToPlaneHomog(cv::Mat &in, cv::Mat &out, cv::Mat &H, int outputWidth) {
	warpPerspective(in, out, H, cv::Size(outputWidth, OUTPUT_SIZE_Y));
}

/* This function is to remap a point in Homography transformed image
 * to a point of original image */
void pointHomogToPointOrig(cv::Mat invH, cv::Point &input, cv::Point &output)
{
	/* Convert from point to matrix */
	cv::Mat posHomog = (cv::Mat_<float>(3, 1) << input.x , input.y, 1);

	/* Convert pos's type to the same type as invH's type for matrix multiplication */
	posHomog.convertTo(posHomog, invH.type());

	cv::Mat posOrig = invH*posHomog;

	/* Normalize the position in the original image */
	output.x = round(posOrig.at<double>(0, 0) / posOrig.at<double>(2, 0));
	output.y = round(posOrig.at<double>(1, 0) / posOrig.at<double>(2, 0));
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
        0, 1, 0, 20,
        0, 0, 1, Z_TRANSLATION,
        0, 0, 0, 1);

    // Camera Intrisecs cv::Matrix 3D -> 2D
    cv::Mat A2 = (cv::Mat_<double>(3,4) <<
        FOCAL_IN_PX,	0,				CAM_RES_X/2,	0,
        0,				FOCAL_IN_PX,	CAM_RES_Y/2,	0,
        0,				0,				1,				0);
	 
	cv::Mat invH = (A2 * (T * (R * A1)));
	H = invH.inv();
    //cout << "theta: " << theta << endl;
    //cout << "gamma: " << gamma << endl;
}


void genHomogMat(cv::Mat &H, float theta, float gamma) {
	float beta = gamma;		/* Due to the order of our rotations gamma (Y-axis) gets mapped to beta (Z-axis) */
	theta = theta - 90;		/* Turn camera downward */

	/* Convert to rads */
	theta = theta * CV_PI / 180.0;
	beta = beta * CV_PI / 180.0;
	
	double costheta = cos(theta);
	double sintheta = sin(theta);
	double cosbeta = cos(beta);
	double sinbeta = sin(beta);

	cv::Mat invH = (cv::Mat_<double>(3, 3) <<
		FOCAL_IN_PX*cosbeta + (CAM_RES_X*sinbeta*sintheta) / 2, (CAM_RES_X*cosbeta*sintheta) / 2 - FOCAL_IN_PX*sinbeta, (CAM_RES_Y*(FOCAL_IN_PX*sinbeta - (CAM_RES_X*cosbeta*sintheta) / 2)) / 2 - (OUTPUT_SIZE_X*(FOCAL_IN_PX*cosbeta + (CAM_RES_X*sinbeta*sintheta) / 2)) / 2 + (CAM_RES_X*Z_TRANSLATION) / 2,
		sinbeta*(FOCAL_IN_PX*costheta + (CAM_RES_Y*sintheta) / 2), cosbeta*(FOCAL_IN_PX*costheta + (CAM_RES_Y*sintheta) / 2), (CAM_RES_Y*Z_TRANSLATION) / 2 + FOCAL_IN_PX*Y_TRANSLATION - (CAM_RES_Y*cosbeta*(FOCAL_IN_PX*costheta + (CAM_RES_Y*sintheta) / 2)) / 2 - (OUTPUT_SIZE_X*sinbeta*(FOCAL_IN_PX*costheta + (CAM_RES_Y*sintheta) / 2)) / 2,
		sinbeta*sintheta, cosbeta*sintheta, Z_TRANSLATION - (CAM_RES_Y*cosbeta*sintheta) / 2 - (OUTPUT_SIZE_X*sinbeta*sintheta) / 2);
		
	H = invH.inv();
}
