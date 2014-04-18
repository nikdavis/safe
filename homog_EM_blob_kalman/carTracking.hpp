#ifndef __CAR_TRACKING_HPP__
#define __CAR_TRACKING_HPP__

#include <iostream>
#include <vector>
#include <math.h>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "homography.hpp"
#include "EKF.hpp"
#include "helpFn.hpp"

using namespace cv;
using namespace std;

// Define for Kalman filter
#define DELAY_MS			( 20 )
#define dt					( (float)DELAY_MS/1000 )
#define SAMPLE_FREQ			( 30 )
#ifndef	dt
#define dt					( 1/SAMPLE_FREQ )
#endif

#define	MEAS_NOISE			( 5 )
#define PROCESS_NOISE		( 10 )

#define EKF_THETA_NOISE		( 1 )
#define EKF_ACCE_NOISE		( 1 )

// Define for temporal filter
#define NUM_IN_FRAMES		( 20 )
#define NUM_OUT_FRAMES		( 15 )
#define ERR_BOX_SIZE_PX		( 25 )
#define MAX_GAP_PX			( 20 )
#define MIN_BLOB_AREA		( 1500.0f )
#define MAX_BLOB_AREA		( 50000.0f )
#define MIN_BOUND_BOX_EREA	( 1200 )
#define PX_FEET_SCALE		( 50.0f/330.0f )

#define POS_LIST_LENGTH		( 10 )

class CarTracking
{
private:
	// LUT for the number of consecutive frames to
	// consider object is out of frame
	static int numOutFrs[6];

	SimpleBlobDetector::Params params;

	/* ---------------------------------------------------------------------------------
	*							SIMPLE BLOB DETECTION
	* --------------------------------------------------------------------------------*/
	typedef struct ObjCand
	{
		// Initialize value for ObjCand
		ObjCand() : inFilter(false),
					match(false),
					inFrs(0),
					outFrs(0),
					Pos(Point(0, 0)), 
					filterPos(Point(0,0)),
					c(0),
					//d(0),
					EKF(), 
					posList(POS_LIST_LENGTH),
					direction(0.0f, 0.0f, 0.0f, 0.0f) {};
		bool			inFilter;
		bool			match;			// 
		int				inFrs;			// Number of consecutive frame to include in the filter
		int				outFrs;			// Number of consecutive frame to exclude in the filter		
		Point			Pos;			// The Position of ObjCand
		Point			filterPos;	
		int				c;
		//int 			d;
		ExtendedKalmanFilter	EKF;
		vector<Point>	posList;
		Vec4f			direction;
	} ObjCand;
	void addNewObjCand(Point newPt);

	void updateInObjCand(int idx, Point Pt, Mat* img);

	void updateOutObjCand(void);

	bool diffDis(Point pt1, Point pt2);

	double calcDis(Point pt1, Point pt2);



	/* ---------------------------------------------------------------------------------
	*								KALMAN FILTER
	* --------------------------------------------------------------------------------*/
	void initExtendKalman(int objCandIdx);

	/* ---------------------------------------------------------------------------------
	*								BOUNDING BOXES
	* --------------------------------------------------------------------------------*/
	vector< vector< Point > > contours_poly;

public:
	// Initialize CarTracking paramaters
	CarTracking(void);

	// LUT for the size of bounding box for objects
	static int boxSize[17];

	/* ---------------------------------------------------------------------------------
	*							SIMPLE BLOB DETECTION
	* --------------------------------------------------------------------------------*/
	vector< ObjCand > objCands;

	Ptr< FeatureDetector > blob_detector;

	vector< KeyPoint > homoKeypoints;

	vector< Point > origKeypoints;

	void detect_filter(Mat* img);
	
	void fittingLine(int idx);

	void cvtCoord(Point* orig, Point* cvt, Mat* img);
	
	void calAngle(Point* carPos, Mat* img, Point2f* normVxy);

	/* ---------------------------------------------------------------------------------
	*								BOUNDING BOXES
	* --------------------------------------------------------------------------------*/
	vector< Rect > boundRect;

	vector< vector< Point > > contours;

	vector< Vec4i > hierarchy;

	void findBoundContourBox(Mat* img);

};


#endif /* __CAR_TRACKING_HPP__ */
