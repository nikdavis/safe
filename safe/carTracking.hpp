// FILE: carTracking.hpp

#ifndef __CAR_TRACKING_HPP__
#define __CAR_TRACKING_HPP__

#include <vector>
#include <cv.h>
#include "EKF.hpp"

#define EKF_THETA_NOISE		( 1 )
#define EKF_ACCE_NOISE		( 1 )

// Define for temporal filter
#define NUM_IN_FRAMES		( 15 )
#define NUM_OUT_FRAMES		( 15 )
#define ERR_BOX_SIZE_PX		( 25 )
#define MAX_GAP_PX			( 20 )
#define MIN_BLOB_AREA		( 1500.0f )
#define MAX_BLOB_AREA		( 50000.0f )
#define MIN_BOUND_BOX_EREA	( 1400 )
#define PX_FEET_SCALE		( 50.0f/330.0f )

#define POS_LIST_LENGTH		( 5 )

#define SAFETY_ELLIPSE_X	( 120 )
#define SAFETY_ELLIPSE_Y	( 240 )

class CarTracking
{
private:
	// LUT for the number of consecutive frames to
	// consider object is out of frame
	static int numOutFrs[6];
	
	// LUT for the size of bounding box for objects
	static int boxSize[17];

	cv::SimpleBlobDetector::Params params;

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
					Pos(cv::Point(0, 0)),
					filterPos(cv::Point(0,0)),
					c(0),
					frCount(0),
					EKF(),
					posList(POS_LIST_LENGTH),
					direction(0.0f, 0.0f, 0.0f, 0.0f) {};
		bool			inFilter;
		bool			match;			// 
		int				inFrs;			// Number of consecutive frame to include in the filter
		int				outFrs;			// Number of consecutive frame to exclude in the filter		
		cv::Point		Pos;			// The Position of ObjCand
		cv::Point		filterPos;
		int				c;
		int 			frCount;
		ExtendedKalmanFilter	EKF;
		std::vector<cv::Point>	posList;
		cv::Vec4f			direction;
	} ObjCand;
	
	cv::SimpleBlobDetector *blob_detector;

	cv::vector< cv::KeyPoint > homoKeypoints;

	cv::vector< cv::Point > origKeypoints;
	
	void addNewObjCand( cv::Point newPt );

	void updateInObjCand( int idx, cv::Point Pt );

	void updateOutObjCand( void );

	bool diffDis( cv::Point pt1, cv::Point pt2 );

	double calcDis( cv::Point pt1, cv::Point pt2) ;

	/* ---------------------------------------------------------------------------------
	*								KALMAN FILTER
	* --------------------------------------------------------------------------------*/
	void initExtendKalman( int objCandIdx );

	/* ---------------------------------------------------------------------------------
	*								BOUNDING BOXES
	* --------------------------------------------------------------------------------*/
	cv::vector< cv::vector< cv::Point > > contours_poly;
	
	cv::vector< cv::vector< cv::Point > > contours;

	cv::vector< cv::Vec4i > hierarchy;

public:
	// Initialize CarTracking paramaters
	CarTracking(void);
	~CarTracking(void);

	/* ---------------------------------------------------------------------------------
	*							SIMPLE BLOB DETECTION
	* --------------------------------------------------------------------------------*/
	cv::vector< ObjCand > objCands;

	void detect_filter(const cv::Mat &img);
	
	void fittingLine(int idx);

	void cvtCoord(const cv::Point &orig, cv::Point &cvt, const cv::Mat &img);
	
	void calAngle(const cv::Point &carPos, const cv::Mat &img, cv::Point2f &normVxy);

	/* ---------------------------------------------------------------------------------
	*								BOUNDING BOXES
	* --------------------------------------------------------------------------------*/
	
	cv::vector< cv::Rect > boundRect;

	void findBoundContourBox( const cv::Mat &img );

};


#endif /* __CAR_TRACKING_HPP__ */



