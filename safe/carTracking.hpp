// FILE: carTracking.hpp

#ifndef __CAR_TRACKING_HPP__
#define __CAR_TRACKING_HPP__

#include <vector>
#include <cv.h>
#include "EKF.hpp"

#define EKF_THETA_NOISE		( 1 )
#define EKF_ACCE_NOISE		( 1 )

#define ekfdt               ( EKF_DELAY_MS / 1000.0 )

// Define for temporal filter
#define NUM_IN_FRAMES		( 2 )
#define NUM_OUT_FRAMES		( 3 )
#define ERR_BOX_SIZE_PX		( 25 )
#define MAX_GAP_PX			( 20 )
#define MIN_BLOB_AREA		( 3000.0f )
#define MAX_BLOB_AREA		( 25000.0f )
#define PX_FEET_SCALE		( 50.0f/330.0f )

#define POS_LIST_LENGTH		( 5 )

class CarTracking
{
private:
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
					Pos(cv::Point2f(0, 0)),
					filterPos(cv::Point2f(0,0)),
					prev_filterPos(cv::Point2f(0,0)),
                    filterVelo(cv::Point2f(0,0)),
                    prev_posDelta(cv::Point2f(0,0)),
					c(0),
					frCount(0),
                    veloKF(4, 2, 0),
					posList(),
					direction(0.0f, 0.0f, 0.0f, 0.0f) {};
		bool					inFilter;
		bool					match;			// 
		int						inFrs;			// Number of consecutive frame to include in the filter
		int						outFrs;			// Number of consecutive frame to exclude in the filter		
		cv::Point2f				Pos;			// The Position of ObjCand
		cv::Point2f				filterPos;
        cv::Point2f       		prev_filterPos;
        cv::Point2f       		filterVelo;
        cv::Point2f             prev_posDelta;
		int						c;
		int 					frCount;
		ExtendedKalmanFilter	EKF;
        cv::KalmanFilter 		veloKF;
		std::vector<cv::Point>	posList;
		cv::Vec4f				direction;
	} ObjCand;
	
	cv::SimpleBlobDetector *blob_detector;

	cv::vector< cv::KeyPoint > homoKeypoints;

	cv::vector< cv::Point2f > origKeypoints;
	
	void addNewObjCand( cv::Point2f newPt );

	void updateInObjCand( int idx, cv::Point2f Pt );

	void updateOutObjCand( void );

	bool diffDis( cv::Point2f pt1, cv::Point2f pt2 );

	double calcDis( cv::Point2f pt1, cv::Point2f pt2) ;

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
	
	ObjCand		testObj;
	
	// LUT for the size of bounding box for objects
	static int boxSize[21];


	/* ---------------------------------------------------------------------------------
	*							SIMPLE BLOB DETECTION
	* --------------------------------------------------------------------------------*/
	cv::vector< ObjCand > objCands;

	void detect_filter(const cv::Mat &img);
	
	void fittingLine(int idx);

	void cvtCoord(const cv::Point2f &orig, cv::Point2f &cvt, const cv::Mat &img);
	
	void calAngle(const cv::Point2f &carPos, const cv::Mat &img, cv::Point2f &normVxy);

	/* ---------------------------------------------------------------------------------
	*								BOUNDING BOXES
	* --------------------------------------------------------------------------------*/
	
	cv::vector< cv::Rect > boundRect;

	void findBoundContourBox( const cv::Mat &img );
	
	/* ---------------------------------------------------------------------------------
	*								KALMAN FILTER
	* --------------------------------------------------------------------------------*/
	void initExtendKalman( int objCandIdx );
	
    void initVeloKF( ObjCand &obj );
    
    void filterVelo( ObjCand &obj );
	
	/* ---------------------------------------------------------------------------------
 	*								TESTING
 	* --------------------------------------------------------------------------------*/
 	
 	void importPos(std::string inputFileName, int lineNumberSought, cv::Point2f &pos, cv::Point2f &velo);
	
	void exportPos(std::string outputFileName, const cv::Point2f &pos, const cv::Point2f &velo);

};


#endif /* __CAR_TRACKING_HPP__ */



