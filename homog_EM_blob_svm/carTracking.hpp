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
#include "CarSVM.hpp"

using namespace cv;
using namespace std;

// Define for Kalman filter
#define DELAY_MS			( 30 )
#define dt					( DELAY_MS/1000 )
#define SAMPLE_FREQ			( 30 )
#ifndef	dt
#define dt					( 1/SAMPLE_FREQ )
#endif

#define	MEAS_NOISE			( 0.0005 )
#define PROCESS_NOISE		( 100 )

// Define for temporal filter
#define NUM_IN_FRAMES		( 10 )
#define NUM_OUT_FRAMES		( 25 )
#define ERR_BOX_SIZE_PX		( 25 )
#define MAX_GAP_PX			( 20 )
#define MIN_BLOB_AREA		( 1500.0f )
#define MAX_BLOB_AREA		( 50000.0f )
#define MIN_BOUND_BOX_EREA	( 1200 )

// Define svm sliding window step
#define SLIDE_STEP			( 15 )
#define SVM_IMG_SIZE		( 16 )

// Define macro
#define DECRE_SAT(x, y)		( x -= (unsigned char)(x > y) )
#define INCRE_SAT(x, y)		( x += (unsigned char)(x < y) )



class CarTracking
{
private:
	// LUT for the number of consecutive frames to
	// consider object is out of frame
	static int numOutFrs[6];

	SimpleBlobDetector::Params params;

	// Object candidate
	struct ObjCand
	{
		// Initialize value for ObjCand
		ObjCand() : inFilter(false),
					match(false),
					inFrs(0),
					outFrs(0),
					width(0),
					Pt(Point(0, 0)),
					lastBox(0) {}
		bool	inFilter;
		bool	match;			// 
		int		inFrs;			// Number of consecutive frame to include in the filter
		int		outFrs;			// Number of consecutive frame to exclude in the filter
		Point	Pt;				// The Position of ObjCand
		int		width;			// The approximate width of object
		Point	lowestPt;		// The lowest point of object
		int		lastBox;
	};

	struct CarKalmanFilter
	{
		KalmanFilter	KF;
		//Mat_<float>		measurement;
		//Mat_<float>		prediction;
		//Mat_<float>		estimated;
		Point			truePos;
		int				objCandIdx;
	};

	vector< Point > obj2KF;

	void addNewObjCand(Point newPt);

	void updateInObjCand(int idx, Point Pt);

	void updateOutObjCand(void);

	int estimateWidth(Mat* img, KeyPoint* kpt, Point* lowestPt);

	bool diffDis(Point pt1, Point pt2);

	double calcDis(Point pt1, Point pt2);

	double similarity(Mat* img, ObjCand* objC, KeyPoint* kpt);

	void addNewKF(int objCandIdx);

	int updateKFidx(int objCanIdx);

	/* Bounding contour boxes */
	vector< vector< Point > > contours;

	vector< Vec4i > hierarchy;

	vector< vector< Point > > contours_poly;

	

public:
	// Initialize CarTracking paramaters
	CarTracking(void);

	// LUT for the size of bounding box for objects
	static int boxSize[17];

	/* Simple blob detection */
	vector< ObjCand > objCands;

	vector< CarKalmanFilter > CarKFs;

	Ptr< FeatureDetector > blob_detector;

	vector< KeyPoint > homoKeypoints;

	vector< Point > origKeypoints;

	void detect_filter(Mat* img);

	/* Bounding contour boxes */
	vector< Rect > boundRect;

	void findBoundContourBox(Mat* img);
	
	/* Kalman filter */
	void updateKF(void);

	/* SVM */
	CarSVM carsvm;
	
	void cropBoundObj(Mat* src, Mat* dst, Mat* invH, Rect* carBox, int objCandIdx);

	bool carSVMpredict(Mat* img, Rect* carBox, double classType, const svm_model *carModel, int objCandIdx);
};


#endif /* __CAR_TRACKING_HPP__ */
