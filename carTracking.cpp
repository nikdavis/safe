#include "carTracking.hpp"

using namespace cv;
using namespace std;


// LUT for the number of consecutive frames to
// consider object is out of frame
int CarTracking::numOutFrs[6] = { 10, 15, 25, 35, 45, 60 };

// LUT for the size of bounding box for objects
int CarTracking::boxSize[17] = { 45, 50, 55, 60, 65, 70, 75, 82, 89, 96, 111, 108, 115, 124, 133, 144, 155 };


CarTracking::CarTracking(void)
{
	// Intialize parameters for simple blob detection
	params.minThreshold = 40;
	params.maxThreshold = 60;
	params.thresholdStep = 5;

	params.minArea = MIN_BLOB_AREA;
	params.minConvexity = 0.3f;			// ????
	params.minInertiaRatio = 0.01f;		// ????

	params.maxArea = MAX_BLOB_AREA;
	params.maxConvexity = 10.0f;			// ????

	params.filterByArea = true;
	params.filterByColor = false;
	params.filterByCircularity = false;

	// set up and create the detector using the parameters
	blob_detector = new cv::SimpleBlobDetector(params);
	blob_detector->create("SimpleBlob");
	
	objCands.reserve(100);
	objCands.resize(100);
}

/* ---------------------------------------------------------------------------------
*							SIMPLE BLOB DETECTION
* --------------------------------------------------------------------------------*/

/* This function will do simple blob detection (only blob whose area is
 * greater than 'MIN_BLOB_AREA' and smaller than 'MAX_BLOB_AREA' are selected)
 * Then, the center points of blobs will be passed through temporal filter
 * to filter out noisy detection. Only blob need to appear in 'NUM_IN_FRAMES'
 * consecutive frames to be considered as objects.
 */
void CarTracking::detect_filter(Mat* img)
{
	if (img->type() != CV_8UC1)
		img->convertTo(*img, CV_8UC1);

	homoKeypoints.clear();
	origKeypoints.clear();
	// Do simple blob detection
	blob_detector->detect(*img, homoKeypoints);

	// Matching keypoints with point of objCands
	unsigned int objCandsSize = objCands.size();
	for (unsigned int j = 0; j < objCandsSize; j++)
	{
		objCands[j].match = false;
	}

	for (unsigned int i = 0; i < homoKeypoints.size(); i++)
	{
		bool newPoint = true;
		for (unsigned int j = 0; j < objCandsSize; j++)
		{
			// If the objCand is already match with a homoKeypoint, skip this objCand
			if (!objCands[j].match)
			{
				// Calculate the distance from keypoints to the previous position of obj Candidates
				// If the distance is small enough, the keypoint is considered as new position of obj candidate
				if (diffDis(homoKeypoints[i].pt, objCands[j].Pos))
				{
					// Update state of objCand element
					updateInObjCand(j, homoKeypoints[i].pt, img);
					
					// If the keypoint is already in objCands vector
					newPoint = false;

					// Skip the rest objCand when a homoKeypoint match a objCand
					break;
				}
			}
		}
		// If there is a new keypoints, add this keypoints into objCands vector
		if (newPoint)
			addNewObjCand(homoKeypoints[i].pt);
	}
	updateOutObjCand();
}

/* This function determines whether or not a center point of blob belongs to 
 * an object candidate. It returns TRUE if the center point is within a certain
 * box of object candidate center point.
 */
inline bool CarTracking::diffDis(Point pt1, Point pt2)
{
	return (abs((int)pt1.x - (int)pt2.x) < ERR_BOX_SIZE_PX) ? ((abs((int)pt1.y - (int)pt2.y) < ERR_BOX_SIZE_PX) ? true : false) : false;
}

inline double CarTracking::calcDis(Point pt1, Point pt2)
{
	return sqrt(pow((double)pt1.x - (double)pt2.x, 2) + pow((double)pt1.y - (double)pt2.y, 2));
}

/* This function will handle the events that there is an new blob detected.
 * The new blob detected will be push back in the object candidate vector 'objCands'
 * to pass through temporal filter.
 */
inline void CarTracking::addNewObjCand(Point newPt)
{
	ObjCand newObjCand;
	newObjCand.inFrs = 1;
	newObjCand.Pos = newPt;
	newObjCand.match = true;

	// Push newObjCand into the vector
	objCands.push_back(newObjCand);
}

/* This function will update the number of consecutive appear frame of an 
 * object candidate. An object candidate will be passed to next execution 
 * step only when it appears long enough in the continue sequence of frames.
 */
inline void CarTracking::updateInObjCand(int idx, Point Pt, Mat* img)
{
	objCands[idx].Pos = Pt;
	objCands[idx].match = true;

	// Update the number of consecutive appeared frames
	++objCands[idx].inFrs;

	// If the object is in Kalman filter, update the Kalman filter.
	if (objCands[idx].inFilter)
	{
		// Predict the next position
		objCands[idx].EKF.predict();

		// Update the new measurement values
		Mat_<float> measurement(2, 1);
		measurement(0) = (float)objCands[idx].Pos.x;
		measurement(1) = (float)objCands[idx].Pos.y;

		// Estimate the new position
		Mat_<float> estimated;
		estimated = objCands[idx].EKF.correct(measurement);

		// Convert position from Mat format to Point format
		objCands[idx].filterPos = Point((int)estimated.at<float>(0), (int)estimated.at<float>(1));
		//cout << "x: " << estimated.at<float>(0) << "-y: " << estimated.at<float>(1) << "-theta: " << estimated.at<float>(2);
		//cout << "-v: " << estimated.at<float>(3) << "-phi: " << estimated.at<float>(4) << "-a: " << estimated.at<float>(5) << endl;
		
		fittingLine(idx);
		cout << "vx: " << objCands[idx].direction(0) << " - vy: " << objCands[idx].direction(1) << " - x0: " << objCands[idx].direction(2) << " - y0: " << objCands[idx].direction(3) << endl;
	}
	else	// If an object is not appeared long enough.
	{
		// If the object just appear again, clear the value of the number of
		// continous frames object disappear.
		objCands[idx].outFrs = 0;
		// If an object appears long enough, appear in at least 'NUM_IN_FRAMES'
		// consecutive frames. At it to the Kalman filter.
		if (objCands[idx].inFrs > NUM_IN_FRAMES)
		{
			// add this object to the Kalman filter and initalize Kalman filter for this object
			objCands[idx].inFilter = true;

			initExtendKalman(idx);
		}
	}
}

/* This function will handle the events that an object candidate, which
 * is already in the filter, does not show up in many consecutive frames.
 * This object candidate will be erased from object candidate vector and
 * not passed to next execution step.
 */
inline void CarTracking::updateOutObjCand(void)
{
	for (unsigned int i = 0; i < objCands.size(); i++)
	{
		// an object is not match, it means that this object is not close
		// to any old object
		if (!objCands[i].match)
		{
			// Even though the object does not appear in some frames, it is still need
			// to update the Kalman filter if it is still in the filter.
			// In this case, there is no measurement data. Therefore, there is no correction
			// step, and the filter position is the predict position.
			if (objCands[i].inFilter)
			{
				// Predict the next position
				Mat_<float> prediction;
				prediction = objCands[i].EKF.predict();

				// Convert position from Mat format to Point format
				objCands[i].filterPos = Point((int)prediction.at<float>(0), (int)prediction.at<float>(1));
				
				fittingLine(i);
				cout << "vx: " << objCands[i].direction(0) << " - vy: " << objCands[i].direction(1) << " - x0: " << objCands[i].direction(2) << " - y0: " << objCands[i].direction(3) << endl;
			}

			objCands[i].inFrs -= 1;
			// If this objCand dose not show up in a long enough time,
			// erase this objCand in objCands vector and erase the KF
			// element
			int numOutFrsIdx = (objCands[i].inFrs > 75) ? 5 : (objCands[i].inFrs / 15);
			if (++objCands[i].outFrs > numOutFrs[numOutFrsIdx])
			{				
				objCands.erase(objCands.begin() + i);
			}	
		}
	}
}

inline void  CarTracking::fittingLine(int idx)
{
	objCands[idx].posList.push_back(objCands[idx].filterPos);
	if (objCands[idx].posList.size() > POS_LIST_LENGTH)
		objCands[idx].posList.erase(objCands[idx].posList.begin());

	fitLine(Mat(objCands[idx].posList), objCands[idx].direction, CV_DIST_L2, 0, 0.01, 0.01);
}

void CarTracking::cvtCoord(Point* orig, Point2d* cvt, Mat* img)
{
	cvt->x = (double)((orig->x - img->cols / 2)*PX_METER_SCALE);
	cvt->y = (double)((img->rows - orig->y)*PX_METER_SCALE);
}

/* ---------------------------------------------------------------------------------
*								BOUNDING BOXES
* --------------------------------------------------------------------------------*/
/* This function will find the bounding contour boxes. The bounding boxes
 * is contained in the vector 'boundRect'.
 * NOTE: the function only return the boxes whose sizes are greater than
 * 'MIN_BOUND_BOX_EREA'
 */
void CarTracking::findBoundContourBox(Mat* img)
{
	Mat tmp = img->clone();
	/// Find contours
	findContours(tmp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	contours_poly.resize(contours.size());
	boundRect.resize(contours.size());

	for (unsigned int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}

	// Erase vector element that its size is less than 'MIN_BLOB_EREA'
	vector< Rect >::iterator it;
	for (it = boundRect.begin(); it != boundRect.end();)
	{
		if (it->area() < MIN_BOUND_BOX_EREA)
			it = boundRect.erase(it);
		else
			it++;
	}
}

/* ---------------------------------------------------------------------------------
 *								KALMAN FILTER
 * --------------------------------------------------------------------------------*/
inline void CarTracking::initExtendKalman(int objCandIdx)
{
	objCands[objCandIdx].EKF.statePre.at<float>(0) = (float)objCands[objCandIdx].Pos.x;
	objCands[objCandIdx].EKF.statePre.at<float>(1) = (float)objCands[objCandIdx].Pos.y;
	objCands[objCandIdx].EKF.statePre.at<float>(2) = (float)CV_PI / 2;
	objCands[objCandIdx].EKF.statePre.at<float>(3) = 0.0f;
	objCands[objCandIdx].EKF.statePre.at<float>(4) = 0.0f;
	objCands[objCandIdx].EKF.statePre.at<float>(5) = 0.0f;
	
	objCands[objCandIdx].EKF.statePost.at<float>(0) = (float)objCands[objCandIdx].Pos.x;
	objCands[objCandIdx].EKF.statePost.at<float>(1) = (float)objCands[objCandIdx].Pos.y;
	objCands[objCandIdx].EKF.statePost.at<float>(2) = (float)CV_PI / 2;
	objCands[objCandIdx].EKF.statePost.at<float>(3) = 0.0f;
	objCands[objCandIdx].EKF.statePost.at<float>(4) = 0.0f;
	objCands[objCandIdx].EKF.statePost.at<float>(5) = 0.0f;

	// 
	setIdentity(objCands[objCandIdx].EKF.measurementMatrix);

	// The process noise (wf, wa) is only dependent on the two driving processes.
	objCands[objCandIdx].EKF.processNoiseCov = (Mat_<float>(EKF_DYNAMIC, EKF_DYNAMIC) << 
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 1.0f, 0,
		0, 0, 0, 0, 0, 1.0f);

	objCands[objCandIdx].EKF.measurementNoiseCov = (Mat_<float>(EKF_MEAS, EKF_MEAS) << 
		200.0f, 0.0f,
		0.0f, 200.0f);

	//objCands[objCandIdx].EKF.measurementNoiseCov = objCands[objCandIdx].EKF.measurementNoiseCov * 200;
	// Initialize A
	objCands[objCandIdx].EKF.calJacobian();

	setIdentity(objCands[objCandIdx].EKF.errorCovPost, Scalar::all(0.1));
}


