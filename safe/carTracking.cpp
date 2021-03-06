// FILE: carTracking.cpp

#include "carTracking.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

// LUT for the size of bounding box for objects
int CarTracking::boxSize[21] = { 45, 50, 55, 60, 65, 70, 75, 82, 89, 96, 103, 111, 119, 128, 137, 147, 157, 168, 179, 191, 203 };


CarTracking::CarTracking(void)
{
	// Intialize parameters for simple blob detection
	params.minThreshold = 40;
	params.maxThreshold = 60;
	params.thresholdStep = 5;

	params.minArea = MIN_BLOB_AREA;
	params.minInertiaRatio = 0.5;

	params.maxArea = MAX_BLOB_AREA;
	params.maxInertiaRatio = 1.0;

	params.filterByArea = true;
	params.filterByColor = false;
	params.filterByCircularity = false;
    params.filterByInertia = false;
    params.filterByConvexity = false;

	// set up and create the detector using the parameters
	blob_detector = new SimpleBlobDetector(params);
	blob_detector->create("SimpleBlob");
}

CarTracking::~CarTracking( void ) {
    delete blob_detector;
    blob_detector = 0;
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
void CarTracking::detect_filter(const Mat &img)
{
	homoKeypoints.clear();
	origKeypoints.clear();
	// Do simple blob detection
	blob_detector->detect(img, homoKeypoints);

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
					updateInObjCand(j, homoKeypoints[i].pt);
					
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
inline bool CarTracking::diffDis(Point2f pt1, Point2f pt2)
{
	return (fabs(pt1.x - pt2.x) < ERR_BOX_SIZE_PX) ? ((fabs( pt1.y - pt2.y) < ERR_BOX_SIZE_PX) ? true : false) : false;
}

inline double CarTracking::calcDis(Point2f pt1, Point2f pt2)
{
	return sqrt(pow((double)pt1.x - (double)pt2.x, 2) + pow((double)pt1.y - (double)pt2.y, 2));
}

/* This function will handle the events that there is an new blob detected.
 * The new blob detected will be push back in the object candidate vector 'objCands'
 * to pass through temporal filter.
 */
inline void CarTracking::addNewObjCand(Point2f newPt)
{
	ObjCand newObjCand;
	newObjCand.inFrs = 1;
	newObjCand.Pos = newPt;
    newObjCand.filterPos = newPt;
	newObjCand.match = true;

	// Push newObjCand into the vector
	objCands.push_back(newObjCand);
}

/* This function will update the number of consecutive appear frame of an 
 * object candidate. An object candidate will be passed to next execution 
 * step only when it appears long enough in the continue sequence of frames.
 */
inline void CarTracking::updateInObjCand(int idx, Point2f Pt)
{
	objCands[idx].Pos = Pt;
	objCands[idx].match = true;

	// Update the number of consecutive appeared frames
	++objCands[idx].inFrs;

	// If the object is in Kalman filter, update the Kalman filter.
	if (objCands[idx].inFilter)
	{
		// This function will calculate velocity from position then 
		// pass the velocity through veloKF.
		filterVelo(objCands[idx]);
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
            initVeloKF(objCands[idx]);
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
		// An object is not match, it means that this object is not close
		// to any old object
		if (!objCands[i].match)
		{
			if (objCands[i].inFilter)
			{
				// Predict veloKF
				Mat_<float> prediction;
				prediction = objCands[i].veloKF.predict();

                objCands[i].prev_filterPos = objCands[i].filterPos;
                objCands[i].filterVelo = Point2f(prediction.at<float>(0), prediction.at<float>(1));
			}

			objCands[i].inFrs -= 1;
			// If this objCand dose not show up in a long enough time,
			// erase this objCand in objCands vector and erase the KF
			// element
			if (++objCands[i].outFrs > NUM_OUT_FRAMES)
			{				
				objCands.erase(objCands.begin() + i);
			}	
		}
	}
}

/* This function will convert to coordinate of a point from top left original 
 * to bottom middle point.
 */
void CarTracking::cvtCoord(const Point2f &orig, Point2f &cvt, const Mat &img)
{
	cvt.x = orig.x - ( img.cols / 2.0 );
	cvt.y = img.rows - orig.y;
}

/* ---------------------------------------------------------------------------------
*								BOUNDING BOXES
* --------------------------------------------------------------------------------*/
/* This function will find the bounding contour boxes. The bounding boxes
 * is contained in the vector 'boundRect'.
 * NOTE: the function only bounds blobs whose sizes are greater than MIN_BLOB_AREA
 * and less than MAX_BLOB_AREA
 */
void CarTracking::findBoundContourBox(const Mat &img)
{
	Mat tmp = img.clone();
	/// Find contours
	findContours(tmp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	contours_poly.resize(contours.size());
	boundRect.resize(contours.size());

	for (unsigned int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}

	// Erase vector element that its size is less than 'MIN_BLOB_AREA'
	vector< Rect >::iterator it;
	for (it = boundRect.begin(); it != boundRect.end();)
	{
		if ( (it->area() < MIN_BLOB_AREA) && (it->area() < MAX_BLOB_AREA))
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
		0, 0, 0, 0, 0.005f, 0,
		0, 0, 0, 0, 0, 0.005f);

	objCands[objCandIdx].EKF.measurementNoiseCov = (Mat_<float>(EKF_MEAS, EKF_MEAS) << 
		200.0f, 50.0f,
		50.0f, 200.0f);

	// Initialize A
	objCands[objCandIdx].EKF.calJacobian();

	setIdentity(objCands[objCandIdx].EKF.errorCovPost, Scalar::all(0.1));
}

void CarTracking::initVeloKF(ObjCand &obj) {
    // Setup the velocity kalman filter

    obj.veloKF.statePre.at<float>(0) = 416/2; // From homog frame size
    obj.veloKF.statePre.at<float>(1) = 480/2;
    obj.veloKF.statePre.at<float>(2) = 0;
    obj.veloKF.statePre.at<float>(3) = 0;

    float dt = 1.0 / 30.0;
    obj.veloKF.transitionMatrix = *(Mat_<float>(4, 4) <<
                    1,      0,      dt,     0,
                    0,      1,      0,      dt,
                    0,      0,      1,      0,
                    0,      0,      0,      1);

    setIdentity(obj.veloKF.measurementMatrix);

    obj.veloKF.processNoiseCov = *(Mat_<float>(4, 4) <<
        pow((float)dt, 4)/4.0,    0,                      pow((float)dt, 3)/3.0,    0,
        0,                      pow((float)dt, 4)/4.0,    0,                      pow((float)dt, 3)/3.0,
        pow((float)dt, 3)/3.0,    0,                      pow((float)dt, 2)/2.0,    0,
        0,                      pow((float)dt, 3)/3.0,    0,                      pow((float)dt, 2)/2.0);

    float meas_noise    = 0.1;
    float process_noise = 0.005;
    obj.veloKF.processNoiseCov = obj.veloKF.processNoiseCov*( process_noise*process_noise );
    setIdentity( obj.veloKF.measurementNoiseCov, Scalar::all( meas_noise*meas_noise ) );
    setIdentity( obj.veloKF.errorCovPost, Scalar::all(0.1));
}

void CarTracking::filterVelo(ObjCand &obj)
{
	// Predict the next velocity
    obj.veloKF.predict();

	// Update the new measurement values
	Mat_<float> measurement(2, 1);

	// As for a new frame, the filterPos now is prev_filterPos
	obj.prev_filterPos = obj.filterPos;
	obj.filterPos = obj.Pos;

    Point2f posdelta = obj.filterPos - obj.prev_filterPos;
    // V = v + at, t = 1/fps, or s/f = 1/30, so 8m/s^s * 1/30s / MPP
    if ( fabs( posdelta.x - obj.prev_posDelta.x ) > ( ( 8 / 30 ) ) / 0.0802105 )
                measurement(0) = obj.prev_posDelta.x;
    else        measurement(0) = posdelta.x;
    if ( fabs( posdelta.y - obj.prev_posDelta.y ) > ( ( 8 / 30 ) ) / 0.0802105 )
                measurement(1) = obj.prev_posDelta.y;
    else        measurement(1) = posdelta.y;
    obj.prev_posDelta = posdelta;
    
    // Estimate
	Mat_<float> estimated;
    estimated = obj.veloKF.correct(measurement);
	obj.filterVelo = Point2f(estimated.at<float>(0), estimated.at<float>(1));	
}

/* ---------------------------------------------------------------------------------
 *								POSITION
 * --------------------------------------------------------------------------------*/
 void CarTracking::importPos(string inputFileName, int lineNumberSought, Point2f &pos, Point2f &velo)
 {
 	string line, csvItem;
    ifstream myfile (inputFileName.c_str());
    int lineNumber = 0;
    if (myfile.is_open()) {
        while (getline(myfile,line)) {
            lineNumber++;
            if(lineNumber == lineNumberSought) {
                istringstream myline(line);
                if (getline(myline, csvItem, ',')) {
                	pos.x = (float)atof(csvItem.c_str());
                }
                if(getline(myline, csvItem, ',')) {
	                pos.y = (float)atof(csvItem.c_str());
                }
                if(getline(myline, csvItem, ',')) {
	                velo.y = (float)atof(csvItem.c_str());
                }
            }
        }
        myfile.close();
    }
 }
 
 void CarTracking::exportPos(string outputFileName, const Point2f &pos, const Point2f &velo)
 {
 	ofstream fout(outputFileName.c_str(), fstream::app);
 	fout << pos.x << "," << pos.y << "," << velo.y << "," << endl;
 	fout.close();
 }
 
 


