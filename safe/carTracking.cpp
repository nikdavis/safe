// FILE: carTracking.cpp

#include "carTracking.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

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
		// Predict the next position
		objCands[idx].EKF.predict();
        objCands[idx].veloKF.predict();

		// Update the new measurement values
		Mat_<float> measurement(2, 1);
		measurement(0) = (float)objCands[idx].Pos.x;
		measurement(1) = (float)objCands[idx].Pos.y;

		// Estimate the new position
		Mat_<float> estimated;
		estimated = objCands[idx].EKF.correct(measurement);

		// Convert position from Mat format to Point format
		objCands[idx].prev_filterPos = objCands[idx].filterPos;
		objCands[idx].filterPos = Point2f(measurement.at<float>(0), measurement.at<float>(1));

        Point2f posdelta = objCands[idx].filterPos - objCands[idx].prev_filterPos;
        // V = v + at, t = 1/fps, or s/f = 1/30, so 8m/s^s * 1/30s / MPP
        if ( fabs( posdelta.x - objCands[idx].prev_posDelta.x ) > ( ( 8 / 30 ) ) / 0.0802105 )
                    measurement(0) = objCands[idx].prev_posDelta.x;
        else        measurement(0) = posdelta.x;
        if ( fabs( posdelta.y - objCands[idx].prev_posDelta.y ) > ( ( 8 / 30 ) ) / 0.0802105 )
                    measurement(1) = objCands[idx].prev_posDelta.y;
        else        measurement(1) = posdelta.y;
        objCands[idx].prev_posDelta = posdelta;
        estimated = objCands[idx].veloKF.correct(measurement);
		objCands[idx].filterVelo = Point2f(estimated.at<float>(0), estimated.at<float>(1));

		//fittingLine(idx);
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
            initVeloKF(idx);
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
			//
			// Even though the object does not appear in some frames, it is still need
			// to update the Kalman filter if it is still in the filter.
			// In this case, there is no measurement data. Therefore, there is no correction
			// step, and the filter position is the predict position.
			//
			// For the KF for velocity, this KF measurement values are fed from the differential 
			// positions in EKF. Therefore, the KF for velocity always has measurement values.
			//
			if (objCands[i].inFilter)
			{
				// Predict the next position
				Mat_<float> prediction;
				prediction = objCands[i].EKF.predict();

				// Convert position from Mat format to Point format
				//objCands[i].filterPos = Point2f(prediction.at<float>(0), prediction.at<float>(1));
				
				// Update for veloKF
				prediction = objCands[i].veloKF.predict();

                objCands[i].prev_filterPos = objCands[i].filterPos;
                objCands[i].filterVelo = Point2f(prediction.at<float>(0), prediction.at<float>(1));
				
				//Mat_<float> measurement(2, 1);
				//objCands[i].prev_filterPos = objCands[i].filterPos;
				//Point posdelta = objCands[i].filterPos - objCands[i].prev_filterPos;
        		//measurement(0) = posdelta.x;
        		//measurement(1) = posdelta.y;
        		
        		//Mat_<float> estimated;
        		//estimated = objCands[i].veloKF.correct(measurement);
				//objCands[i].filterVelo = Point2f(estimated.at<float>(0), estimated.at<float>(1));
				
				//fittingLine(i);
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

inline void  CarTracking::fittingLine(int idx)
{
	objCands[idx].frCount++;
	if ((objCands[idx].frCount % 6) == 0)
	{	
		Point p(objCands[idx].filterPos.x, objCands[idx].filterPos.y + objCands[idx].frCount * 5);
		//objCands[idx].posList.push_back(objCands[idx].filterPos);
		
		objCands[idx].posList.push_back(p);
		if (objCands[idx].posList.size() > POS_LIST_LENGTH)
			objCands[idx].posList.erase(objCands[idx].posList.begin());
		else if (objCands[idx].posList.size() == 1)
		{
			Point p1(objCands[idx].filterPos.x, objCands[idx].filterPos.y + objCands[idx].frCount * 5 + 10);
			objCands[idx].posList.push_back(p1);
		}
		
		fitLine(objCands[idx].posList, objCands[idx].direction, CV_DIST_L2, 0, 0.01, 0.01);
		
		/*if (idx == 0)
		{
			ofstream fout;
			//fout.open("position.csv");
			fout.open("position.csv", fstream::app);
			//for (unsigned int i = 0; i < objCands[idx].posList.size(); i++)
			//{
			//	fout << objCands[idx].posList.at(i).x << "," << objCands[idx].posList.at(i).y << endl;
			//}
			fout << objCands[idx].Pos.x << "," << objCands[idx].Pos.y << ",";
			fout << objCands[idx].filterPos.x << "," << objCands[idx].filterPos.y << ",";
			fout << objCands[idx].direction(0) << "," << objCands[idx].direction(1) << endl;
			fout.close();
		}*/
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

void CarTracking::calAngle(const Point2f &carPos, const Mat &img, Point2f &normVxy)
{
	float Vx = (float)((img.cols / 2) - carPos.x);
	float Vy = (float)(img.rows - carPos.y);

	if ((Vx == 0) && (Vy == 0))
	{
		normVxy.x = 0;
		normVxy.y = 0;
	}
	else
	{
		normVxy.x = Vx / sqrt(powf(Vx, 2) + powf(Vy, 2));
		normVxy.y = Vy / sqrt(powf(Vx, 2) + powf(Vy, 2));
	}
}

/* ---------------------------------------------------------------------------------
*								BOUNDING BOXES
* --------------------------------------------------------------------------------*/
/* This function will find the bounding contour boxes. The bounding boxes
 * is contained in the vector 'boundRect'.
 * NOTE: the function only bounds blobs whose sizes are greater than MIN_BOUND_BOX_EREA'
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

	// Erase vector element that its size is less than 'MIN_BOUND_BOX_EREA'
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
		0, 0, 0, 0, 0.005f, 0,
		0, 0, 0, 0, 0, 0.005f);

	objCands[objCandIdx].EKF.measurementNoiseCov = (Mat_<float>(EKF_MEAS, EKF_MEAS) << 
		200.0f, 50.0f,
		50.0f, 200.0f);

	// Initialize A
	objCands[objCandIdx].EKF.calJacobian();

	setIdentity(objCands[objCandIdx].EKF.errorCovPost, Scalar::all(0.1));
}

inline void CarTracking::initVeloKF(int objCandIdx) {
    // Setup the velocity kalman filter

    objCands[objCandIdx].veloKF.statePre.at<float>(0) = 416/2; // From homog frame size
    objCands[objCandIdx].veloKF.statePre.at<float>(1) = 480/2;
    objCands[objCandIdx].veloKF.statePre.at<float>(2) = 0;
    objCands[objCandIdx].veloKF.statePre.at<float>(3) = 0;

    float dt = 1.0 / 30.0;
    objCands[objCandIdx].veloKF.transitionMatrix = *(Mat_<float>(4, 4) <<
                    1,      0,      dt,     0,
                    0,      1,      0,      dt,
                    0,      0,      1,      0,
                    0,      0,      0,      1);

    setIdentity(objCands[objCandIdx].veloKF.measurementMatrix);

    objCands[objCandIdx].veloKF.processNoiseCov = *(Mat_<float>(4, 4) <<
        pow((float)dt, 4)/4.0,    0,                      pow((float)dt, 3)/3.0,    0,
        0,                      pow((float)dt, 4)/4.0,    0,                      pow((float)dt, 3)/3.0,
        pow((float)dt, 3)/3.0,    0,                      pow((float)dt, 2)/2.0,    0,
        0,                      pow((float)dt, 3)/3.0,    0,                      pow((float)dt, 2)/2.0);

    float meas_noise    = 0.1;
    float process_noise = 0.005;
    objCands[objCandIdx].veloKF.processNoiseCov = objCands[objCandIdx].veloKF.processNoiseCov*( process_noise*process_noise );
    setIdentity( objCands[objCandIdx].veloKF.measurementNoiseCov, Scalar::all( meas_noise*meas_noise ) );
    setIdentity( objCands[objCandIdx].veloKF.errorCovPost, Scalar::all(0.1));
}



