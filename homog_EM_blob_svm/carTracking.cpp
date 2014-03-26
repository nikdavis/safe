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

	//updateKF();
}

/* This function determines whether or not a center point of blob belongs to 
 * an object candidate. It returns TRUE if the center point is within a certain
 * box of object candidate center point.
 */
__inline bool CarTracking::diffDis(Point pt1, Point pt2)
{
	return (abs((int)pt1.x - (int)pt2.x) < ERR_BOX_SIZE_PX) ? ((abs((int)pt1.y - (int)pt2.y) < ERR_BOX_SIZE_PX) ? true : false) : false;
}

__inline double CarTracking::calcDis(Point pt1, Point pt2)
{
	return sqrt(pow((double)pt1.x - (double)pt2.x, 2) + pow((double)pt1.y - (double)pt2.y, 2));
}

int CarTracking::estimateWidth(Mat* img, KeyPoint* kpt, Point* lowestPt)
{
	int row = (int)kpt->pt.y;
	int col = (int)kpt->pt.x;
	int gap = 0;
	int width = 0;
	while ((col < 350) && (!img->at<unsigned char>(row, col++))) {
		width++;
		if (++gap > MAX_GAP_PX)
		{
			width -= MAX_GAP_PX;
			break;
		}
	}
	while ((col < 350) && (img->at<unsigned char>(row, col++))) {
		width++;
	}
	gap = 0;
	while ((col < 350) && (!img->at<unsigned char>(row, col++))) {
		width++;
		if (++gap > MAX_GAP_PX)
		{
			width -= MAX_GAP_PX;
			break;
		}
	}
	while ((col < 350) && (img->at<unsigned char>(row, col++))) {
		width++;
	}

	col = (int)kpt->pt.x;
	gap = 0;
	while ((col < 0) && (!img->at<unsigned char>(row, col--))) {
		width++;
		if (++gap > MAX_GAP_PX)
		{
			width -= MAX_GAP_PX;
			break;
		}
	}
	while ((col < 0) && (img->at<unsigned char>(row, col--))) {
		width++;
	}
	gap = 0;
	while ((col < 0) && (!img->at<unsigned char>(row, col--))) {
		width++;
		if (++gap > MAX_GAP_PX)
		{
			width -= MAX_GAP_PX;
			break;
		}
	}
	while ((col < 0) && (img->at<unsigned char>(row, col--))) {
		width++;
	}
	
	// If the width is 0, there is an unexpected blob. So skip update and return
	if (!width)
		return 0;

	int height = (int)(kpt->size)*(int)(kpt->size) / width;
	//cout << "Size: " << (kpt->size) << " - width: " << width << " - height : " << height << endl;
	return width;
}

__inline double CarTracking::similarity(Mat* img, ObjCand* objC, KeyPoint* kpt)
{
	//double newWidth = estimateWidth(img, kpt, )
	return 0;
}

/* This function will handle the events that there is an new blob detected.
 * The new blob detected will be push back in the object candidate vector 'objCands'
 * to pass through temporal filter.
 */
__inline void CarTracking::addNewObjCand(Point newPt)
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
__inline void CarTracking::updateInObjCand(int idx, Point Pt)
{
	objCands[idx].Pos = Pt;
	objCands[idx].match = true;

	// Update the number of consecutive appeared frames
	++objCands[idx].inFrs;

	// If the object is in Kalman filter, update the Kalman filter.
	if (objCands[idx].inFilter)
	{
		Mat_<float> x_measurement(1, 1);
		Mat_<float> y_measurement(1, 1);
		Mat_<float> x_estimated;
		Mat_<float> y_estimated;

		// Predict the next position
		objCands[idx].KFx.predict();
		objCands[idx].KFy.predict();

		// Update the new measurement values
		x_measurement(0) = (float)objCands[idx].Pos.x;
		y_measurement(0) = (float)objCands[idx].Pos.y;

		// Update the new position
		x_estimated = objCands[idx].KFx.correct(x_measurement);
		y_estimated = objCands[idx].KFy.correct(y_measurement);

		// Convert position from Mat format to Point format
		objCands[idx].filterPos = Point((int)x_estimated.at<float>(0), (int)y_estimated.at<float>(0));
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
			initKalman(idx);
		}
	}
}

/* This function will handle the events that an object candidate, which
 * is already in the filter, does not show up in many consecutive frames.
 * This object candidate will be erased from object candidate vector and
 * not passed to next execution step.
 */
__inline void CarTracking::updateOutObjCand(void)
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
				Mat_<float> x_prediction;
				Mat_<float> y_prediction;

				// Predict the next position
				x_prediction = objCands[i].KFx.predict();
				y_prediction = objCands[i].KFy.predict();

				// Convert position from Mat format to Point format
				objCands[i].filterPos = Point((int)x_prediction.at<float>(0), (int)y_prediction.at<float>(0));
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
__inline void CarTracking::initKalman(int objCandIdx)
{
	objCands[objCandIdx].filterPos = objCands[objCandIdx].Pos;

	/* 6 dynamic parameters (x, y, Vx, Vy, Ax, Ay), 2 measurement parameters (x, y), and no control */

	// For x, Vx, Ax
	objCands[objCandIdx].KFx.init(3, 1, 0);

	objCands[objCandIdx].KFx.statePre.at<float>(0) = (float)objCands[objCandIdx].Pos.x;		// X
	objCands[objCandIdx].KFx.statePre.at<float>(1) = 0;										// Vx
	objCands[objCandIdx].KFx.statePre.at<float>(2) = 0;										// Ax

	objCands[objCandIdx].KFx.statePost.at<float>(0) = (float)objCands[objCandIdx].Pos.x;		// X
	objCands[objCandIdx].KFx.statePost.at<float>(1) = 0;										// Vx
	objCands[objCandIdx].KFx.statePost.at<float>(2) = 0;										// Ax

	// The system equation
	objCands[objCandIdx].KFx.transitionMatrix = *(Mat_<float>(3, 3) <<
		1, dt, dt*dt/2,	
		0, 1, dt,
		0, 0, 1);

	// The output matrix
	setIdentity(objCands[objCandIdx].KFx.measurementMatrix);

	//Processs Noise Covariance
	objCands[objCandIdx].KFx.processNoiseCov = *(Mat_<float>(3, 3) <<
		pow(dt, 4) / 4,	pow(dt, 3) / 3,	pow(dt, 2)/ 2,	
		pow(dt, 3) / 3,	pow(dt, 2) / 2,	dt,				
		pow(dt, 2) / 2,	dt,				1);

	objCands[objCandIdx].KFx.processNoiseCov = objCands[objCandIdx].KFx.processNoiseCov*(PROCESS_NOISE*PROCESS_NOISE);

	//Measurement Noise Covariance
	setIdentity(objCands[objCandIdx].KFx.measurementNoiseCov, Scalar::all(MEAS_NOISE*MEAS_NOISE));
	setIdentity(objCands[objCandIdx].KFx.errorCovPost, Scalar::all(0.1));

	// For y, Vy, Ay
	objCands[objCandIdx].KFy.init(3, 1, 0);

	objCands[objCandIdx].KFy.statePre.at<float>(0) = (float)objCands[objCandIdx].Pos.y;		// Y
	objCands[objCandIdx].KFy.statePre.at<float>(1) = 0;										// Vy
	objCands[objCandIdx].KFy.statePre.at<float>(2) = 0;										// Ay

	objCands[objCandIdx].KFy.statePost.at<float>(0) = (float)objCands[objCandIdx].Pos.y;	// Y
	objCands[objCandIdx].KFy.statePost.at<float>(1) = 0;									// Vy
	objCands[objCandIdx].KFy.statePost.at<float>(2) = 0;									// Ay

	// The system equation
	objCands[objCandIdx].KFy.transitionMatrix = *(Mat_<float>(3, 3) <<
		1, dt, dt*dt / 2,
		0, 1, dt,
		0, 0, 1);

	// The output matrix
	setIdentity(objCands[objCandIdx].KFy.measurementMatrix);

	//Processs Noise Covariance
	objCands[objCandIdx].KFy.processNoiseCov = *(Mat_<float>(3, 3) <<
		pow(dt, 4) / 4, pow(dt, 3) / 3, pow(dt, 2) / 2,
		pow(dt, 3) / 3, pow(dt, 2) / 2, dt,
		pow(dt, 2) / 2, dt, 1);

	objCands[objCandIdx].KFy.processNoiseCov = objCands[objCandIdx].KFy.processNoiseCov*(PROCESS_NOISE*PROCESS_NOISE);

	//Measurement Noise Covariance
	setIdentity(objCands[objCandIdx].KFy.measurementNoiseCov, Scalar::all(MEAS_NOISE*MEAS_NOISE));
	setIdentity(objCands[objCandIdx].KFy.errorCovPost, Scalar::all(0.1));
}

/* ---------------------------------------------------------------------------------
*								CAR SVM PREDICT
* --------------------------------------------------------------------------------*/

void CarTracking::boundBox(Mat* img, Point* p, Rect* box, int idx)
{
	// Select the high car probability area
	int l = boxSize[idx];
	box->x = max(p->x - cvRound(1*l), 0);
	box->y = max(p->y - cvRound(1*l), 0);
	box->width	= min(cvRound(2 * l), img->cols - box->x);
	box->height = min(cvRound(1.5 * l), img->rows - box->y);
}

void CarTracking::cropBoundObj(Mat* src, Mat* dst, Mat* invH, Rect* carBox, int objCandIdx)
{
	Point p;
	// Convert point in homog image to point in original image
	pointHomogToPointOrig(invH, &objCands[objCandIdx].Pos, &p);

	// Select the high car probability area
	int idx = ((int)objCands[objCandIdx].Pos.y / 15);
	int l = boxSize[idx];
	int x = max(p.x - l, 0);
	int y = max(p.y - l, 0);
	int w = min(2 * l, src->cols - x);
	int h = min(cvRound(1.5 * l), src->rows - y);
	Rect box(x, y, w, h);
	*carBox = box;

	// Crop the high car probability area 
	*dst = (*src)(box);
}

/* This function will run the sliding window through the box with high probability having car
 * and using svm double check.
 * NOTE: the way of sliding window will be developed later. Currently, I just use a square window slide 
 * from left to right.
 */
bool CarTracking::carSVMpredict(Mat* img, Rect* carBox, double classType, const svm_model *carModel, int objCandIdx)
{
	//int middle = cvRound(img->cols / 2) - objCands[objCandIdx].lastBox;
	int h = cvRound(img->rows / 2);
	bool left_most = false, right_most = false;

	int x = objCands[objCandIdx].lastBox - objCands[objCandIdx].lastBox/5;
	while (true)
	{
		Rect box(x, 0, 2 * h, 2 * h);
		/*if (box.tl().x < 0) {
			left_most = true;
		}*/
			
		if (box.br().x > img->cols) {
			return false;
		}

		/*if (left_most || right_most)
			continue;*/

		Mat cropImg = (*img)(box);
		if (cropImg.channels() == 3)
			cvtColor(cropImg, cropImg, CV_RGB2GRAY);
		else if (cropImg.channels() == 4)
			cvtColor(cropImg, cropImg, CV_RGBA2GRAY);

		resize(cropImg, cropImg, Size(SVM_IMG_SIZE, SVM_IMG_SIZE));

		if (cropImg.type() != CV_8UC1)
			cropImg.convertTo(cropImg, CV_8UC1);

		if (carsvm.predict(&cropImg, carsvm.POS, carModel))
		{
			*carBox = box;
			objCands[objCandIdx].lastBox = x;
			return true;
		}
		
		x += SLIDE_STEP;
	}
}

/* ---------------------------------------------------------------------------------
*								SOBEL
* --------------------------------------------------------------------------------*/
void CarTracking::doSobel(Mat* img, Mat* dst)
{
	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(*img, grad_x, CV_16S, 1, 0, 3);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(*img, grad_y, CV_16S, 0, 1, 3);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, *dst);
}


