#include "carTracking.hpp"

using namespace cv;
using namespace std;


// LUT for the number of consecutive frames to
// consider object is out of frame
int CarTracking::numOutFrs[6] = { 10, 15, 25, 35, 45, 60 };

// LUT for the size of bounding box for objects
int CarTracking::boxSize[17] = { 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125 };


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
				if (diffDis(homoKeypoints[i].pt, objCands[j].Pt))
				{
					// Update state of objCand element
					updateInObjCand(j, homoKeypoints[i].pt);
					//cout << "objCand: " << j << " - Homo: " << i << endl;
					int res = estimateWidth(img, &homoKeypoints[i], &objCands[j].lowestPt);
					if (!res)
						objCands[j].width = res;
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

	//int height = (int)(kpt->size)*(int)(kpt->size) / width;
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
	newObjCand.Pt = newPt;
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
	objCands[idx].Pt = Pt;
	objCands[idx].match = true;

	// Update the number of consecutive appeared frames
	++objCands[idx].inFrs;
	if (!objCands[idx].inFilter)
	{
		objCands[idx].inFilter = false;
		objCands[idx].outFrs = 0;
		//objCands[idx].outCons = false;
		if (objCands[idx].inFrs > NUM_IN_FRAMES)
		{
			//obj2KF.push_back(Point(idx, CarKFs.size()));
			objCands[idx].inFilter = true;
			//cout << "I am here" << endl;
			//addNewKF(idx);	
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
		if (!objCands[i].match)
		{
			objCands[i].inFrs -= 1;
			// If this objCand dose not show up in a long enough time,
			// erase this objCand in objCands vector and erase the KF
			// element
			int numOutFrsIdx = (objCands[i].inFrs > 75) ? 5 : (objCands[i].inFrs / 15);
			if (++objCands[i].outFrs > numOutFrs[numOutFrsIdx])
			{
				//CarKFs.erase(CarKFs.begin() + i);
				/*if (objCands[i].inFilter)
				{
					int pos = updateKFidx(i);
					CarKFs.erase(CarKFs.begin() + pos);
				}*/
				
				objCands.erase(objCands.begin() + i);
			}
		}
	}
}


/* ---------------------------------------------------------------------------------
*								BOUNDING BOXES
* --------------------------------------------------------------------------------*/
/* This function will find the bounding contour boxes. The bounding boxes
 * is contained in the vector 'boundRect'
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
 

void CarTracking::addNewKF(int objCandIdx)
{
	CarKalmanFilter carKF;

	carKF.KF.init(4, 2, 0);				/* 4 dynamic parameters (x, y, Vx, Vy), 2 measurement parameters (x, y), and no control */

	carKF.objCandIdx = objCandIdx;

	//
	// x_k = A*x_(k-1) + w_(k-1)
	// z_k = H*x_k + v_k
	//
	// NOTE:	A: transitionMatrix
	//			H: measurementMatrix
	//			

	carKF.KF.statePre.at<float>(0) = (float)objCands[carKF.objCandIdx].Pt.x;
	carKF.KF.statePre.at<float>(1) = (float)objCands[carKF.objCandIdx].Pt.y;
	carKF.KF.statePre.at<float>(2) = 0;
	carKF.KF.statePre.at<float>(3) = 0;

	// The system equation
	carKF.KF.transitionMatrix = *(Mat_<float>(4, 4) <<
		1, 0, dt, 0,
		0, 1, 0, dt,
		0, 0, 1, 0,
		0, 0, 0, 1);

	// The output matrix
	setIdentity(carKF.KF.measurementMatrix);

	//Processs Noise Covariance
	carKF.KF.processNoiseCov = *(Mat_<float>(4, 4) <<
		pow((float)dt, 4) / 4, 0, pow((float)dt, 3) / 3, 0,
		0, pow((float)dt, 4) / 4, 0, pow((float)dt, 3) / 3,
		pow((float)dt, 3) / 3, 0, pow((float)dt, 2) / 2, 0,
		0, pow((float)dt, 3) / 3, 0, pow((float)dt, 2) / 2);

	carKF.KF.processNoiseCov = carKF.KF.processNoiseCov*(PROCESS_NOISE*PROCESS_NOISE);

	//Measurement Noise Covariance
	setIdentity(carKF.KF.measurementNoiseCov, Scalar::all(MEAS_NOISE*MEAS_NOISE));
	setIdentity(carKF.KF.errorCovPost, Scalar::all(0.1));

	CarKFs.push_back(carKF);
}

void CarTracking::updateKF(void)
{
	cout << "KF size: " << CarKFs.size() << endl;
	Mat_<float> measurement(2, 1);
	Mat_<float> estimated(2, 1);
	for (unsigned int i = 0; i < CarKFs.size(); i++)
	{
		// Predict the next position
		CarKFs[i].KF.predict();

		// Update the new measurement values
		measurement(0) = (float)objCands[obj2KF[i].x].Pt.x;
		measurement(1) = (float)objCands[obj2KF[i].x].Pt.y;
		cout << "objCand: " << CarKFs[i].objCandIdx << " - KF: " << i << endl;
		// Update the new position
		estimated = CarKFs[i].KF.correct(measurement);

		// Convert position from Mat format to Point format
		CarKFs[i].truePos = Point(estimated.at<float>(0), estimated.at<float>(1));
		cout << "X: " << estimated.at<float>(0) << " - Y: " << estimated.at<float>(1) << endl;
	}
}

__inline int CarTracking::updateKFidx(int objCanIdx)
{
	int pos;
	for (unsigned int i = 0; i < obj2KF.size(); i++)
	{
		if (obj2KF[i].x == objCanIdx)
		{
			pos = i;
			break;
		}
	}

	for (unsigned int i = pos; i < obj2KF.size(); i++)
	{
		obj2KF[i].x -= 1;
		obj2KF[i].y -= 1;
	}

	obj2KF.erase(obj2KF.begin() + pos);

	return pos;
}

/* ---------------------------------------------------------------------------------
*								CAR SVM PREDICT
* --------------------------------------------------------------------------------*/

void CarTracking::cropBoundObj(Mat* src, Mat* dst, Mat* invH, Rect* carBox, int objCandIdx)
{
	Point p;
	// Convert point in homog image to point in original image
	pointHomogToPointOrig(invH, &objCands[objCandIdx].Pt, &p);

	// Select the high car probability area
	int idx = ((int)objCands[objCandIdx].Pt.y / 15);
	int l = boxSize[idx];
	int x = max(p.x - cvRound(1.5*l), 0);
	int y = max(p.y - cvRound(3 * l / 4 + 0.1*l), 0);
	int w = min(cvRound(3 * l), src->cols - x);
	int h = min(cvRound(1 * l), src->rows - y);
	Rect box(x, y, w, h);
	*carBox = box;
	// Crop the high car probability area 
	*dst = (*src)(box);
}

bool CarTracking::carSVMpredict(Mat* img, Rect* carBox, double classType, const svm_model *carModel, int objCandIdx)
{
	//int middle = cvRound(img->cols / 2) - objCands[objCandIdx].lastBox;
	int h = cvRound(img->rows / 2);
	//bool left_most = false, right_most = false;

	int x = objCands[objCandIdx].lastBox - objCands[objCandIdx].lastBox/3;
	while (true)
	{
		Rect box(x, 0, 2 * h, 2 * h);
		/*if (box.tl().x < 0) {
			left_most = true;
		}*/
			
		if (box.br().x > img->cols) {
			//right_most = true;
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
