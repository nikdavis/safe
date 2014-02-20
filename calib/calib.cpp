
#include "calib.hpp"

using namespace cv;
using namespace std;



int CameraCalibrator::CaddCalibrationData(string imgPath, Size& checkerboardSize)
{
	// Count the number of successfully loaded checkerboard Images
	int successes = 0;
	// Allocate the vector for object points (3D points) and image points (2D points) for each image
	vector<Point2f> imageCorners;
	vector<Point3f> objectCorners;

	VideoCapture sequence(imgPath);
	Mat src;
	cout << "Loading checkerboard image..." << endl;
	for (;;)
	{
		
		sequence >> src;
		if(src.empty())
		{
			cout << "End of Sequence" << endl;
			break;
		}
		//Mat dispImg = src.clone();

		// Finds the positions of internal corners of the chessboard.
		bool patternfound = findChessboardCorners(src, checkerboardSize, imageCorners);

		// refine the corner's locations: the function "findChessboardCorners" may not give good enough coordinates for corners
		if(patternfound)
			cornerSubPix(src, imageCorners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		// Plot the corner to the source image for visualization
		//drawChessboardCorners(dispImg, checkerboardSize, imageCorners, patternfound);
		//imshow("Source", dispImg);

		// Put 3D points and 2D points into the vectors for calibration
		if (imageCorners.size() == checkerboardSize.area())
		{
			for(int i = 0; i < checkerboardSize.height; i++)
			{
				for(int j = 0; j < checkerboardSize.width; j++)
				{
					objectCorners.push_back(cv::Point3f(float(i)*squareLength, float(j)*squareLength, 0.0f));
				}
			}
			//2D image point from one view
			imagePoints.push_back(imageCorners);
			//corresponding 3D scene points
			objectPoints.push_back(objectCorners);
			objectCorners.clear();
			//imshow("Source", dispImg);
		}
		//waitKey(500);
		successes++;
	}
	return successes;
}

// Calibrate the camera to find instrinsic matrix of camera and distortion coeffecients.
// This function return the re-projection error.
double CameraCalibrator::doCalibrate(Size& imageSize)		
{
	cout << "Calibrating... " << endl;
	double reprojErr = calibrateCamera(	objectPoints, //the 3D points
										imagePoints,
										imageSize, 
										cameraMatrix, //output camera matrix
										distCoeffs,
										rvecs,tvecs,
										flag);
	cout << "Done" << endl;
	return reprojErr;
}

// Get the mapping matrices from distorted image to undistorted image
// image: the distorted image
void CameraCalibrator::getUndistorMapping(const Mat image)
{
	initUndistortRectifyMap(cameraMatrix, 
							distCoeffs, 
							Mat(),
							getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, image.size(), 1, image.size(), 0),
							image.size(), 
							CV_16SC2, 
							map1, map2);
}

// Get the mapping matrices from distorted image to undistorted image
// imageSize: the size of distorted image. This value is set to be (640x480) as default. 
void CameraCalibrator::getUndistorMapping(Size& imageSize)
{
	initUndistortRectifyMap(cameraMatrix, 
							distCoeffs, 
							Mat(),
							getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
							imageSize, 
							CV_16SC2, 
							map1, map2);
}

// Write Camera parameters to .xml file to use it later.
// + outputFileName: the .xml file name.
// + imgPath: the path to the first image. This is used to extract data from images.
void CameraCalibrator::writeCameraParams(string outputFileName, string imgPath) const                        //Write serialization for this class
{
	FileStorage fs( outputFileName, FileStorage::WRITE );
	VideoCapture sequence(imgPath);
	Mat src;
	int numOfImage = 0;
	for (;;)
	{
		sequence >> src;
		if(src.empty())
			break;
		numOfImage++;
	}
	src = imread(imgPath, CV_LOAD_IMAGE_GRAYSCALE);
    fs	<< "BoardSize_Width"		<< checkerboardSize.width
		<< "BoardSize_Height"		<< checkerboardSize.height
		<< "Square_Size"			<< squareLength
		<< "Calibrate_Pattern"		<< patternToUse
		<< "Number_Of_Image"		<< numOfImage
		<< "Image_Width"			<< src.cols
		<< "Image_Height"			<< src.rows
		<< "Camera_Matrix"			<< cameraMatrix			 
		<< "Distortion_coeffecient" << distCoeffs
		<< "Flag"					<< flag;
	fs.release(); 
}

// Read camera parameters from .xml file.
void CameraCalibrator::readCameraParams(string inputFileName)                          //Read serialization for this class
{
	FileStorage fs( inputFileName, FileStorage::READ );
    /*fs["BoardSize_Width" ]			>> checkerboardSize.width;
    fs["BoardSize_Height"]			>> checkerboardSize.height;
	fs["Square_Size"]				>> squareLength;
    fs["Calibrate_Pattern"]			>> patternToUse;*/
	fs["Camera_Matrix"]				>> cameraMatrix;
	fs["Distortion_coeffecient"]	>> distCoeffs;
	fs.release();
}
};

