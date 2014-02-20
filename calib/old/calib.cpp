#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <stdlib.h>
 
int n_boards = 0;	//Number of snapshots of the chessboard
int frame_step;		//Frames to be skipped
int board_w;		//Enclosed corners horizontally on the chessboard
int board_h;		//Enclosed corners vertically on the chessboard


using namespace cv;
using namespace std;


int main(int argc, char ** argv) {
	// Allocate the vector of vector for object points (3D points) and image points (2D points)
	vector<vector<Point3f> > objectPoints;
	vector<vector<Point2f> > imagePoints;
	Mat src;
	Size innerCorners = Size(8, 6);			// The Number of inner corners per a chessboard row and column			
	float squareLength = 26;				// Square Length
    int flag = 0;							//flag to specify how calibration is done
    int i = 0;
    int channels = 0;


	namedWindow("Source", CV_WINDOW_AUTOSIZE);
	namedWindow("Undistorted image", CV_WINDOW_AUTOSIZE);

    /* Load video capture of checkerboard sequence */
	cout << "Loading checkerboard image..." << endl;
	string pathToData("../images/checkerboard 1.pgm");
	VideoCapture sequence(pathToData);
    sequence >> src;
    channels = src.channels();
    cout << "Channels: " << channels << endl;
    if(channels > 1)
        cvtColor( src, src, CV_BGR2GRAY );

	vector<Point2f> imageCorners;
	vector<Point3f> objectCorners;
	cout << "Image size: " << src.size() << endl;
	if(src.empty())
    {
        cout << "End of Sequence" << endl;
		//waitKey(0);
        //break;
    }
	Mat dispImg = src.clone();


	// Check whether or not the input images are good for calibration
	// Finds the positions of internal corners of the chessboard.
	bool patternfound = findChessboardCorners(src, innerCorners, imageCorners);

	// refine the corner's locations: the function "findChessboardCorners" may not give good enough coordinates for corners
	if(patternfound) {
		cornerSubPix(src, imageCorners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
	    // Plot the corner to the source image for visualization
	    drawChessboardCorners(dispImg, innerCorners, imageCorners, patternfound);

	    // Put 3D points and 2D points into the vectors for calibration
	    if (imageCorners.size() == innerCorners.area())
	    {
		    for(int i = 0; i < innerCorners.height; i++)
		    {
			    for(int j=0; j < innerCorners.width; j++)
			    {
				    objectCorners.push_back(cv::Point3f(float(i)*squareLength, float(j)*squareLength, 0.0f));
			    }
		    }
		    //2D image point from one view
		    imagePoints.push_back(imageCorners);
		    //corresponding 3D scene points
		    objectPoints.push_back(objectCorners);
		    imshow("Source", dispImg);
            waitKey(500);
	    }
    }


	//-----------------------------------------------------------------------------------------------
	// Calibration
	//-----------------------------------------------------------------------------------------------
	// Allocate the camera matrix: 3x3 matrix
	//Note:
	//Intrinsic Matrix - 3x3 
	// [fx 0 cx]              [k1 k2 p1 p2   k3(optional)]
	// [0 fy cy]
	// [0  0  1]
	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

	// Allocate the distortion vectors: the distortion coefficients can be 4, 5, or 8 elements
	// [k1 k2 p1 p2   k3(optional)]
	Mat distCoeffs = Mat::zeros(5, 1, CV_64F);

	// Allocate the rotation and translation vectors
	vector<Mat> rvecs,tvecs;

	cout << "Calibrating... " << endl;
	//cout << "Image size = " << src.size() << endl;
	calibrateCamera(objectPoints, //the 3D points
					imagePoints,
					Size(640, 480), 
					cameraMatrix, //output camera matrix
					distCoeffs,
					rvecs,tvecs,
					flag);
    cout << "Done" << endl;
    cout << "Camera matrix = " << endl << cameraMatrix << endl;
    cout << "Distortion coefficients = " << endl << distCoeffs << endl;
    cout << "Rotation vectors = " << endl << rvecs.at(0) << endl;
    cout << "Translation vectors = " << endl << tvecs.at(0) << endl;

    Mat newCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, Size(640, 480), 1, Size(640, 480), 0);
    cout << "New camera matrix = " << endl << newCameraMatrix << endl;
    Mat view, rview, map1, map2;
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
        getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, Size(640, 480), 1, Size(640, 480), 0),
        Size(640, 480), CV_16SC2, map1, map2);
    cout << map1.size() << endl;
    cout << map2.size() << endl;

    double minVal; 
    double maxVal; 
    Point minLoc; 
    Point maxLoc;

    minMaxLoc( map1, &minVal, &maxVal);//, &minLoc, &maxLoc );

    cout << "min val : " << minVal << endl;
    cout << "max val: " << maxVal << endl;

    minMaxLoc( map2, &minVal, &maxVal);//, &minLoc, &maxLoc );

    cout << "min val : " << minVal << endl;
    cout << "max val: " << maxVal << endl;
    /* initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),cameraMatrix,
        Size(640, 480), CV_32FC1, map1, map2);*/

    //remap(src, rview, map1, Mat(), INTER_LINEAR);
    Mat dst = src.clone();
    undistort(src, dst, cameraMatrix, distCoeffs);


    imshow("Undistorted image", dst);
    waitKey(0);
    return 0;
}
