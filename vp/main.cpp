/*
 * Project:  vanishingPoint
 *
 * File:     main.cpp
 *
 * Contents: Creation, initialisation and usage of MSAC object
 *           for vanishing point estimation in images or videos
 *
 * Author:   Marcos Nieto <marcos.nieto.doncel@gmail.com>
 *
 * Homepage: www.marcosnieto.net/vanishingPoint
 */


#include <iostream>
#include <stdio.h>

#define USE_PPHT
#define MAX_NUM_LINES	200

#define ROTATE_TAU               0
#define MIN_TAU                  5
#define MAX_TAU                  15
#define TAU                      7
#define TAU_DELTA                10
#define LANE_FILTER_ROW_OFFSET   0

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "MSAC.h"

using namespace std;
using namespace cv;


/** This function contains the actions performed for each image*/
void processImage(MSAC &msac, int numVps, cv::Mat &imgGRAY, cv::Mat &outputImg, Point &vanPt)
{
	cv::Mat imgCanny, imgLane;

      imgLane = Mat::zeros(imgGRAY.rows, imgGRAY.cols, CV_32F);
      int px_out = 0;
      int tau_cnt = 0;
#if ROTATE_TAU
      int tau = MIN_TAU;
#else
      int tau = TAU;
#endif
      for (int row=0;row<imgGRAY.rows;row++) {
         unsigned char * data = imgGRAY.ptr<uchar>(row);
         float * out = imgLane.ptr<float>(row);
         
         for (int col=0;col<imgGRAY.cols;col++) {

            if((col >= tau) && (col < (imgGRAY.cols - tau)) && (row > ((imgGRAY.rows / 2) + LANE_FILTER_ROW_OFFSET))) {   // Check that we're within kernel size
               // Filter from Nietos 2010
               px_out= 2 * data[col];
               px_out -= data[col-tau];
               px_out -= data[col+tau];
               px_out -= abs((int)(data[col-tau] - data[col+tau]));
               px_out = (px_out < 0) ? 0 : px_out;
               px_out = (px_out > 255) ? 255 : px_out;
               out[col] = (unsigned char)px_out;
               //cout << out[col] << endl;
            } else {
               out[col] = 0;
            }
         }
         if(row > ((imgGRAY.rows / 2) + LANE_FILTER_ROW_OFFSET)) {
            tau_cnt++;
            if((tau_cnt % ((imgGRAY.rows / 2) / TAU_DELTA)) == 0) {
               //cout << "row: " << row << endl;
               //cout << "tau: " << tau << endl;
               tau++;
            }
         }
      }  

    //cv::cvtColor(imgLane, imgGRAY, CV_BGR2GRAY);
    imgLane.convertTo(imgGRAY, CV_8UC1);
    //cv::normalize(imgGRAY, imgGRAY, 0, 255, NORM_MINMAX, CV_8UC1);
	// Canny
	cv::Canny(imgGRAY, imgCanny, 100, 200, 3);

	// Hough
	vector<vector<cv::Point> > lineSegments;
	vector<cv::Point> aux;
#ifndef USE_PPHT
	vector<Vec2f> lines;
	cv::HoughLines( imgCanny, lines, 1, CV_PI/180, 200);

	for(size_t i=0; i< lines.size(); i++)
	{
		float rho = lines[i][0];
		float theta = lines[i][1];

		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;

		Point pt1, pt2;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));

		aux.clear();
		aux.push_back(pt1);
		aux.push_back(pt2);
		lineSegments.push_back(aux);

		line(outputImg, pt1, pt2, CV_RGB(0, 0, 0), 1, 8);
	
	}
#else
	vector<Vec4i> lines;	
	int houghThreshold = 70;
	if(imgGRAY.cols*imgGRAY.rows < 400*400)
		houghThreshold = 100;		
	
	cv::HoughLinesP(imgCanny, lines, 1, CV_PI/180, houghThreshold, 10,10);

	while(lines.size() > MAX_NUM_LINES)
	{
		lines.clear();
		houghThreshold += 10;
		cv::HoughLinesP(imgCanny, lines, 1, CV_PI/180, houghThreshold, 10, 10);
	}
	for(size_t i=0; i<lines.size(); i++)
	{		
		Point pt1, pt2;
		pt1.x = lines[i][0];
		pt1.y = lines[i][1];
		pt2.x = lines[i][2];
		pt2.y = lines[i][3];
		line(outputImg, pt1, pt2, CV_RGB(0,0,0), 2);
		/*circle(outputImg, pt1, 2, CV_RGB(255,255,255), CV_FILLED);
		circle(outputImg, pt1, 3, CV_RGB(0,0,0),1);
		circle(outputImg, pt2, 2, CV_RGB(255,255,255), CV_FILLED);
		circle(outputImg, pt2, 3, CV_RGB(0,0,0),1);*/

		// Store into vector of pairs of Points for msac
		aux.clear();
		aux.push_back(pt1);
		aux.push_back(pt2);
		lineSegments.push_back(aux);
	}
	
#endif

	// Multiple vanishing points
	std::vector<std::vector<int> > CS;	// index of Consensus Set for all vps: CS[vpNum] is a vector containing indexes of lineSegments belonging to Consensus Set of vp numVp
	std::vector<int> numInliers;

	std::vector<std::vector<std::vector<cv::Point> > > lineSegmentsClusters;
	
	// Call msac function for multiple vanishing point estimation
	msac.multipleVPEstimation(lineSegments, lineSegmentsClusters, numInliers, vps, numVps); 
	
   if(vps.size() == 0) {
      printf("-1,-1\n");
      vanPt.x = -1.0;
      vanPt.y = -1.0;
   } else if
   {
      //printf("%.3f,%.3f\n", v, vps[v].at<float>(0,0), vps[v].at<float>(1,0));
	   double vpNorm = cv::norm(vps[v]);
	   if(fabs(vpNorm - 1) < 0.001)
	   {
	   	printf("-1,-1 (inf)\n");
         vanPt.x = -1.0;
         vanPt.y = -1.0;
	   } else {
         
      }
       
   }
   vanPt.clear();
   vanPt.push_back(pt1);
	vanPt.push_back(pt2);
   
		
	// Draw line segments according to their cluster
	msac.drawCS(outputImg, lineSegmentsClusters, vps);
}

/** Main function*/
int main(int argc, char** argv)
{	
	// Images
	cv::Mat inputImg, imgGRAY;	
	cv::Mat outputImg;
	std::vector<cv::Mat> vps;			// vector of vps: vps[vpNum], with vpNum=0...numDetectedVps

	// Other variables
	char *videoFileName = 0;
	char *imageFileName = 0;
	cv::VideoCapture video;
	bool useCamera = true;
	int mode = MODE_NIETO;
	int numVps = 1;
	bool playMode = true;
	bool stillImage = false;
	bool verbose = false;

	int procWidth = -1;
	int procHeight = -1;
	cv::Size procSize;


	// Open video input
	if( useCamera )
		video.open(0);
	else
	{
		if(!stillImage)
			video.open(videoFileName);
	}

	// Check video input
	int width = 0, height = 0, fps = 0, fourcc = 0;
	if(!stillImage)
	{
		if( !video.isOpened() )
		{
			printf("ERROR: can not open camera or video file\n");
			return -1;
		}
		else
		{
			// Show video information
			width = (int) video.get(CV_CAP_PROP_FRAME_WIDTH);
			height = (int) video.get(CV_CAP_PROP_FRAME_HEIGHT);
			fps = (int) video.get(CV_CAP_PROP_FPS);
			fourcc = (int) video.get(CV_CAP_PROP_FOURCC);

			if(!useCamera)
				printf("Input video: (%d x %d) at %d fps, fourcc = %d\n", width, height, fps, fourcc);
			else
				printf("Input camera: (%d x %d) at %d fps\n", width, height, fps);
		}
	}
	else
	{
		inputImg = cv::imread(imageFileName);
		if(inputImg.empty())
			return -1;

		width = inputImg.cols;
		height = inputImg.rows;

		printf("Input image: (%d x %d)\n", width, height);

		playMode = false;
	}

	// Resize	
	if(procWidth != -1)
	{
	
		procHeight = height*((double)procWidth/width);
		procSize = cv::Size(procWidth, procHeight);

		printf("Resize to: (%d x %d)\n", procWidth, procHeight);	
	}
	else
		procSize = cv::Size(width, height);

	// Create and init MSAC
	MSAC msac;
	msac.init(mode, procSize, verbose);
	
	printf("\n");
	int frameNum=0;
	for( ;; )
	{
		if(!stillImage)
		{
			printf("%d,", frameNum);
			frameNum++;

			// Get current image		
			video >> inputImg;
		}	

		if( inputImg.empty() )
			break;
		
		// Resize to processing size
		cv::resize(inputImg, inputImg, procSize);		

		// Color Conversion
		if(inputImg.channels() == 3)
		{
			cv::cvtColor(inputImg, imgGRAY, CV_BGR2GRAY);	
			inputImg.copyTo(outputImg);
		}
		else
		{
			inputImg.copyTo(imgGRAY);
			cv::cvtColor(inputImg, outputImg, CV_GRAY2BGR);
		}

		// ++++++++++++++++++++++++++++++++++++++++
		// Process		
		// ++++++++++++++++++++++++++++++++++++++++
        
		processImage(msac, numVps, imgGRAY, outputImg, vps);


		// View
		imshow("Output", outputImg);\

		char q = (char)waitKey(1);

		if( q == 27 )
		{
			printf("\nStopped by user request\n");
			break;
		}
	}

	if(!stillImage)
		video.release();
	
	return 0;	
	
}
