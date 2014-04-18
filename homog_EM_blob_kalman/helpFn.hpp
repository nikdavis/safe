#ifndef __HELP_FN_HPP__
#define __HELP_FN_HPP__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <sstream> 
#include <fstream> 

using namespace cv;
using namespace std;

#define RAND(out, range)	( out = (int)(rand() % range) )

void writeCSV(Mat* data, string fileName, bool append = false, string delimiter = ",");

void writeCSV(Mat* data, string fileName, string delimiter);

void writeArrayCSV(double* data, int length, string fileName, bool isCol = true);

void writeArrayCSV(double* data, int length, string fileName, string delimeter, bool isCol = true);

bool saveImg(Mat* img, string fileNameFormat, int* fileNum);

bool saveBoxImg(Mat* img, Rect* box, string fileNameFormat, int* fileNum);

void rotateImg(Mat* src, Mat* dst, int angleDegrees);


#endif /* __HELP_FN_HPP__ */
