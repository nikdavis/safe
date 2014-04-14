#ifndef __CAR_SVM__
#define __CAR_SVM__

#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <pthread.h>
#include <string>
#include <sstream> 
#include <fstream>
#include "svm.hpp"

#define IMAGE_SIZE		( 16 )

using namespace cv;
using namespace std;

class CarSVM
{
public:
	//svm_problem data;
	enum CAR_CLASS : int
	{
		NEG = -1,
		POS = 1
	};

	void mat2SVMprob(Mat* dataset, Mat* labels, svm_problem* data);

	void writeCSV(Mat data, string fileName);

	void readSVM_CSV(string fileName, int maxSvmNodeSize, svm_problem* data);

	void writeSVM_CSV(Mat* training_data, Mat* label_mat, string fileName);

	void writeSVM_CSV(svm_problem* dataset, string fileName);

	bool combineDataset(Mat* dataset1, Mat* dataset2, Mat* outDataset, Mat* label1, Mat* label2, Mat* outLabel);

	int	addDataset(Mat* dataset, Mat* label, string pathToFirstImg, int numImages, float labelValue);

	void rearrangeDataset(Mat* training_data, Mat* label_mat, int numIter);

	void rearrangeDataset(Mat* training_data, Mat* label_mat);

	void linspace(double startValue, double endValue, int numOfPoint, double* outArray);

	void logspace(double startValue, double endValue, int numOfPoint, double* outArray);

	bool predict(Mat* img, double classType, const svm_model *carModel);
	
	bool predictProb(Mat* img, double classType, const svm_model *carModel, double* prob_estimates);
};


#endif /*  __CAR_SVM__ */
