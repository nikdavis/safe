#include "opencv2/legacy/legacy.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream> 
#include <fstream>
#include "timer.hpp"

using namespace cv;
using namespace std;

#define P_CONST ((double) 0.3989422804)

Mat GRAY_RANGE = (Mat_<float>(256, 1) << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
										16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
										32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
										48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
										64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
										80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
										96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
										112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
										128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
										144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
										160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
										176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
										192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
										208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
										224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
										240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255);

Mat calcProb(Mat src, double sigma, double miu)
{
	if ((src.type() != CV_32F) && (src.type() != CV_64F))
		src.convertTo(src, CV_32F);
	double p;
	//out = (1/sigma*sqrt(2*pi))*exp(-(X-miu).^2/(2*sigma^2));
	// P_CONST: 1/sqrt(2*pi)
	Mat subMat, powMat, divMat, negDivMat, expMat, outMat;
	subtract(src, Scalar(miu), subMat);		// X - miu
	pow(subMat, 2, powMat);					// (X - miu).^2
	divide(powMat, 2 * sigma*sigma, divMat);	// (X-miu).^2/(2*sigma^2)
	subtract(0, divMat, negDivMat);			// -(X-miu).^2/(2*sigma^2)
	exp(negDivMat, expMat);					// exp(-(X-miu).^2/(2*sigma^2));
	p = P_CONST / sigma;					// (1/(sigma*sqrt(2*pi)))
	outMat = expMat*p;						// (1/sigma*sqrt(2*pi))*exp(-(X-miu).^2/sigma.^2);
	//multiply(expMat, Mat::ones(expMat.size(), expMat.type()), outMat, p);			
	return outMat;
}

void writeCSV(Mat data, string fileName)
{
	ofstream fout(fileName);
	for (int r = 0; r < data.rows; r++)
	{
		for (int c = 0; c < data.cols; c++)
		{
			fout << data.at<float>(r, c) << ';';
		}
		fout << endl;
	}
	fout.close();
}

Mat calcHistogram(Mat* img)
{
	Mat hist;
	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;

	/// Compute the histograms:
	calcHist(img, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	hist.at<float>(0) = 0;
	Mat histReturn = hist.clone();
	writeCSV(hist, "hist.csv");

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 2, 0);
	}

	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);

	return histReturn;
}



int main(int argc, char** argv)
{
	//Mat src = imread("E:\\OpenCV Images\\frame2.png", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat src = imread("E:\\OpenCV Images\\frame1_regular.png", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat src = imread("E:\\OpenCV Images\\test.png", CV_LOAD_IMAGE_GRAYSCALE);
	if (argc < 2)
	{
		cout << "Usage: main image #Iter" << endl;
		return -1;
	}
	
	Mat src0 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	resize(src0, src0, Size(), 0.5, 0.5);
	//Mat src = src0(Rect(1, cvRound(src0.rows / 3), src0.cols - 1, (cvRound(2 * src0.rows / 3) - 2) ));
	Mat src = src0.clone();
	imshow("Original Image", src);
	//const int N = (int)atoi(argv[2]);
	const int N = 5;

	//int nsamples = 255;	
	int nsamples = src.size().area();
	Mat samples(nsamples, 1, CV_32FC1);	
	Mat sample(1, 1, CV_32FC1);
	const TermCriteria& termcrit = TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, (int)atoi(argv[2]), FLT_EPSILON);
	EM em_model(N, 1, termcrit);
	
	
	calcHistogram(&src);
	//cout << hist.rows << "x" << hist.cols << endl;
	//samples = hist.rowRange(1, 255);
	//writeCSV(samples, "samples.csv");
	//(Mat_<float>(256, 1)
	//cout << src.col(0) << endl;
	
	src.convertTo(src, CV_32FC1);
	cout << src.at<float>(0,0) << endl;
	int idx = 0;
	for (int r = 0; r < src.rows; r++)
	{
		for (int c = 0; c < src.cols; c++)
		{
			samples.at<float>(idx++) = src.at<float>(r, c);
		}
	}
	cout << src.size().area() << endl;
	cout << samples.size().area() << endl;
	cout << samples.rows << "x" << samples.cols << endl;
	double min, max;
	minMaxLoc(samples, &min, &max);
	cout << "Min = " << min << endl << "Max = " << max << endl;

	// initialize model parameters

	timer trainTimer("training: 	");
	trainTimer.start();
	Mat model_mean;
	if (N == 5)
		model_mean = (Mat_<float>(5, 1) << 0, 60, 100, 40, 210);
	if (N == 4)
		model_mean = (Mat_<float>(4, 1) << 60, 100, 40, 210);
		
	Mat model_weight = (Mat_<float>(3, 1) << 0.781511, 0.129899, 0.0885905);
	vector<Mat> model_covs;
	model_covs.push_back((Mat_<float>(1, 1) << 106.59));
	model_covs.push_back((Mat_<float>(1, 1) << 1551.14));
	model_covs.push_back((Mat_<float>(1, 1) << 148.819));
	//em_model.trainE(samples, model_mean);
	//em_model.trainE(samples, model_mean, model_covs, model_weight);
	em_model.train(samples);
	trainTimer.stop();
	trainTimer.printm();


	const vector<Mat>& covs = em_model.get<vector<Mat> >("covs");
	for (int i = 0; i < N; i++)
		cout << "covs" << i + 1 << " = " << covs.at(i).at<double>(0) << endl; 

	const Mat& means = em_model.get<Mat>("means");
	for (int i = 0; i < N; i++)
		cout << "mean" << i + 1 << " = " << means.at<double>(i) << endl;

	const Mat& weights = em_model.get<Mat>("weights");
	for (int i = 0; i < N; i++)
		cout << "weight" << i + 1 << " = " << weights.at<double>(i) << endl;
	cout << "sum of weight = " << sum(weights)[0] << endl;

	
	Mat src1, src2, src3, src4, src5;
	src1 = src.clone();
	src2 = src.clone();
	src3 = src.clone();
	src4 = src.clone();
	src5 = src.clone();

	for (int r = 0; r < src.rows; r++)
	{
		for (int c = 0; c < src.cols; c++)
		{
			sample.at<float>(0) = src.at<float>(r, c);
			int response = cvRound(em_model.predict(sample)[1]);
			switch (response)
			{
			case 0:
				src1.at<float>(r, c) = 255;
				src2.at<float>(r, c) = 0;
				src3.at<float>(r, c) = 0;
				src4.at<float>(r, c) = 0;
				src5.at<float>(r, c) = 0;
				break;
			case 1:
				src1.at<float>(r, c) = 0;
				src2.at<float>(r, c) = 255;
				src3.at<float>(r, c) = 0;
				src4.at<float>(r, c) = 0;
				src5.at<float>(r, c) = 0;
				break;
			case 2:
				src1.at<float>(r, c) = 0;
				src2.at<float>(r, c) = 0;
				src3.at<float>(r, c) = 255;
				src4.at<float>(r, c) = 0;
				src5.at<float>(r, c) = 0;
				break;
			case 3:
				src1.at<float>(r, c) = 0;
				src2.at<float>(r, c) = 0;
				src3.at<float>(r, c) = 0;
				src4.at<float>(r, c) = 255;
				src5.at<float>(r, c) = 0;
				break;
			case 4:
				src1.at<float>(r, c) = 0;
				src2.at<float>(r, c) = 0;
				src3.at<float>(r, c) = 0;
				src4.at<float>(r, c) = 0;
				src5.at<float>(r, c) = 255;
				break;
			default:
				break;
			}
		}
	}
	
	imshow("1", src1);
	imshow("2", src2);
	imshow("3", src3);
	imshow("4", src4);
	imshow("5", src5);
	waitKey(0);

	return 0;
}

