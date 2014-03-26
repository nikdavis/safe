#include "helpFn.hpp"

void writeCSV(Mat* data, string fileName)
{
	Mat tmp;
	data->convertTo(tmp, CV_32F);
	ofstream fout(fileName);
	for (int r = 0; r < tmp.rows; r++)
	{
		for (int c = 0; c < tmp.cols; c++)
		{
			fout << tmp.at<float>(r, c) << ';';
		}
		fout << endl;
	}
	fout.close();
}

bool saveImg(Mat* img, string fileNameFormat, int* fileNum)
{
	Mat save;
	if (img->channels() == 3)
		cvtColor(*img, save, CV_RGB2GRAY);
	else if (img->channels() == 4)
		cvtColor(*img, save, CV_RGBA2GRAY);
	else
		save = img->clone();
	// Save image as pgm
	vector< int > compression_params;			//vector that stores the compression parameters of the image
	compression_params.push_back(CV_IMWRITE_PXM_BINARY);
	compression_params.push_back(9);
	string sFileNum = static_cast<ostringstream*>(&(ostringstream() << *fileNum))->str();
	string sFileName = fileNameFormat;
	if (*fileNum < 10)
		sFileName += ("_000" + sFileNum + ".pgm");
	else if ((*fileNum >= 10) && (*fileNum < 100))
		sFileName += ("_00" + sFileNum + ".pgm");
	else if ((*fileNum >= 100) && (*fileNum < 1000))
		sFileName += ("_0" + sFileNum + ".pgm");
	else
		sFileName += ("_" + sFileNum + ".pgm");
	return imwrite(sFileName, save, compression_params);
}

bool saveBoxImg(Mat* img, Rect* box, string fileNameFormat, int* fileNum)
{
	// Check whether the ROI is within the image
	if ((box->tl().x < 0) || (box->tl().y < 0) || (box->br().x > img->cols) || (box->br().y > img->rows))
	{
		return false;
	}

	// Check whether the box area is greater than 0
	if (!box->area())
	{
		return false;
	}


	Mat save = (*img)(*box).clone();
	// Convert image to grayscale if it is not
	if (save.channels() == 3)
		cvtColor(save, save, CV_RGB2GRAY);
	else if (save.channels() == 4)
		cvtColor(save, save, CV_RGBA2GRAY);
	
	// Save image as pgm
	vector< int > compression_params;			//vector that stores the compression parameters of the image
	compression_params.push_back(CV_IMWRITE_PXM_BINARY);
	compression_params.push_back(9);
	string sFileNum = static_cast<ostringstream*>(&(ostringstream() << *fileNum))->str();
	string sFileName = fileNameFormat;
	if (*fileNum < 10)
		sFileName += ("_000" + sFileNum + ".pgm");
	else if ((*fileNum >= 10) && (*fileNum < 100))
		sFileName += ("_00" + sFileNum + ".pgm");
	else if ((*fileNum >= 100) && (*fileNum < 1000))
		sFileName += ("_0" + sFileNum + ".pgm");
	else
		sFileName += ("_" + sFileNum + ".pgm");
	return imwrite(sFileName, save, compression_params);
}

void rotateImg(Mat* src, Mat* dst, int* angleDegrees)
{
	if (!(*angleDegrees))
	{
		// Compute rotation matrix
		CvPoint2D32f center = cvPoint2D32f(src->size().width / 2, src->size().height / 2);
		Mat rot_mat = getRotationMatrix2D(center, *angleDegrees, 1);

		// Do the transformation
		warpAffine(*src, *dst, rot_mat, src->size());
	}
}
