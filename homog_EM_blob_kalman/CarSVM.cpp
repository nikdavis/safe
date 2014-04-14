#include "CarSVM.hpp"

using namespace cv;
using namespace std;

/* This function is to convert the dataset in the format of cv::Mat to 
 * svm_problem format. This format is necessary for training step.
 */
void CarSVM::mat2SVMprob(Mat* dataset, Mat* labels, svm_problem* data)
{
	if (dataset->rows != labels->rows)
	{
		cout << "Training data's and label's size are not match." << endl;
		return;
	}
	if (labels->cols > 1)
	{
		cout << "Label's size must be a column Matrix." << endl;
		return;
	}

	// Convert dataset's and label's type to CV_32F
	if (dataset->type() != CV_32F)
		dataset->convertTo(*dataset, CV_32F);
	if (labels->type() != CV_32F)
		labels->convertTo(*labels, CV_32F);

	// Allocate memory for svm_problem
	data->l = dataset->rows;
	data->y = new double[dataset->rows];
	data->x = new svm_node*[dataset->rows];
	for (int i = 0; i < dataset->rows; i++) 
		data->x[i] = new svm_node[dataset->cols];
	
	for (int dataIdx = 0; dataIdx < dataset->rows; dataIdx++)
	{
		int index = 0;
		for (int c = 0; c < dataset->cols; c++)
		{
			if (dataset->at<float>(dataIdx, c) != 0)
			{
				data->x[dataIdx][index].index = c;
				data->x[dataIdx][index].value = (double)(dataset->at<float>(dataIdx, c));
				index++;
			}
		}
		// The svm_node index of the last node is (-1)
		data->x[dataIdx][index].index = (int)(-1);

		// Assign label for each line of dataset
		data->y[dataIdx] = (double)(labels->at<float>(dataIdx));
	}
}

/* Read dataset in format of svm_problem in a CSV file and return dataset
 * in a svm_problem variable.
 * The maxSvmNodeSize is the maximum size of svm_node array.
 */
void CarSVM::readSVM_CSV(string fileName, int maxSvmNodeSize, svm_problem* data)
{
	ifstream input(fileName);
	
	// Count the number of lines in the csv input file. This is to determine
	// the amount of memory needed to allocate svm_problem pointer
	int numLines = count(istreambuf_iterator<char>(input), istreambuf_iterator<char>(), '\n');
	input.close();

	ifstream csvInput(fileName);
	data = new svm_problem[1];
	data->y = new double[numLines];
	data->x = new svm_node*[numLines];
	for (int i = 0; i < numLines; i++) {
		data->x[i] = new svm_node[maxSvmNodeSize];
	}

	int lineNumber = 0;
	
	string line;
	string *lineItems = new string[maxSvmNodeSize + 1];
	int lineItemCount = 0;
	while (!csvInput.eof())
	{
		getline(csvInput, line, '\n');
		if (!line.empty())
		{
			int pos;
			lineItemCount = 0;
			do {
				pos = line.find(" ");
				lineItems[lineItemCount] = line.substr(0, pos);
				line = line.substr(pos + 1);
				lineItemCount++;
			} while (pos > 0);
			data->y[lineNumber] = (double)atoi(lineItems[0].c_str());
			for (int i = 1; i < lineItemCount; i++)
			{
				int p = lineItems[i].find(":");
				data->x[lineNumber][i - 1].index = (int)atoi(lineItems[i].substr(0, p).c_str());
				data->x[lineNumber][i - 1].value = (double)atoi(lineItems[i].substr(p + 1).c_str());
			}
			data->x[lineNumber][lineItemCount - 1].index = (int)(-1);
			lineNumber++;
		}
	}
	data->l = lineNumber;
	cout << data->l << endl;

	// Close the CSV file
	csvInput.close();
}

/* Write cv::Mat to CSV file.
 */
void CarSVM::writeCSV(Mat data, string fileName)
{
	ofstream fout(fileName);
	data.convertTo(data, CV_32F);
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

/* Write data with label to CSV file in the format of SVM.
 */
void CarSVM::writeSVM_CSV(svm_problem* dataset, string fileName)
{
	// Create a file named 'fileName' (if it is not exist) then open
	ofstream fout(fileName);
	for (int dataIdx = 0; dataIdx < dataset->l; dataIdx++)
	{
		fout << dataset->y[dataIdx] << " ";
		while (dataset->x[dataIdx]->index != (int)(-1))
		{
			fout << dataset->x[dataIdx]->index << ":" << dataset->x[dataIdx]->value << " ";
			dataset->x[dataIdx]++;
		}
		fout << endl;
	}
	fout.close();
}

/* Write data with label to CSV file in the format of SVM.
 */
void CarSVM::writeSVM_CSV(Mat* data, Mat* labels, string fileName)
{
	if (data->rows != labels->rows)
	{
		cout << "Training data's and label's size are not match." << endl;
		return;
	}
	if (labels->cols > 1)
	{
		cout << "Label's size must be a column Matrix." << endl;
		return;
	}

	// Create a file named 'fileName' (if it is not exist) then open
	ofstream fout(fileName);

	if (data->type() != CV_32F)
		data->convertTo(*data, CV_32F);
	if (labels->type() != CV_32F)
		labels->convertTo(*labels, CV_32F);

	for (int dataIdx = 0; dataIdx < data->rows; dataIdx++)
	{
		fout << (double)(labels->at<float>(dataIdx)) << " ";
		for (int c = 0; c < data->cols; c++)
		{
			if (data->at<float>(dataIdx, c) != 0)
				fout << c << ":" << (double)(data->at<float>(dataIdx, c)) << " ";
		}
		fout << endl;
	}
	fout.close();
	return;
}

/* This function will combine two datasets of two classes.
 */
bool CarSVM::combineDataset(Mat* dataset1, Mat* dataset2, Mat* outDataset, Mat* label1, Mat* label2, Mat* outLabel)
{
	if (dataset1->cols != dataset2->cols)
	{
		cout << "Input datasets do not have the same width" << endl;
		return false;
	}
	if (dataset1->rows != label1->rows)
	{
		cout << "Input dataset1 do not have the same number of data" << endl;
		return false;
	}

	if (dataset2->rows != label2->rows)
	{
		cout << "Input dataset2 do not have the same number of data" << endl;
		return false;
	}

	if (dataset1->type() != CV_32FC1)
		dataset1->convertTo(*dataset1, CV_32FC1);
	if (dataset2->type() != CV_32FC1)
		dataset2->convertTo(*dataset2, CV_32FC1);
	if (label1->type() != CV_32FC1)
		label1->convertTo(*label1, CV_32FC1);
	if (label2->type() != CV_32FC1)
		label2->convertTo(*label2, CV_32FC1);

	int numData = dataset1->rows + dataset2->rows;
	Mat training_mat(numData, dataset1->cols, CV_32FC1);
	Mat labels_mat(numData, 1, CV_32FC1);

	int i = 0;
	for (int j = 0; j < dataset1->rows; j++)
	{
		dataset1->row(j).copyTo(training_mat.row(i));
		label1->row(j).copyTo(labels_mat.row(i++));
	}

	for (int j = 0; j < dataset2->rows; j++)
	{
		dataset2->row(j).copyTo(training_mat.row(i));
		label2->row(j).copyTo(labels_mat.row(i++));
	}

	training_mat.copyTo(*outDataset);
	labels_mat.copyTo(*outLabel);
	return true;
}

/* Load sequence of images then convert each images to row matrix. Each image will be contained in
 * a row of the 'dataset' matrix. The labels for images are contained in the 'label' matrix in 
 * corresponding row.
 * NOTE that the image will be applied adaptive threshold with block size is a half of the min
 * of image width and height (block size must be an odd number).
 */
int CarSVM::addDataset(Mat* dataset, Mat* label, string pathToFirstImg, int numImages, float labelValue)
{
	Mat src;
	int file_num = 0;
	string pathToData(pathToFirstImg);
	VideoCapture sequence(pathToData);

	sequence >> src;
	if (src.empty()) {
		cout << "Sequence is empty" << endl;
		return 0;
	}

	if (src.channels() == 3)
		cvtColor(src, src, CV_RGB2GRAY);
	else if (src.channels() == 4)
		cvtColor(src, src, CV_RGBA2GRAY);
	resize(src, src, Size(IMAGE_SIZE, IMAGE_SIZE));

	Mat training_mat(numImages, src.size().area(), src.type());
	Mat labels_mat(numImages, 1, CV_32FC1);

	int h = src.rows;
	int w = src.cols;

	// Block size must be an odd number
	int blockSize = min(h, w) / 2;
	blockSize = (blockSize % 2) ? blockSize : (blockSize - 1);

	for (file_num = 0; file_num < numImages; file_num++)
	{
		if (src.channels() == 3)
			cvtColor(src, src, CV_RGB2GRAY);
		else if (src.channels() == 4)
			cvtColor(src, src, CV_RGBA2GRAY);

		resize(src, src, Size(IMAGE_SIZE, IMAGE_SIZE));
		adaptiveThreshold(src, src, 1, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, blockSize, 0);
		normalize(src, src, 0, 1, NORM_MINMAX);
		for (int i = 0; i < h; i++)
		{
			src.row(i).copyTo(training_mat.row(file_num).colRange(i*w, (i + 1)*w));
		}
		labels_mat.at<float>(file_num) = labelValue;

		sequence >> src;
		if (src.empty())
		{
			cout << "End of Sequence" << endl;
			break;
		}
	}
	training_mat.convertTo(*dataset, CV_32FC1);
	labels_mat.convertTo(*label, CV_32FC1);
	return file_num;
}

/* This function will rearrange dataset for training in a random order. This step is
 * necessary to make training more accurate.
 */
void CarSVM::rearrangeDataset(Mat* training_data, Mat* label_mat, int numIter)
{
	if (training_data->rows != label_mat->rows)
	{
		cout << "Training data's and label's size are not match." << endl;
		return;
	}

	int numData = training_data->rows;
	if (training_data->type() != CV_32FC1)
		training_data->convertTo(*training_data, CV_32FC1);
	Mat temp_data_mat(1, training_data->cols, CV_32FC1);
	Mat temp_label_mat(1, 1, CV_32FC1);

	// Interate 'numIter' to rearrange dataset
	for (int n = 0; n < numIter; n++)
	{
		int x = (int)(rand() % numData);
		int y = (int)(rand() % numData);

		// swap data
		training_data->row(x).copyTo(temp_data_mat.row(0));
		training_data->row(y).copyTo(training_data->row(x));
		temp_data_mat.row(0).copyTo(training_data->row(y));

		// swap label
		label_mat->row(x).copyTo(temp_label_mat.row(0));
		label_mat->row(y).copyTo(label_mat->row(x));
		temp_label_mat.row(0).copyTo(label_mat->row(y));
	}
}

/* This function will rearrange dataset for training in a random order. This step is
* necessary to make training more accurate.
*/
void CarSVM::rearrangeDataset(Mat* training_data, Mat* label_mat)
{
	if (training_data->rows != label_mat->rows)
	{
		cout << "Training data's and label's size are not match." << endl;
		return;
	}

	int numData = training_data->rows;
	if (training_data->type() != CV_32FC1)
		training_data->convertTo(*training_data, CV_32FC1);
	Mat temp_data_mat(1, training_data->cols, CV_32FC1);
	Mat temp_label_mat(1, 1, CV_32FC1);

	// Interate 'numData' to rearrange dataset
	for (int n = 0; n < numData; n++)
	{
		int x = (int)(rand() % numData);
		int y = (int)(rand() % numData);

		// swap data
		training_data->row(x).copyTo(temp_data_mat.row(0));
		training_data->row(y).copyTo(training_data->row(x));
		temp_data_mat.row(0).copyTo(training_data->row(y));

		// swap label
		label_mat->row(x).copyTo(temp_label_mat.row(0));
		label_mat->row(y).copyTo(label_mat->row(x));
		temp_label_mat.row(0).copyTo(label_mat->row(y));
	}
}

/* Generate a linear array
 */
void CarSVM::linspace(double startValue, double endValue, int numOfPoint, double* outArray)
{
	if (numOfPoint < 2)
	{
		cout << "Number of points must be more than 2." << endl;
		return;
	}
	double stepValue = (startValue - endValue) / (numOfPoint - 1);
	for (int i = 0; i < numOfPoint; i++)
	{
		outArray[i] = startValue + stepValue*i;
	}
}

/* Generates numOfPoint points between decades 'startValue' and 'endValue'.
 * The output array is in *outArray.
 */
void CarSVM::logspace(double startValue, double endValue, int numOfPoint, double* outArray)
{
	if (numOfPoint < 2)
	{
		cout << "Number of points must be more than 2." << endl;
		return;
	}
	double stepValue = (log10(endValue) - log10(startValue)) / (numOfPoint - 1);
	for (int i = 0; i < numOfPoint; i++)
	{
		outArray[i] = startValue*pow(10, stepValue*i);
	}
}

bool CarSVM::predict(Mat* img, double classType, const svm_model *carModel)
{
	resize(*img, *img, Size(IMAGE_SIZE, IMAGE_SIZE));
	if (img->channels() == 3)
		cvtColor(*img, *img, CV_RGB2GRAY);
	else if (img->channels() == 4)
		cvtColor(*img, *img, CV_RGBA2GRAY);

	int h = img->rows;
	int w = img->cols;

	// Block size must be an odd number
	int blockSize = min(h, w) / 2;
	blockSize = (blockSize % 2) ? blockSize : (blockSize - 1);

	// Allocate memory space for test_mat and test_node
	Mat test_mat(1, img->size().area(), img->type());
	svm_node* test_node = new svm_node[img->size().area()];

	
	adaptiveThreshold(*img, *img, 1, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, blockSize, 0);
	normalize(*img, *img, 0, 1, NORM_MINMAX);
	for (int i = 0; i < h; i++)
	{
		img->row(i).copyTo(test_mat.row(0).colRange(i*w, (i + 1)*w));
	}

	test_mat.convertTo(test_mat, CV_32F);
	int index = 0;
	for (int c = 0; c < test_mat.cols; c++)
	{
		if (test_mat.at<float>(c) != 0)
		{
			test_node[index].index = c;
			test_node[index].value = (double)(test_mat.at<float>(c));
			index++;
		}
	}
	// The svm_node index of the last node is (-1)
	test_node[index].index = (int)(-1);

	// Predict
	double result = svm_predict(carModel, test_node);
	
	return (result == classType) ? true : false;
}

bool CarSVM::predictProb(Mat* img, double classType, const svm_model *carModel, double* prob_estimates)
{
	resize(*img, *img, Size(IMAGE_SIZE, IMAGE_SIZE));
	if (img->channels() == 3)
		cvtColor(*img, *img, CV_RGB2GRAY);
	else if (img->channels() == 4)
		cvtColor(*img, *img, CV_RGBA2GRAY);
	
	int h = img->rows;
	int w = img->cols;

	// Block size must be an odd number
	int blockSize = min(h, w) / 2;
	blockSize = (blockSize % 2) ? blockSize : (blockSize - 1);

	// Allocate memory space for test_mat and test_node
	Mat test_mat(1, img->size().area(), img->type());
	svm_node* test_node = new svm_node[img->size().area()];

	adaptiveThreshold(*img, *img, 1, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, blockSize, 0);
	normalize(*img, *img, 0, 1, NORM_MINMAX);
	for (int i = 0; i < h; i++)
	{
		img->row(i).copyTo(test_mat.row(0).colRange(i*w, (i + 1)*w));
	}

	test_mat.convertTo(test_mat, CV_32F);
	int index = 0;
	for (int c = 0; c < test_mat.cols; c++)
	{
		if (test_mat.at<float>(c) != 0)
		{
			test_node[index].index = c;
			test_node[index].value = (double)(test_mat.at<float>(c));
			index++;
		}
	}
	// The svm_node index of the last node is (-1)
	test_node[index].index = (int)(-1);

	// Predict
	double result = svm_predict_probability(carModel, test_node, prob_estimates);

	return (result == classType) ? true : false;
}
