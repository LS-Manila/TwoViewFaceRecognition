// This program is used to train the cv::face::FaceRecognizer class 
// for face recognition. The program will read a csv file that contains
// lines composed of the filename and label of the person in the 
// database.

#include <iostream> // std::cout
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip> // std::setpricision

//#include <opencv/cv.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/face.hpp>

using namespace std;
using namespace cv;

// Global variables

vector<Mat> images;
vector<int> labels;

static void read_csv(const string& fileName, vector<Mat>& images, 
					 vector<int>& labels, char separator = ';'){
	ifstream file(fileName.c_str(), ifstream::in);
	if(!file){
		string errorMessage = "No valid input file was given. Please "
							  "check the given filename.";
		CV_Error(CV_StsBadArg, errorMessage);
	}
	string line, path, classLabel;
	while(getline(file, line)){
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classLabel);
		if(!path.empty() && !classLabel.empty()){
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classLabel.c_str()));
		}
	}
}

int main(int argc, char** argv){
	if(argc !=2){
		cerr << "Error: path to training set missing." << endl 
			 << "Usage: ./trainer <path_to_csv_file>" << endl;
		exit(1);
	}
	
	string csv_ext = string(argv[1]);
	
	try{
		read_csv(csv_ext, images, labels);
	}catch (cv::Exception& e){
		cerr << "Error opening file \"" << csv_ext << "\". Reason: ";
		cerr << e.msg << endl;
		exit(1);
	}
	
	// Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size:
	int height = images[0].rows;
	// The following lines simply get the last images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::FaceRecognizer on) and the test data we test
    // the model with, do not overlap.
    Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();
	
	double t;	
	t = (double)getTickCount();
	
	// The following lines create an Eigenfaces model for
    // face recognition and train it with the images and
    // labels read from the given CSV file.
    // This here is a full PCA, if you just want to keep
    // 10 principal components (read Eigenfaces), then call
    // the factory method like this:
    //
    //      cv::createEigenFaceRecognizer(10);
    //
    // If you want to create a FaceRecognizer with a
    // confidence threshold (e.g. 123.0), call it with:
    //
    //      cv::createEigenFaceRecognizer(10, 123.0);
    //
    // If you want to use _all_ Eigenfaces and have a threshold,
    // then call the method like this:
    //
    //      cv::createEigenFaceRecognizer(0, 123.0);
    //
	Ptr<face::FaceRecognizer> model = face::createEigenFaceRecognizer();
	model -> train(images, labels);
	model -> save("eigenfaces_at.yml");
	
	int predictedLabel = model->predict(testSample);
    //
    // To get the confidence of a prediction call the model with:
    //
    //      int predictedLabel = -1;
    //      double confidence = 0.0;
    //      model->predict(testSample, predictedLabel, confidence);
    //
    string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
    cout << result_message << endl;
	
	t = ((double)getTickCount() - t)/getTickFrequency();	
	cout << "Training took " << setprecision(3) << t << " s." << endl;
	
	return 0;
}
