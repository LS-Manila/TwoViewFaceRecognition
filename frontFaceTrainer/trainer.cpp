// This program is used to train the cv::face::FaceRecognizer class 
// for face recognition. The program will read a csv file that contains
// lines composed of the filename and label of the person in the 
// database.

#include <fstream>
#include <iomanip> 
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

using namespace std;
using namespace cv;
using namespace face;

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
		exit(0);
	}
	
	string csv_ext = string(argv[1]);
	
	try{
		read_csv(csv_ext, images, labels);
	}catch (cv::Exception& e){
		cerr << "Error opening file \"" << csv_ext << "\". Reason: ";
		cerr << e.msg << endl;
		exit(0);
	}
	
	double t;	
	t = (double)getTickCount();
	
	Ptr<face::FaceRecognizer> model = createEigenFaceRecognizer();
	model -> train(images, labels);
	model -> save("eigenfaces_at.yml");
	
	t = ((double)getTickCount() - t)/getTickFrequency();	
	cout << "Training took " << setprecision(3) << t << " s." << endl;
	
	exit(0);
}
