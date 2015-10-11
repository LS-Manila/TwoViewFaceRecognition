#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

using namespace std;
using namespace cv;
using namespace face;

void backgroundSubtractionFrontal(Mat frame1, Mat image1, 
	Ptr<BackgroundSubtractorMOG2> pMOG1, Ptr<FaceRecognizer> model1);
void backgroundSubtractionProfile(Mat frame2, Ptr<BackgroundSubtractorMOG2> pMOG2,
	Mat image2);
void faceDetectionFrontal(Mat frame1, Mat frontalBodyROI, Mat image1,
	vector<Rect> boundRect,	Ptr<FaceRecognizer> model1, size_t i);
void faceDetectionProfile(Mat frame2, Mat profileBodyROI, Mat image2, 
	vector<Rect> boundRect, size_t i);
void recognizeFrontal(Mat faceSmallFrontal, Mat image1, Ptr<FaceRecognizer> model1);

CascadeClassifier frontalDetector;
CascadeClassifier profileDetector;

Point eyeLeft;
Point eyeRight;

string databaseName[] = {
	"John Jhonas Primavera",
	"Christian Glenn Hatol",
	"Jerome Cansado",
	"Xavier Palomares",
	"Ed Lorence De Guzman",
	"Gerard Lou Libby",
	"Ma. Joanna Venus"
};

void backgroundSubtractionFrontal(Mat frame1, Mat image1, 
	Ptr<BackgroundSubtractorMOG2> pMOG1, Ptr<FaceRecognizer> model1)
{
	Mat foregroundFrontal, backgroundFrontal;
	
	cvtColor(frame1, frame1, CV_RGB2GRAY);
	
	//Compute the foreground mask and the background image for front 
	//face detection
	pMOG1 -> apply(frame1, foregroundFrontal);
	pMOG1 -> getBackgroundImage(backgroundFrontal);
	
	if(!backgroundFrontal.empty()){
		//~ imshow("Background - Front", backgroundFrontal);
	}
	
	medianBlur(foregroundFrontal, foregroundFrontal, 9);
	erode(foregroundFrontal, foregroundFrontal, Mat());
	dilate(foregroundFrontal, foregroundFrontal, Mat());
	//~ imshow("Foreground - Front", foregroundFrontal);
	
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat thresholdOutput;
	
	threshold(foregroundFrontal, thresholdOutput, 45, 255, THRESH_BINARY);
	findContours(thresholdOutput, contours, hierarchy, CV_RETR_TREE,
		CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	
	vector< vector<Point> > contoursPoly(contours.size());
	vector<Rect> boundRect(contours.size());
	
	Mat drawing = Mat::zeros(thresholdOutput.size(), CV_8UC3);
	double area;
	int thresholdArea = 9500;
	
	for(size_t i = 0; i < contours.size(); i++){
		area = contourArea(contours[i]);
		
		if(area > thresholdArea){
			drawContours(drawing, contours, i, Scalar(255, 0, 0), 2, 8,
				hierarchy, 0, Point());
			boundRect[i] = boundingRect(contours[i]);
			rectangle(image1, boundRect[i].tl(), boundRect[i].br(),
				Scalar(0, 255, 0), 2, 8, 0);
			Rect bodyROI(boundRect[i].tl(), boundRect[i].br());
			Mat frontalBodyROI = frame1(bodyROI);
			faceDetectionFrontal(frame1, frontalBodyROI, image1, 
				boundRect, model1, i);
		}
	}
}

void faceDetectionFrontal(Mat frame1, Mat frontalBodyROI, Mat image1, 
	vector<Rect> boundRect, Ptr<FaceRecognizer> model1, size_t i)
{
	equalizeHist(frontalBodyROI, frontalBodyROI);
	imshow("Moving object 1", frontalBodyROI);
	
	vector<Rect> frontalFaces;
	
	frontalDetector.detectMultiScale(frontalBodyROI, frontalFaces, 1.1,
		4, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		
	for(size_t j = 0; j < frontalFaces.size(); j++){
		int faces_y1 = frontalFaces[j].y + boundRect[i].y;
		if (faces_y1 < 0){
			faces_y1 = 0;
		}
		int faces_y2 = frontalFaces[j].y + frontalFaces[j].height + boundRect[i].y;
		if (faces_y2 > frontalBodyROI.rows){
			faces_y2 = frontalBodyROI.rows;
		}
		
		Point f1(frontalFaces[j].x + boundRect[i].x, faces_y1);
		Point f2(frontalFaces[j].x + frontalFaces[j].width + boundRect[i].x, faces_y2);
		rectangle(image1, f1, f2, Scalar(0, 0, 255), 2, 8, 0);
		
		Rect ROI(f1, f2);
		Mat faceROI = frame1(ROI);
		Mat faceSmallFrontal;
		int faceScale = 50;
		
		if(!faceROI.empty()){
			if(faceROI.cols != faceScale){
				resize(faceROI.clone(), faceSmallFrontal, Size(faceScale, faceScale));
			}
			else{
				faceSmallFrontal = faceROI.clone();
			}
			imshow("Detected face", faceSmallFrontal);
		}
		else destroyWindow("Detected face");
		recognizeFrontal(faceSmallFrontal, image1, model1);
	}
}

void recognizeFrontal(Mat faceSmallFrontal, Mat image1, Ptr<FaceRecognizer> model1)
{
	int label = -1;
	double confidence = 0.0;
	
	model1 -> predict(faceSmallFrontal, label, confidence);
	
	string nameText = format("Name: ", label);
	
	if(confidence < 2300){
		if(label >= 0 && label <= 6){
			nameText.append(databaseName[label]);
		}
	}
	else nameText.append("Unknown");
	
	putText(image1, nameText, Point(5, 30), FONT_HERSHEY_COMPLEX, 0.5,
		Scalar(0, 0, 0), 1, LINE_8);
}

void backgroundSubtractionProfile(Mat frame2, Ptr<BackgroundSubtractorMOG2> pMOG2,
	Mat image2)
{
	Mat foregroundProfile, backgroundProfile;
	
	cvtColor(frame2, frame2, CV_RGB2GRAY);
	
	//Compute the foreground mask and the background image for side 
	//face detection
	pMOG2 -> apply(frame2, foregroundProfile);
	pMOG2 -> getBackgroundImage(backgroundProfile);
	
	if(!backgroundProfile.empty()){
		//~ imshow("Background - Profile", backgroundProfile);
	}
	
	medianBlur(foregroundProfile, foregroundProfile, 9);
	erode(foregroundProfile, foregroundProfile, Mat());
	dilate(foregroundProfile, foregroundProfile, Mat());
	//~ imshow("Foreground - Profile", foregroundProfile);
	
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat thresholdOutput;
	
	threshold(foregroundProfile, thresholdOutput, 45, 255, THRESH_BINARY);
	findContours(thresholdOutput, contours, hierarchy, CV_RETR_TREE,
		CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	
	vector< vector<Point> > contoursPoly(contours.size());
	vector<Rect> boundRect(contours.size());
	
	Mat drawing = Mat::zeros(thresholdOutput.size(), CV_8UC3);
	double area;
	int thresholdArea = 9500;
	
	for(size_t i = 0; i < contours.size(); i++){
		area = contourArea(contours[i]);
		
		if(area > thresholdArea){
			drawContours(drawing, contours, i, Scalar(255, 0, 0), 2, 8,
				hierarchy, 0, Point());
			boundRect[i] = boundingRect(contours[i]);
			rectangle(image2, boundRect[i].tl(), boundRect[i].br(),
				Scalar(0, 255, 0), 2, 8, 0);
			Rect bodyROI(boundRect[i].tl(), boundRect[i].br());
			Mat profileBodyROI = frame2(bodyROI);
			faceDetectionProfile(frame2, profileBodyROI, image2, 
				boundRect, i);
		}
	}
}

void faceDetectionProfile(Mat frame2, Mat profileBodyROI, Mat image2, 
	vector<Rect> boundRect, size_t i)
{
	equalizeHist(profileBodyROI, profileBodyROI);
	imshow("Moving object 2", profileBodyROI);
	
	vector<Rect> profileFaces;
	
	frontalDetector.detectMultiScale(profileBodyROI, profileFaces, 1.1,
		4, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		
	for(size_t j = 0; j < profileFaces.size(); j++){
		int faces_y1 = profileFaces[j].y + boundRect[i].y;
		if (faces_y1 < 0){
			faces_y1 = 0;
		}
		int faces_y2 = profileFaces[j].y + profileFaces[j].height + boundRect[i].y;
		if (faces_y2 > profileBodyROI.rows){
			faces_y2 = profileBodyROI.rows;
		}
		
		Point f1(profileFaces[j].x + boundRect[i].x, faces_y1);
		Point f2(profileFaces[j].x + profileFaces[j].width + boundRect[i].x, faces_y2);
		rectangle(image2, f1, f2, Scalar(0, 0, 255), 2, 8, 0);
		
		Rect ROI(f1, f2);
		Mat faceROI = frame2(ROI);
		Mat faceSmallProfile;
		int faceScale = 50;
		
		if(!faceROI.empty()){
			if(faceROI.cols != faceScale){
				resize(faceROI.clone(), faceSmallProfile, Size(faceScale, faceScale));
			}
			else{
				faceSmallProfile = faceROI.clone();
			}
			imshow("Detected face", faceSmallProfile);
		}
		else destroyWindow("Detected face");
	}
}

int main(){
	VideoCapture camera1, camera2;
	camera1.open("MVI_5658.MOV");
	camera2.open("front.mp4");
	
	//Initialize MOG2 background subtraction for front face (pMOG1) and
	//profile face (pMOG2)
	Ptr<BackgroundSubtractorMOG2> pMOG1, pMOG2;
	pMOG1 = createBackgroundSubtractorMOG2(1000, 64, false);
	pMOG2 = createBackgroundSubtractorMOG2(1000, 64, false);
	
	//Load the cascades
	if (!frontalDetector.load("haarcascade_frontalface_default.xml")){
		cout << "--(!)Error loading faceDetector cascade." << endl;
		exit(0);
	}	
	if (!profileDetector.load("haarcascade_profileface.xml")){
		cout << "--(!)Error loading faceDetector cascade." << endl;
		exit(0);
	}
	
	//Load the face recognize algorithm
	Ptr<FaceRecognizer> model1 = createEigenFaceRecognizer();
	model1 -> load("frontFaceTrainer/eigenfaces_at.yml");
	
	Mat frame1, image1;
	Mat frame2, image2;
	
	for(;;){
		camera1 >> frame1;
		camera2 >> frame2;
		if(!frame1.empty() && !frame2.empty()){
			image1 = frame1.clone();
			image2 = frame2.clone();
			
			backgroundSubtractionFrontal(frame1, image1, pMOG1, model1);
			backgroundSubtractionProfile(frame2, pMOG2, image2);
			
			imshow("Front face recognition", image1);
			imshow("Profile face recognition", image2);
		}
		else exit(0);
		int c = waitKey(27);
		if(27 == char(c)){
			break;
		}
	}
	exit(0);
}
