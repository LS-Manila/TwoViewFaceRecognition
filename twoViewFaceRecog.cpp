/***********************************************************************
 *  	De La Salle University - Science and Technology Complex        *
 *              FACE RECOGNITION USING TWO-VIEW IP CAMERA              *
 *                                                                     *
 * Jerome C. Cansado	Christian Glenn T. Hatol   Jhonas N. Primavera *
 *                                                                     *
 *                 Adviser: Engr. Melvin Cabatuan                      *
 **********************************************************************/

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <fstream>

using namespace std;
using namespace cv;
using namespace face;

void backgroundSubtractionFrontal(Mat frame1, Mat image1, 
	Ptr<BackgroundSubtractorMOG2> pMOG1, Ptr<FaceRecognizer> model1);
void backgroundSubtractionProfile(Mat frame2, Mat image2,
	Ptr<BackgroundSubtractorMOG2> pMOG2, Ptr<FaceRecognizer> model2);
void faceDetectionFrontal(Mat frame1, Mat frontalBodyROI, Mat image1,
	vector<Rect> boundRect,	Ptr<FaceRecognizer> model1, size_t i);
void faceDetectionProfile(Mat frame2, Mat profileBodyROI, Mat image2, 
	vector<Rect> boundRect, Ptr<FaceRecognizer> model2, size_t i);
void eyeDetectionFrontal(Mat faceROI, vector<Rect> boundRect, Mat image1, Mat frame1,
	vector<Rect> frontalFaces, size_t i, size_t j, Ptr<FaceRecognizer> model1);
void recognizeFrontal(string fileName1, int Hour, int Min, int Sec, 
	Mat image1,	Ptr<FaceRecognizer> model1);
void recognizeProfile(string fileName2, Ptr<FaceRecognizer> model2);
void alignRotateCrop(Mat frame1, Mat image1, Ptr<FaceRecognizer> model1);
void writeData(string nameText, string statusText, string timeStamp);
void compareResults(int label1, int label2, double cofidence1, 
	double confidence2, int Hour, int Min, int Sec);

CascadeClassifier frontalDetector;
CascadeClassifier profileDetector;
CascadeClassifier eyeDetector;

Point eye_left;
Point eye_right;

int framesPerDetection;
int label1 = -1;
int label2 = -1;
double confidence1 = 0.0;
double confidence2 = 0.0;
stringstream ssTime;
string timeStamp;

string databaseName[] = {
	"Alexander Co Abad",
	"Ariane Aguilar",
	"Jerome Cansado",
	"Ed Lorence De Guzman",
	"Christian Glenn Hatol",
	"Gerard Lou Libby",
	"Xavier Palomares",
	"John Jhonas Primavera",
	"Aeysol Rosaldo",
	"Ma. Joanna Venus"
};

string databaseStatus[] = {
	"Faculty",
	"Student",
	"Student",
	"Student",
	"Student",
	"Student",
	"Student",
	"Student",
	"Student",
	"Student"
};

float Distance(Point p1, Point p2)
{
	int dx = abs(p2.x - p1.x);
	int dy = abs(p2.y - p1.y);
	return sqrt(dx*dx + dy*dy);
}

 Mat rotate(Mat& image, double angle, Point centre)
 {
	Point2f src_center(centre.x, centre.y);
	angle = angle*180.0/3.14157;
	Mat rot_matrix = getRotationMatrix2D(src_center, angle, 1.0);
	 
	Mat rotated_image(Size(image.size().height, image.size().width), 
		image.type());
		
	warpAffine(image, rotated_image, rot_matrix, image.size());
	
	return(rotated_image);
 }

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
		imshow("Background - Front", backgroundFrontal);
	}
	
	medianBlur(foregroundFrontal, foregroundFrontal, 9);
	erode(foregroundFrontal, foregroundFrontal, Mat());
	dilate(foregroundFrontal, foregroundFrontal, Mat());
	imshow("Foreground - Front", foregroundFrontal);
	
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
	//~ imshow("Moving object 1", frontalBodyROI);
	
	vector<Rect> frontalFaces;				//Detected object(s)	
	float searchScaleFactor = 1.1;          //How many sizes to search
	int minNeighbors = 4;                   //Reliability vs many faces
	int flags = 0 | CASCADE_SCALE_IMAGE;    //Search for many faces
	Size minFeatureSize(30, 30);            //Smallest face size
	
	frontalDetector.detectMultiScale(frontalBodyROI, frontalFaces, 
		searchScaleFactor, minNeighbors, flags, minFeatureSize);
		
	if(frontalFaces.size() == 0){   //If the program does not detect any
		framesPerDetection = 0;     //faces, set the counter to zero.
	}	
		
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
		equalizeHist(faceROI, faceROI);
		imshow("Detected face", faceROI);
		
		eyeDetectionFrontal(faceROI, boundRect, image1, frame1, 
			frontalFaces, i, j, model1);
		
		framesPerDetection++;
	}
}

void eyeDetectionFrontal(Mat faceROI, vector<Rect> boundRect, Mat image1, Mat frame1,
	vector<Rect> frontalFaces, size_t i, size_t j, Ptr<FaceRecognizer> model1)
{
	vector<Rect> eyes;
	
	eyeDetector.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
		Size(30, 30));
		
	//If two (2) eyes were detected...
	//~ cout << eyes.size() << endl; 
	if (eyes.size() == 2){
		for (size_t k = 0; k < 2; k++){
			Point eye_center(boundRect[i].x + frontalFaces[j].x + eyes[1-k].x + 
				eyes[1-k].width/2, boundRect[i].y + frontalFaces[j].y + eyes[1-k].y + 
				eyes[1-k].height/2);
		
			if (j==0) //Left eye
			{ 
				eye_left.x = eye_center.x;
				eye_left.y = eye_center.y;
			}
			if (j==1) //Right eye
			{  
				eye_right.x = eye_center.x;
				eye_right.y = eye_center.y;
			}
		}		
	}
	
	//Sometimes, the detected eyes are reversed so switch them
	if (eye_right.x < eye_left.x){
		int tmpX = eye_right.x;
		int tmpY = eye_right.y;
		eye_right.x = eye_left.x;
		eye_right.y = eye_left.y;
		eye_left.x = tmpX;
		eye_left.y = tmpY;
	}
	
	circle(image1, eye_right, 4, Scalar(0,255,255), -1, 8, 0);
	circle(image1, eye_left, 4, Scalar(0,255,255), -1, 8, 0);
	
	alignRotateCrop(frame1, image1, model1);
	
}

void alignRotateCrop(Mat frame1, Mat image1, Ptr<FaceRecognizer> model1)
{
	//Offset percentage
	Point offset_pct;
	offset_pct.x = 0.2*100;
	offset_pct.y = offset_pct.x;
	
	//Size of new picture
	Point dest_sz;
	dest_sz.x = 50;
	dest_sz.y = dest_sz.x;
	
	//Calculate offsets in original image
	int offset_h = (offset_pct.x*dest_sz.x/100);
	int offset_v = (offset_pct.y*dest_sz.y/100);
			
	//Get the direction
	Point eye_direction;
	eye_direction.x = eye_right.x - eye_left.x;
	eye_direction.y = eye_right.y - eye_left.y;
			
	//Calculate rotation angle in radians
	float rotation = atan2((float)(eye_direction.y),
	(float)(eye_direction.x));
					
	//Distance between them
	float dist = Distance(eye_left, eye_right);
				
	//Calculate the reference eye-width
	int reference = dest_sz.x - 2*offset_h;
				
	//Scale factor
	float scale = dist/(float)reference;
				
	//Rotate original image around the left eye
	Mat frameRot = rotate(frame1, (double)rotation, eye_left);
		
	imshow("Rotated image", frameRot);
			
	//Crop the rotated image
	Point crop_xy;		//Top left corner coordinates
	crop_xy.x = eye_left.x - scale*offset_h;
	crop_xy.y = eye_left.y - scale*offset_v;
					
	Point crop_size;
	crop_size.x = dest_sz.x*scale;	//Cropped image width
	crop_size.y = dest_sz.y*scale;	//Cropped image height
				
	//Crop the full image 
	Rect myROI(crop_xy.x, crop_xy.y, crop_size.x, crop_size.y);
	if((crop_xy.x + crop_size.x < frameRot.size().width) &&
		(crop_xy.y + crop_size.y < frameRot.size().height)){
			frameRot = frameRot(myROI);
		}
	else{
		cout << "Error cropping" << endl;
		exit(0);
	}
	
	Mat faceSmallFrontal;
	
	if(!frameRot.empty()){
		if(frameRot.cols != dest_sz.x){
			resize(frameRot, faceSmallFrontal, Size(dest_sz));	
		}
		else{
			faceSmallFrontal = frameRot;
		}
		equalizeHist(faceSmallFrontal, faceSmallFrontal);
		imshow("Cropped image", faceSmallFrontal);	
		if(framesPerDetection == 4){ //Capture the 4th frame
			time_t currentTime;
			struct tm *localTime;
			
			imshow("Saved image", faceSmallFrontal);

			time( &currentTime );                   // Get the current time
			localTime = localtime( &currentTime );  // Convert the current time to the local time
		
			int Hour   = localTime->tm_hour;
			int Min    = localTime->tm_min;
			int Sec    = localTime->tm_sec;
						
			stringstream ss;
			string fileName1;
			ss << Hour << "-" << Min << "-" << Sec << "-frontal" << ".pgm";
			fileName1 = ss.str();
			ss.str("");
			imwrite(fileName1, faceSmallFrontal);
			
			recognizeFrontal(fileName1, Hour, Min, Sec, image1, model1);
		}					
	}
}

void recognizeFrontal(string fileName1, int Hour, int Min, int Sec, 
	Mat image1,	Ptr<FaceRecognizer> model1)
{
	Mat scaledFace1 = imread(fileName1, CV_LOAD_IMAGE_UNCHANGED);
	
	model1 -> predict(scaledFace1, label1, confidence1);
	
	compareResults(label1, label2, confidence1, confidence2, Hour, Min,
		Sec);
}

void compareResults(int label1, int label2, double cofidence1, 
	double confidence2, int Hour, int Min, int Sec)
{
	int label = -1;
	double confidence = 0.0;
	
	if (label1 != -1 && label2 != -1 && confidence1 != 0 && confidence2 != 0){
		if (confidence1 < confidence2){
			label = label1;
		}	
		else{
			label = label2;
		}
		
		string nameText = format("Name: ", label);
		string statusText = format("Status: ", label);
			
		ssTime << "Time: " << Hour << ":" << Min << ":" << Sec;
		timeStamp = ssTime.str();
		ssTime.str("");
		
		if(confidence < 1580){
			if(label >= 0 && label <= 9){
				nameText.append(databaseName[label]);
				statusText.append(databaseStatus[label]);
			}
		}
		else{
			nameText.append("Unknown");
			statusText.append("Unknown");		
	}
	
	cout << nameText << endl << statusText << endl << timeStamp << endl
		 << "-----------------------------" << endl;
		 
	writeData(nameText, statusText, timeStamp);
	}
}

void writeData(string nameText, string statusText, string timeStamp){
	
	fstream oStrm;
	oStrm.open("data.txt", fstream::in | fstream::out | fstream::app);
		
	oStrm << nameText << endl << statusText << endl << timeStamp 
		  << endl << "-----------------------------" << endl;
	oStrm.close();	
}

void backgroundSubtractionProfile(Mat frame2, Mat image2, 
	Ptr<BackgroundSubtractorMOG2> pMOG2, Ptr<FaceRecognizer> model2)
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
				boundRect, model2, i);
		}
	}
}

void faceDetectionProfile(Mat frame2, Mat profileBodyROI, Mat image2, 
	vector<Rect> boundRect, Ptr<FaceRecognizer> model2, size_t i)
{
	equalizeHist(profileBodyROI, profileBodyROI);
	//~ imshow("Moving object 2", profileBodyROI);
	
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
		
		if(!faceROI.empty()){
			if(faceROI.cols != 50){
				resize(faceROI.clone(), faceSmallProfile, Size(50, 50));
			}
			else{
				faceSmallProfile = faceROI.clone();
			}
			//~ imshow("Cropped image", faceSmallProfile);	
			//~ if(framesPerDetection == 2){ //Capture the 2nd frame
				time_t currentTime;
				struct tm *localTime;
				
				//~ imshow("Saved image", faceSmallProfile);

				time( &currentTime );                   // Get the current time
				localTime = localtime( &currentTime );  // Convert the current time to the local time
			
				int Hour   = localTime->tm_hour;
				int Min    = localTime->tm_min;
				int Sec    = localTime->tm_sec;
							
				stringstream ss2, ssTimeTemp;
				string fileName2, timeStampTemp;
				ssTimeTemp << "Time: " << Hour << ":" << Min << ":" << Sec;
				timeStampTemp = ssTimeTemp.str();
				ssTimeTemp.str("");
				if (timeStamp.compare(timeStampTemp) == 0){
					ss2 << Hour << "-" << Min << "-" << Sec << "-profile" << ".pgm";
					fileName2 = ss2.str();
					ss2.str("");
					imwrite(fileName2, faceSmallProfile);					
					recognizeProfile(fileName2, model2);
				}
			//~ }
			//~ imshow("Detected face", faceSmallProfile);
		}
		//~ else destroyWindow("Detected face");
	}
}

void recognizeProfile(string fileName2, Ptr<FaceRecognizer> model2)
{	
	Mat scaledFace2 = imread(fileName2, CV_LOAD_IMAGE_UNCHANGED);
	model2 -> predict(scaledFace2, label2, confidence2);
	
	//~ cout << confidence2 << endl;
	
	//~ string nameText = format("Name: ", label2);
	//~ string statusText = format("Status: ", label2);
	//~ 
	//~ //To avoid false recognition, accept only the predictions within
	//~ //a certain confidence level. 
	//~ if(confidence2 < 1580){
		//~ if(label2 >= 0 && label2 <= 9){
			//~ nameText.append(databaseName[label2]);
			//~ statusText.append(databaseStatus[label2]);
		//~ }
	//~ }
	//~ else{
		//~ nameText.append("Unknown");
		//~ statusText.append("Unknown");
	//~ }
}



int main(){
	VideoCapture camera1, camera2;
	camera1.open(0);
	camera2.open("sidecam.avi");
	
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
	if (!eyeDetector.load("haarcascade_lefteye_2splits.xml")){
		cout << "--(!)Error loading eyeDetector cascade." << endl;
		exit(0);
	}	
		
	//Load the face recognize algorithm
	Ptr<FaceRecognizer> model1 = createEigenFaceRecognizer();
	model1 -> load("frontFaceTrainer/eigenfaces_at.yml");
	Ptr<FaceRecognizer> model2 = createEigenFaceRecognizer();
	model2 -> load("sideFaceTrainer/eigenfaces_at.yml");
	
	Mat frame1, image1;
	Mat frame2, image2;
	
	for(;;){
		camera1 >> frame1;
		camera2 >> frame2;
		if(!frame1.empty() && !frame2.empty()){
			image1 = frame1.clone();
			image2 = frame2.clone();
			
			backgroundSubtractionFrontal(frame1, image1, pMOG1, model1);
			backgroundSubtractionProfile(frame2, image2, pMOG2, model2);
			
			imshow("Front face recognition", image1);
			//~ imshow("Profile face recognition", image2);
		}
		else exit(0);
		int c = waitKey(27);
		if(27 == char(c)){
			break;
		}
	}
	exit(0);
}
