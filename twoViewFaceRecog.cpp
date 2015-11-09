#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

using namespace cv;
using namespace std;
using namespace face;

CascadeClassifier frontFaceDetector;
CascadeClassifier rightProfileDetector;

bool isFrontFaceDetected = false;
bool isProfileFaceDetected = false;

//~ bool static isRunOnce = true;

int label1 = -1;
int label2 = -1;
int a_1 = 0, b_1 = 0, c_1 = 0, d_1 = 0, e_1 = 0, f_1 = 0, g_1 = 0, h_1 = 0, i_1 = 0, j_1 = 0;
int a_2 = 0, b_2 = 0, c_2 = 0, d_2 = 0, e_2 = 0, f_2 = 0, g_2 = 0, h_2 = 0, i_2 = 0, j_2 = 0;
int maxLabel1 = 0;	
int maxLabel2 = 0;
int frontFrameCounter = 0;
int profileFrameCounter = 0;
double confidence1 = 0.0;
double confidence2 = 0.0;

string timeStamp;

void detectFrontFaces(Mat front_body_roi, vector<Rect>boundRectFront,
     size_t f, Mat image1, Mat frame1_gray, Ptr<FaceRecognizer> modelFront);
     
void detectProfileFaces(Mat profile_body_roi, vector<Rect>boundRectProfile,
	 size_t p, Mat image2, Mat frame2_gray, Ptr<FaceRecognizer> modelProfile);
	 
void recognizeFrontFaces(Mat face_scaled, Ptr<FaceRecognizer> modelFront);

void recognizeProfileFaces(Mat face_scaled, Ptr<FaceRecognizer> modelProfile);

void tallyFront(int label1, double confidence1);

void tallyProfile(int label2, double confidence2);

void compareAndAnalyze();

string databaseName[] = {
	"Alexander Co Abad",      //0
	"Ariane Aguilar",         //1
	"Jerome Cansado",         //2
	"Ed Lorence De Guzman",   //3
	"Christian Glenn Hatol",  //4
	"Gerard Lou Libby",       //5
	"Xavier Palomares",       //6
	"John Jhonas Primavera",  //7
	"Aeysol Rosaldo",         //8
	"Ma. Joanna Venus"        //9
};

string databaseStatus[] = {
	"Faculty",                //0
	"Student",                //1
	"Student",                //2
	"Student",                //3
	"Student",                //4
	"Student",                //5
	"Student",                //6
	"Student",                //7
	"Student",                //8
	"Student"                 //9
};

Mat backgroundSubtractFront(Mat &gray1, Ptr<BackgroundSubtractorMOG2> pMOG2_front)
{
	Mat foreground, background;
	
	pMOG2_front -> apply(gray1, foreground);
	pMOG2_front -> getBackgroundImage(background);
	
	//~ imshow("Background - Front", background);
	
	medianBlur(foreground, foreground, 9);
	Mat element = getStructuringElement(MORPH_RECT, Size(3,3), Point(1,1));
	Mat kernel;
	morphologyEx(foreground, foreground, MORPH_CLOSE, kernel, Point(-1,-1), 5);
	erode(foreground, foreground, element);
	dilate(foreground, foreground, element);
	
	//~ imshow("Foreground - Front", foreground);
		
	Mat threshold_output;
	
	threshold(foreground, threshold_output, 45, 255, THRESH_BINARY);
		             
	return threshold_output;
}

Mat backgroundSubtractProfile(Mat &gray2, Ptr<BackgroundSubtractorMOG2> pMOG2_profile)
{
	Mat foreground, background;
		
	pMOG2_profile -> apply(gray2, foreground);
	pMOG2_profile -> getBackgroundImage(background);
	
	//~ imshow("Background - Right Profile", background);
	
	medianBlur(foreground, foreground, 9);
	Mat element = getStructuringElement(MORPH_RECT, Size(3,3), Point(1,1));
	Mat kernel;
	morphologyEx(foreground, foreground, MORPH_CLOSE, kernel, Point(-1,-1), 5);
	erode(foreground, foreground, element);
	dilate(foreground, foreground, element);
	
	//~ imshow("Foreground - Right Profile", foreground);
		
	Mat threshold_output;
	
	threshold(foreground, threshold_output, 45, 255, THRESH_BINARY);
		             
	return threshold_output;
}

void detectFrontFaces(Mat front_body_roi, vector<Rect>boundRectFront,
     size_t f, Mat image1, Mat frame1_gray, Ptr<FaceRecognizer> modelFront)
{	
	vector<Rect> frontalFaces;				//Detected object(s)	
	float searchScaleFactor = 1.1;          //How many sizes to search
	int minNeighbors = 4;                   //Reliability vs many faces
	int flags = 0 | CASCADE_SCALE_IMAGE;    //Search for many faces
	Size minFeatureSize(30, 30);            //Smallest face size
	
	frontFaceDetector.detectMultiScale(front_body_roi, frontalFaces,
		searchScaleFactor, minNeighbors, flags, minFeatureSize);
		
	if(frontalFaces.size() != 0){	//If faces are detected
		isFrontFaceDetected = true;
		
		for(size_t i = 0; i < 1; i++)
		{
			int faces_y1 = frontalFaces[i].y + boundRectFront[f].y;
			
			int faces_y2 = frontalFaces[i].y + frontalFaces[i].height + boundRectFront[f].y;
			
			Point f1(frontalFaces[i].x + boundRectFront[f].x, faces_y1);
			Point f2(frontalFaces[i].x + frontalFaces[i].width + boundRectFront[f].x, faces_y2);
			rectangle(image1, f1, f2, Scalar(0,0,255), 2, 8, 0);
			
			//~ cout << "asdf: " << frontalFaces[i].x + frontalFaces[i].width + boundRectFront[f].x << endl;
			
			Rect ROI(f1, f2);
			Mat faceROI = frame1_gray(ROI);
			Mat face_scaled;
			
			if(!faceROI.empty())
			{
				frontFrameCounter++;
				
				if(faceROI.cols != 50 && faceROI.rows!= 50){
					resize(faceROI.clone(), face_scaled, Size(50,50));
				}
				else{
					face_scaled = faceROI.clone();
				}
				equalizeHist(face_scaled, face_scaled);
				imshow("Detected front face", face_scaled);
				if(frontFrameCounter == 1){
					time_t currentTime;
					struct tm *localTime;
					time( &currentTime );                   // Get the current time
					localTime = localtime( &currentTime );  // Convert the current time to the local time
				
					int Hour   = localTime->tm_hour;
					int Min    = localTime->tm_min;
					int Sec    = localTime->tm_sec;
								
					stringstream ss;
					
					ss << "Time: " << Hour << ":" << Min << ":" << Sec;
					timeStamp = ss.str();
					ss.str("");
				}
				
				recognizeFrontFaces(face_scaled, modelFront);
			}
		}
	}
	else{
		frontFrameCounter = 0;
		int array[10] = 
			{
				a_1, b_1, c_1, d_1, e_1, f_1, g_1, h_1, i_1, j_1
			};

		for(int i=0;i<10;i++)
		{
			if(array[i]>maxLabel1)
			maxLabel1=array[i]; 
		}
	}
}

void detectProfileFaces(Mat profile_body_roi, vector<Rect>boundRectProfile,
	 size_t p, Mat image2, Mat frame2_gray, Ptr<FaceRecognizer> modelProfile)
{	
	vector<Rect> profileFaces;				//Detected object(s)	
	float searchScaleFactor = 1.2;          //How many sizes to search
	int minNeighbors = 2;                   //Reliability vs many faces
	int flags = 0 | CASCADE_SCALE_IMAGE;    //Search for many faces
	Size minFeatureSize(30, 30);            //Smallest face size
	
	rightProfileDetector.detectMultiScale(profile_body_roi, profileFaces,
		searchScaleFactor, minNeighbors, flags, minFeatureSize);
		
	if(profileFaces.size() !=0){
		isProfileFaceDetected = true;
		
		for(size_t i = 0; i < 1; i++)
		{
			int faces_y1 = profileFaces[i].y + boundRectProfile[p].y;
			
			int faces_y2 = profileFaces[i].y + profileFaces[i].height + boundRectProfile[p].y;
			
			Point f1(profileFaces[i].x + boundRectProfile[p].x, faces_y1);
			Point f2(profileFaces[i].x + profileFaces[i].width + boundRectProfile[p].x, faces_y2);
			rectangle(image2, f1, f2, Scalar(0,0,255), 2, 8, 0);
			
			Rect ROI(f1, f2);
			Mat faceROI = frame2_gray(ROI);
			Mat face_scaled;
			
			profileFrameCounter++;
			stringstream ss;
				string file;
				ss << profileFrameCounter<<".pgm";
				file = ss.str();
				imwrite(file, faceROI);
			
						
			if(!faceROI.empty())
			{
				
				if(faceROI.cols != 50 && faceROI.rows!= 50){
					resize(faceROI, face_scaled, Size(50, 50));
				}
				else{
					face_scaled = faceROI;
				}			
					
				
				imshow("Detected profile face", face_scaled);
				recognizeProfileFaces(face_scaled, modelProfile);
			}
		}
	}
	else{
		profileFrameCounter = 0;
		int array[10] = 
			{
				a_2, b_2, c_2, d_2, e_2, f_2, g_2, h_2, i_2, j_2
			};

		for(int i=0;i<10;i++)
		{
			if(array[i]>maxLabel2)
			maxLabel2=array[i];
		}
	}
}

void recognizeFrontFaces(Mat face_scaled, Ptr<FaceRecognizer> modelFront)
{
	
	modelFront -> predict(face_scaled, label1, confidence1);
	
	tallyFront(label1, confidence1);
	
	//~ cout << "LabelFront: " << label1 << endl;
	//~ cout << "Confidence: " << confidence1 << endl;
}

void recognizeProfileFaces(Mat face_scaled, Ptr<FaceRecognizer> modelProfile)
{
	
	modelProfile -> predict(face_scaled, label2, confidence2);
	
	tallyProfile(label2, confidence2);
	
	//~ cout << "LabelSide: " << label2 << endl;
	//~ cout << "Confidence: " << confidence2 << endl;
}

//Functions tallyFront and tallyProfile tallies the number of recognized
//faces for a given video sequence or person. The highest number of tallied
//label will be set as maxLabel1 or maxLabel2 and will be considered as
//the predicted label of the system.

void tallyFront(int label1, double confidence1){
		if (label1 == 0) a_1++;
		if (label1 == 1) b_1++;
		if (label1 == 2) c_1++;
		if (label1 == 3) d_1++;
		if (label1 == 4) e_1++;
		if (label1 == 5) f_1++;
		if (label1 == 6) g_1++;
		if (label1 == 7) h_1++;
		if (label1 == 8) i_1++;
		if (label1 == 9) j_1++;
}

void tallyProfile(int label2, double confidence2){
		if (label2 == 0) a_2++;
		if (label2 == 1) b_2++;
		if (label2 == 2) c_2++;
		if (label2 == 3) d_2++;
		if (label2 == 4) e_2++;
		if (label2 == 5) f_2++;
		if (label2 == 6) g_2++;
		if (label2 == 7) h_2++;
		if (label2 == 8) i_2++;
		if (label2 == 9) j_2++;
}

void compareAndAnalyze()
{
	string nameText, statusText;
	
	if (maxLabel1 == maxLabel2){
		nameText = format("Name: ", maxLabel1);
		statusText = format("Status: ", maxLabel1);
		
		nameText.append(databaseName[maxLabel1]);
		statusText.append(databaseStatus[maxLabel1]);
	}	
	else{
		nameText = format("Name: Unknown");
		statusText = format("Status: Unknown");
	}
	
	isFrontFaceDetected = false;
	isProfileFaceDetected = false;
		
	//~ if (isRunOnce){
		//~ cout << nameText << endl << statusText << endl << timeStamp << endl
		     //~ << "-----------------------------" << endl;	
		//~ isRunOnce = false;
	//~ }
}

int main()	
{
	VideoCapture camera1, camera2;
	camera1.open("videos/ian2front.avi");
	camera2.open("videos/ian2side.avi");
		
	if(!camera1.isOpened()){
		cout << "Error opening front camera." << endl;
		exit(0);
	}
	
	if(!camera2.isOpened()){
		cout << "Error opening side camera." << endl;
		exit(0);
	}
	
	//Load the cascades
	frontFaceDetector.load("haarcascade_frontalface_default.xml");
	rightProfileDetector.load("haarcascade_profileface.xml");
	
	//Initialize MOG2 background subtraction 
	Ptr<BackgroundSubtractorMOG2> pMOG2front, pMOG2profile;
	pMOG2front = createBackgroundSubtractorMOG2(1000, 128, false);
	pMOG2profile = createBackgroundSubtractorMOG2(1000, 35, false);
	
	//Load the face recognizer algorithm
	Ptr<FaceRecognizer> modelFront = createFisherFaceRecognizer();
	modelFront -> load("frontFaceTrainer/eigenfaces_at.yml");
	Ptr<FaceRecognizer> modelProfile = createFisherFaceRecognizer();
	modelProfile -> load("sideFaceTrainer/eigenfaces_at.yml");	
	
	Mat frame1, frame2, frame1_gray, frame2_gray, image1, image2, thresholdOut;
	
	vector< vector<Point> > contours_front, contours_profile;
	vector<Vec4i> hierarchy;
				
	for(;;)
	{
		camera1 >> frame1;
		camera2 >> frame2;
		
		if(!frame1.empty() && !frame2.empty()){
			image1 = frame1.clone();
			image2 = frame2.clone();
			
			cvtColor(frame1, frame1_gray, CV_BGR2GRAY);
			cvtColor(frame2, frame2_gray, CV_BGR2GRAY);
			
			////////////////////////////////////////////////////////////
			
			Mat thresh_front = backgroundSubtractFront(frame1_gray, pMOG2front);
						
			findContours(thresh_front, contours_front, hierarchy,
				CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
				
			vector<Rect> boundRectFront(contours_front.size());			
			double areaFrontContours;
			
			for(size_t f = 0; f < contours_front.size(); f ++){
				areaFrontContours = contourArea(contours_front[f]);
								
				if(areaFrontContours > 850 && areaFrontContours < 20000){
					//~ cout << "Front contour area: " << areaFrontContours << endl;
					boundRectFront[f] = boundingRect(contours_front[f]);
					rectangle(image1, boundRectFront[f].tl(), boundRectFront[f].br(),
						Scalar(0,255,0), 2, 8, 0);
					Rect roi_1(boundRectFront[f].tl(), boundRectFront[f].br());
					Mat front_body_roi = frame1_gray(roi_1);		
					detectFrontFaces(front_body_roi, boundRectFront, f, image1, frame1_gray, modelFront);			
				}
			}
			
			////////////////////////////////////////////////////////////
			
			Mat thresh_profile = backgroundSubtractProfile(frame2_gray, pMOG2profile);
			
			findContours(thresh_profile, contours_profile, hierarchy,
				CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
				
			vector<Rect> boundRectProfile(contours_profile.size());			
			double areaProfileContours;
			
			for(size_t p = 0; p < contours_profile.size(); p ++){
				areaProfileContours = contourArea(contours_profile[p]);
								
				if(areaProfileContours > 1500 && areaProfileContours < 80000){
					//~ cout << "Profile contour area: " << areaProfileContours << endl;
					boundRectProfile[p] = boundingRect(contours_profile[p]);
					rectangle(image2, boundRectProfile[p].tl(), boundRectProfile[p].br(),
						Scalar(0,255,0), 2, 8, 0);
					Rect roi_2(boundRectProfile[p].tl(), boundRectProfile[p].br());
					Mat profile_body_roi = frame2_gray(roi_2);		
					detectProfileFaces(profile_body_roi, boundRectProfile, p, image2, frame2_gray, modelProfile);			
				}
			}
			
			////////////////////////////////////////////////////////////
			
			if (isFrontFaceDetected == true || isProfileFaceDetected == true)
			{
				compareAndAnalyze();
				cout << "Compared" << endl;
			}
											
			imshow("Front face detection", image1);
			imshow("Right profile detection", image2);
		}
		
		int c = waitKey(27);
		if(27 == char(c)){
			break;
		}			
				
	}
	
	exit(0);
}
