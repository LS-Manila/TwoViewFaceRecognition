#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

using namespace std;
using namespace cv;
using namespace cv::face;

void faceDetection(size_t i, vector<Rect> boundRect, Mat body_roi, Mat frame_gry,
	Mat image, Ptr<face::FaceRecognizer> model);
void eyeDetection(Mat face_small, size_t h, vector<Rect> faces);
void recognize(Mat face_small, Ptr<face::FaceRecognizer> model,
	Rect faceROI, Mat image);
	
CascadeClassifier faceDetector;
CascadeClassifier profileDetector;
CascadeClassifier eyeDetector;

Point eyeLeft;
Point eyeRight;

string database_name[] = {
	"John Jhonas Primavera",
	"Christian Glenn Hatol",
	"Jerome Cansado",
	"Unknown"
};

int main(){	
	
	VideoCapture camera1, camera2;
	camera1.open("videos/MVI_5658.MOV");
	//~ camera1.open(0);
	//camera2.open("2.mp4");
	
	namedWindow("Background", 0);
	resizeWindow("Background", 640, 360);
	
	namedWindow("Detection", 0);
	resizeWindow("Detection", 640, 360);
	
	Ptr<BackgroundSubtractorMOG2> pMOG;
	pMOG = createBackgroundSubtractorMOG2(1000, //Length of the history
										  64, //varThreshold
										  false //bShadowDetection
										  );
	
	//Load the cascades
	if (!faceDetector.load("haarcascade_frontalface_default.xml")){
		cout << "--(!)Error loading faceDetector cascade." << endl;
		exit(0);
	}	
	if (!profileDetector.load("haarcascade_profileface.xml")){
		cout << "--(!)Error loading faceDetector cascade." << endl;
		exit(0);
	}
	if (!eyeDetector.load("haarcascade_eye.xml")){
		cout << "--(!)Error loading eyeDetector cascade." << endl;
		exit(0);
	}
		
	//Load the face recognizer algorithm	
	Ptr<face::FaceRecognizer> model = face::createEigenFaceRecognizer();
	model -> load("frontTrainerLBPH/eigenfaces_at.yml");
	
	Mat frame, image, frame_gry, foreground, background;
	Mat frame2;	
	for (;;){
		camera1 >> frame;
		//camera2 >> frame2;
		if (!frame.empty()){
			//imshow("Side view", frame2);
			//Clone the original image (cv::Mat frame) to cv::Mat image
			image = frame.clone();
			
			//Convert capture frame to grayscale
			cvtColor(frame, frame_gry, CV_RGB2GRAY);
			
			//Compute the foreground mask and the background image
			pMOG->apply(frame_gry, foreground);			
			pMOG->getBackgroundImage(background);
			
			if(!background.empty()){
				imshow("Background", background);
			}
			
			//Blur, erode, then dilate the foreground image
			//threshold(foreground, foreground, 200, 255, THRESH_BINARY);
			medianBlur(foreground, foreground, 9);
			erode(foreground, foreground, Mat());
			dilate(foreground, foreground, Mat());
			imshow("Foreground", foreground);
			
			vector< vector<Point> > contours;			
			vector<Vec4i> hierarchy;			
			Mat threshold_output;
			threshold(foreground, threshold_output, 45, 255, THRESH_BINARY);	
			findContours(threshold_output, contours, hierarchy, CV_RETR_TREE,
						 CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
									 
			vector< vector<Point> > contours_poly(contours.size());
			vector<Rect> boundRect(contours.size());
			
			Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
			double area;
			double threshold_area = 9500;
			for(size_t i = 0; i < contours.size(); i++)
			{
				area = contourArea(contours[i]);
				
				//cout << contours[i].x() << endl;
				if(area > threshold_area)
				{			
					//cout << "area: " << area << endl;	
					drawContours(drawing, contours, i, Scalar(255, 0, 0), 2,
								 8, hierarchy, 0, Point());
					
					boundRect[i] = boundingRect(contours[i]);
					rectangle(image, boundRect[i].tl(), boundRect[i].br(),
							  Scalar(0,255,0),  2, 8,0);
										
					Rect bodyROI(boundRect[i].tl(), boundRect[i].br());
					Mat body_roi = frame_gry(bodyROI);
					faceDetection(i, boundRect, body_roi, frame_gry,
								  image, model);										
				}								
			}				
			imshow("Detection", image);
			//imshow("Edges", drawing);
			
		}
		else exit(1);
		int c = waitKey(27);
        if (27 == char(c)){
            break;
        }
	}
	
	return 0;
}

void faceDetection(size_t i, vector<Rect> boundRect, Mat body_roi, Mat frame_gry,
				   Mat image, Ptr<face::FaceRecognizer> model)
{
	equalizeHist(body_roi, body_roi);
	//~ imshow("adsafsdf",body_roi);
	vector<Rect> faces;
	
	faceDetector.detectMultiScale(body_roi, faces, 1.1, 4, 
		         0 | CASCADE_SCALE_IMAGE, Size(30,30));
		         
	for(size_t h = 0; h < faces.size(); h++){	
		
		int faces_y1 = faces[h].y + boundRect[i].y;
		if (faces_y1 < 0){
			faces_y1 = 0;
		}
		int faces_y2 = faces[h].y + faces[h].height + boundRect[i].y;
		if (faces_y2 > body_roi.rows){
			faces_y2 = body_roi.rows;
		}
		Point f1(faces[h].x + boundRect[i].x, faces_y1);
		Point f2(faces[h].x + faces[h].width + boundRect[i].x, faces_y2);
		
		Rect faceROI(f1, f2);
		
		Mat face_roi = frame_gry(faceROI);
		//~ if(!face_roi.empty()){
		//~ imshow("Detected face", face_roi);}
		
		rectangle(image, f1, f2, Scalar(0, 0, 255), 2, 8, 0);	
		
		Mat face_small;
		int face_scale = 50;
	
		if(!face_roi.empty()){
			if(face_roi.cols != face_scale){
				resize(face_roi.clone(), face_small, Size(face_scale,face_scale));
			}
			else {
				face_small = face_roi.clone();
			}
			eyeDetection(face_small, h, faces);
			imshow("Detected face", face_small);
		}
		
		else destroyWindow("Detected face");
		recognize(face_small, model, faceROI, image);
	}	
}

void eyeDetection(Mat face_small, size_t h, vector<Rect> faces){
	vector<Rect> eyes;
	
	eyeDetector.detectMultiScale(face_small, eyes, 1.4, 2, 
		        0 | CASCADE_SCALE_IMAGE, Size(30,30));
		        //~ cout << eyes.size() << endl;
	if (eyes.size() == 2){
	//Detect eyes
	for (size_t i = 0; i < 2; i++){
		Point eye_center(faces[h].x + eyes[1-i].x + eyes[1-i].width/2,
			faces[h].y + eyes[1-i].y + eyes[1-i].height/2);
			
		//~ if (i == 0) //Left eye
		//~ {
			//~ eyeLeft.x = eye_center.x;
			//~ eyeLeft.y = eye_center.y;
		//~ }
		//~ 
		//~ if (i == 1) //Right eye
		//~ {
			//~ eyeRight.x = eye_center.x;
			//~ eyeRight.y = eye_center.y;
		//~ }
	}
}
}

void recognize(Mat face_small, Ptr<face::FaceRecognizer> model, 
	Rect faceROI, Mat image){
	
	int label = -1;
	double confidence = 0.0;
	
	model -> predict(face_small, label, confidence);
	
	//~ string result_message = format("Predicted class = %d / Confidence = %f",
							//~ label,confidence);
	//~ cout << result_message << endl;
	//~ 
	cout << "confidence: " << confidence << endl;
	//~ 
	string name_text = format("Name: ", label);
	//~ //cout << prediction << endl;
	
	if (confidence < 2300){
	if (label >= 0 && label <=3){
			name_text.append(database_name[label]);
		}}
	else name_text.append("Unknown");
	
	//~ int pos_x = max(faceROI.tl().x - 10, 0);
	//~ int pos_y = max(faceROI.tl().y - 10, 0);
	//~ 
	//~ putText(image, name_text, Point(pos_x,pos_y), FONT_HERSHEY_COMPLEX,
			//~ 0.5, Scalar(0,0,0), 1, LINE_8);	
	putText(image, name_text, Point(5,30), FONT_HERSHEY_COMPLEX,
			0.5, Scalar(0,0,0), 1, LINE_8);
}

