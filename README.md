# 2-View Face Recognition

##Profile Detection Using LBP-Trained Cascade

![alt tag](http://imgur.com/XAMSXAS)

![alt tag](http://imgur.com/UM35ZJw)

![alt tag](http://imgur.com/9j4A4Rx)

![alt tag](http://imgur.com/y90pIGo)

![alt tag](http://imgur.com/jB1b0wR)

__________________________________________________________________________________________________________________________

## Past Screenshots:

![alt tag](https://github.com/DeLaSalleUniversity-Manila/TwoViewFaceRecognition/blob/master/screenshots/facerec_2view.jpg)

![alt tag](https://github.com/DeLaSalleUniversity-Manila/TwoViewFaceRecognition/blob/master/screenshots/facerec_2view2.jpg)


## TODO

* Explore other two face recognition algorithms (this will add data to Chapter 5): 

[1] EigenFaceRecognizer
```cpp
Ptr<FaceRecognizer> createEigenFaceRecognizer(int num_components=0, double threshold=DBL_MAX)¶
```

[2] LBPHFaceRecognizer
```cpp
Ptr<FaceRecognizer> createLBPHFaceRecognizer(int radius=1, int neighbors=8, int grid_x=8, int grid_y=8, double threshold=DBL_MAX) 
```

*  Save the 'confidence' value of your chosen face in a variable for both frontal and sideview images:

```cpp
void recognizeProfileFaces(Mat face_scaled, Ptr<FaceRecognizer> modelProfile)
{
	
	modelProfile -> predict(face_scaled, label2, confidence2);
	
	tallyProfile(label2, confidence2);
	
	//~ cout << "LabelSide: " << label2 << endl;
	//~ cout << "Confidence: " << confidence2 << endl;
}
```

* After chosing the face, set a threshold for confidence in both frontal and sideview images based on experiments:
```cpp
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
		
		//~ cout << a_1 << " " << b_1 << " " << c_1 << " " << d_1 << " " 
		     //~ << e_1 << " " << f_1 << " " << g_1 << " " << h_1 << " " 
		     //~ << i_1 << " " << j_1 << endl;
}
```

Maybe,

```cpp
if (confidenceChoice1 > frontalThreshold && confidenceChoice2 > sideThreshold)
  // Recognition success!
```



* You'll have to perform more experiments and gather more data. (At least 30 different people!)
* Add Screenshots 
* Update me with the accuracy
