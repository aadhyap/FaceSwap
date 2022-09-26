import cv2
#import face_recognition
import dlib
import numpy as np



#Gets box coordinates of face
def getFace():
    print("started")

    #live video
    cap= cv2.VideoCapture(0)
    face_locations = []

    frontalFaceDetector = dlib.get_frontal_face_detector()

    while True:

        ret, frame = cap.read() #frame 

        print("FRAME")
        print(frame)
        print("RET")
        print(ret)
        # Convert the image from BGR color (which OpenCV uses) to RGB   
        # color (which face_recognition uses)
        #rgb_frame = frame[:, :, ::-1]
        # Find all the faces in the current frame of video

        #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]





        


        # Now the dlip shape_predictor class will take model and with the help of that, it will show 
        


        #img= cv2.imread(rgb_frame)
        #cv2.imshow('frame',rgb_frame)
        #cv2.waitKey(1000)
        #gray=cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        #gray = rgb_frame

        


        allFaces = frontalFaceDetector(rgb_frame, 0)
        

        for face in allFaces:
          x1=face.left()
          y1=face.top()
          x2=face.right()
          y2=face.bottom()
        # Drawing a rectangle around the face
          #cv2.rectangle(rgb_frame, (x1,y1), (x2,y2),(0,255,0),3)
        #cv2_imshow(img)

        faceLandmarkDetector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        landmarks = faceLandmarkDetector(rgb_frame, face)



            # jaw, ear to ear
        overlay(0, 17, landmarks, rgb_frame)
     
        # left eyebrow
        overlay(17, 22, landmarks, rgb_frame)
     
        # right eyebrow
        overlay(22, 27, landmarks, rgb_frame)
     
        # line on top of nose
        overlay(27, 31, landmarks, rgb_frame)
     
        # bottom part of the nose
        overlay(31, 36, landmarks, rgb_frame)
     
        # left eye
        overlay(36, 42, landmarks, rgb_frame)
     
        # right eye
        overlay(42, 48, landmarks, rgb_frame)
     
        # outer part of the lips
        overlay(48, 60, landmarks, rgb_frame)
     
        # inner part of the lips
        overlay(60, 68, landmarks, rgb_frame)
            
        
        cv2.imshow("landmarks", rgb_frame)

        cv2.waitKey(100)





def overlay(startpoint, endpoint, landmarks, gray):
    for n in range(startpoint,endpoint):
   
        x=landmarks.part(n).x
        y=landmarks.part(n).y
        cv2.circle(gray, (x, y), 4, (0, 0, 255), -1)
        # Green color in BGR
        color = (0, 255, 0)
             
        # Line thickness of 9 px
        thickness = 2
            
        end_point = (x,y)

        if(n != startpoint):

            gray = cv2.line(gray, start_point, end_point, color, thickness)
        start_point = (x, y)


'''LOOKING AT A SPECIFIC POINT

x=landmarks.part(31).x
y=landmarks.part(31).y
# Drawing a circle
cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
cv2_imshow(img)


'''

#cv2.imshow('Video', frame)
        
# Wait for Enter key to stop
#if cv2.waitKey(25) == 13:
#break

#def drawPoints():
getFace()
