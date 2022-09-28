import cv2
#import face_recognition
import dlib
import numpy as np

#TODO Create triangles on another image

#Gets box coordinates of face
def getFace():
   
    #live video
    #cap= cv2.VideoCapture("face.mp4")


    image1 = cv2.imread("me.png")
    image2 = cv2.imread("nithin.png")




    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
     
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

 
    face_locations = []

    frontalFaceDetector = dlib.get_frontal_face_detector()

    count = 0
    while True:

        #ret, frame = cap.read() #frame 

        print("FRAME")
        print(frame)
        print("RET")
        print(ret)
      
        #rgb_frame = frame[:, :, ::-1]


        #img= cv2.imread(rgb_frame)
        #cv2.imshow('frame',rgb_frame)
        #cv2.waitKey(1000)
        #gray=cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        
        #live video
        #rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        #gray = rgb_frame


        rgb_frame = image1

        


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

        points = []


            # jaw, ear to ear
        points += overlay(0, 17, landmarks, rgb_frame)
     
        # left eyebrow
        points += overlay(17, 22, landmarks, rgb_frame)
     
        # right eyebrow
        points += overlay(22, 27, landmarks, rgb_frame)
     
        # line on top of nose
        points += overlay(27, 31, landmarks, rgb_frame)
     
        # bottom part of the nose
        points += overlay(31, 36, landmarks, rgb_frame)
     
        # left eye
        points += overlay(36, 42, landmarks, rgb_frame)
     
        # right eye
        points += overlay(42, 48, landmarks, rgb_frame)
     
        # outer part of the lips
        points += overlay(48, 60, landmarks, rgb_frame)
     
        # inner part of the lips
        points += overlay(60, 68, landmarks, rgb_frame)
            
        
        #cv2.imshow("landmarks", rgb_frame)

        #cv2.waitKey(100)

        print("POINTS")

        print(points)



        image = rgb_frame
        size = image.shape
        rect = (0, 0, size[1], size[0])
        subdiv = cv2.Subdiv2D(rect);

        triangle(rect, points, subdiv)
        count += 1
        drawTriangles(rgb_frame, subdiv, count)





def overlay(startpoint, endpoint, landmarks, gray):
    points = []
  
    for n in range(startpoint,endpoint):
   
        x=landmarks.part(n).x
        y=landmarks.part(n).y
        points.append([x,y])
     
        cv2.circle(gray, (x, y), 4, (0, 0, 255), -1)

        # Green color in BGR
        color = (0, 255, 0)
             
        # Line thickness of 9 px
        thickness = 2
            
        end_point = (x,y)

        if(n != startpoint):
            gray = cv2.line(gray, start_point, end_point, color, thickness)
        start_point = (x, y)


    landmarkPoints = np.array(points, np.int32)

    return points

   

def triangle(image, fiducials, subdiv):
    
    for points in fiducials:
        subdiv.insert((points[0], points[1])) #points add to subdiv

    
    #3 pairs (3 points)
    triangles = subdiv.getTriangleList() #from fiducial points get allall the traingels

    triangle = []
    allpoints = []





    for p in triangles:
    


        pt1 = [p[0], p[1]]
        pt2 = [p[2], p[3]]
        pt3 = [p[4], p[5]]

        allpoints.append(pt1)
        allpoints.append(pt2)
        allpoints.append(pt3)

        compute(pt1, pt2, pt3)

        if circumcircle(image, pt1) and circumcircle(image, pt2) and circumcircle(image, pt3):
            
            ind = []
            for j in range(0, 3):
                for k in range(0, len(fiducials)):
                    if abs(allpoints[j][0] - fiducials[k][0]) < 1.0 and abs(allpoints[j][1] - fiducials[k][1]) < 1.0:
                        ind.append(k)
            if len(ind) == 3:
                triangle.append((ind[0], ind[1], ind[2]))

        print("TRIANGLES")
        print(triangle)

     
    

    return triangle

    
#check the circumcircle of the three point triangle if it has  another point
def circumcircle(rect, point): 
    if point[0] < rect[0]:
        return False
    elif point[1] <rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def compute(pt1, pt2, pt3, x, y):

    #destination
    B = [[pt1[0], pt2[0], pt3[0]], [pt1[1], pt2[1], pt3[1]], [1, 1, 1]]
    point = [x, y]





def drawTriangles(img, subdiv, count):

    triangles = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])


    for t in triangles:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if circumcircle(r, pt1) and circumcircle(r, pt2) and circumcircle(r, pt3):
            print("contains")
            #print("point1", pt1)
            cv2.line(img, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, (255, 255, 255), 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, (255, 255, 255), 1, cv2.LINE_AA, 0)

    print("Triangles")
    #cv2.imshow("traingles",img)
    cv2.imwrite("face" + str(count) + ".png", img)
    cv2.waitKey(2000)






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
