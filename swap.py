import cv2
#import face_recognition
import dlib
import numpy as np
from scipy import interpolate
import pry

#TODO Create triangles on another image

#Gets box coordinates of face
def getFace():
   
    #live video
    #cap= cv2.VideoCapture("face.mp4")


    image1 = cv2.imread("jackie.png")
    image2 = cv2.imread("kobe.png")


    width = 900
    height = 900
    dim = (width, height)
     
    # resize image
    image1= cv2.resize(image1, dim, interpolation = cv2.INTER_AREA)
    image2= cv2.resize(image2, dim, interpolation = cv2.INTER_AREA)
    copy2 = image2.copy()
    copy1 = image1.copy()
  

 
    face_locations = []

    

    count = 0
    #while True:

        #ret, frame = cap.read() #frame 
      
        #rgb_frame = frame[:, :, ::-1]


        #img= cv2.imread(rgb_frame)
        #cv2.imshow('frame',rgb_frame)
        #cv2.waitKey(1000)
        #gray=cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        
        #live video
        

    rgb_frame = image1
    #rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        #gray = rgb_frame



    points, square = getLandmarks(rgb_frame)
    rgb_frame= drawSquare(rgb_frame, square, 0)

   
        

    
    #cv2.imshow("landmarks", rgb_frame)

    #cv2.waitKey(100)

    #print("POINTS")

    #print(points)



    image = rgb_frame
    size = image.shape
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect);

    indexpoints, sourceTri = triangle(rect, points, subdiv) #sourceTri is dictionary of all the triangles
    count += 1
    image1, newsource = drawTriangles(image, sourceTri)


    dest = image2
    #dest = cv2.cvtColor(dest, cv2.COLOR_BGR2RGB)

    points2, square2 = getLandmarks(dest)
    image2 = drawSquare(dest, square2, 1)
    



    destTriangles, dictTri = triangleDestination(image2, indexpoints, points2, square2) #dictTri is the destination of all the triangles in the other image
  




    #start masking
    allMatrices =  makebMat(newsource)
    #correspondance(image, allMatrices, square, dictTri, image2, sourceTri)


    cv2.imshow("image2", image2)
    img1, warpedimg = swap(copy1, allMatrices, copy2, sourceTri, dictTri, square2)
    cv2.imwrite("img.png", img1)
    cv2.imwrite("warpedimg.png", warpedimg)
    #swap(image2, allMatrices, image, sourceTri, destTriangles, all_abg, img1bound):




def getLandmarks(rgb_frame):

    frontalFaceDetector = dlib.get_frontal_face_detector()
    allFaces = frontalFaceDetector(rgb_frame, 0)

    for face in allFaces:
      x1=face.left()
      y1=face.top()
      x2=face.right()
      y2=face.bottom()

    square = [x1, y1, x2, y2]
    print("SQUARE w")
    print(square)


    # Drawing a rectangle around the face


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

    return points, square




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

        #if(n != startpoint):
            #gray = cv2.line(gray, start_point, end_point, color, thickness)
        #start_point = (x, y)




    landmarkPoints = np.array(points, np.int32)

    return points

   


def drawSquare(image, points, zero):
    
    if(zero > 0):
        #image = cv2.rectangle(image, (points[0], points[1]), (points[2]+20, points[3]+30), (255, 0, 0), 2)
        print(points[0], points[1]), (points[2], points[3])
        for i in range(len(image)):
            for j in range(len(image)):
                if(j >= points[0] and i >=points[1] and j <= points[2]and i<=points[3]):
                    k = 2

                else:
                    image[i][j] = (0, 0, 0)
    else:
        
        image = cv2.rectangle(image, (points[0], points[1]), (points[2], points[3]), (255, 0, 0), 2)
    return image

def triangle(image, fiducials, subdiv):
    
    for points in fiducials:
        subdiv.insert((points[0], points[1])) #points add to subdiv

    
    #3 pairs (3 points)
    triangles = subdiv.getTriangleList() #from fiducial points get allall the traingels

    triangle = []
    allpoints = []


    sourceTri = {}






    count = 1
    for p in triangles:
    


        pt1 = [p[0], p[1]]
        pt2 = [p[2], p[3]]
        pt3 = [p[4], p[5]]

        allpoints.append(pt1)
        allpoints.append(pt2)
        allpoints.append(pt3)


        #compute(pt1, pt2, pt3)

        if circumcircle(image, pt1) and circumcircle(image, pt2) and circumcircle(image, pt3):
            
            ind = []
            for j in range(0, 3):
                for k in range(0, len(fiducials)):
                    if abs(allpoints[j][0] - fiducials[k][0]) < 1.0 and abs(allpoints[j][1] - fiducials[k][1]) < 1.0:
                        ind.append(k)
            tri = [pt1, pt2, pt3]
            sourceTri[count] = tri
            count += 1
            if len(ind) == 3:
                tri = [ind[0], ind[1], ind[2]]
                triangle.append((ind[0], ind[1], ind[2]))
                



    

    indexTriangle = []
    for t in triangles:
        index1 = findIndexpoint(fiducials, t[0], t[1])
        index2 = findIndexpoint(fiducials, t[2], t[3])
        index3 = findIndexpoint(fiducials, t[4], t[5])
        indexTriangle.append([index1, index2, index3])
      


    return indexTriangle, sourceTri



def findIndexpoint(points, x,y):
    #print("POINTS")
    #print(points)
    for p in range(len(points)):
        if (x == points[p][0] and y == points[p][1]):
            return p


def triangleDestination(img, indexTriangle, points2, square):
    size = img.shape
    r = (0, 0, size[1], size[0])
    
    dictTri = {}

    count = 1
    for t in indexTriangle:
        destTriangles = []
        pt1 = (points2[t[0]][0], points2[t[0]][1]) #indexes 
        pt2 = (points2[t[1]][0], points2[t[1]][1])
        pt3 = (points2[t[2]][0], points2[t[2]][1])

        destTriangles = [pt1, pt2, pt3]

        

        dictTri[count] = destTriangles
        count += 1
        
      


        #if circumcircle(r, pt1) and circumcircle(r, pt2) and circumcircle(r, pt3):
       
            #print("point1", pt1)
            #cv2.line(img, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA, 0)
            #cv2.line(img, pt2, pt3, (255, 255, 255), 1, cv2.LINE_AA, 0)
            #cv2.line(img, pt3, pt1, (255, 255, 255), 1, cv2.LINE_AA, 0)

   
    #cv2.imshow("traingles",img)
    #cv2.imwrite("nitinFace" + ".png", img)
    #cv2.waitKey(2000)

    return destTriangles, dictTri





    
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







def drawTriangles(img, triangles):



    sourceTri = {}
    size = img.shape
    r = (0, 0, size[1], size[0])


    count = 1
    s = 1
    for i in range(1,len(triangles)):
        t = triangles[i]
        pt1 = t[0]
        pt2 = t[1]
        pt3 = t[2]

     

        
        if circumcircle(r, pt1) and circumcircle(r, pt2) and circumcircle(r, pt3):
       
            #print("point1", pt1)


            pt1 = [int(np.float32(pt1[0])), int(np.float32(pt1[1]))]
            pt2 = [int(np.float32(pt2[0])), int(np.float32(pt2[1]))]
            pt3 = [int(np.float32(pt3[0])), int(np.float32(pt3[1]))]





            sourceTri[count] = [pt1, pt2, pt3]

        count += 1
        s += 1




   
    #cv2.imshow("traingles",img)
    cv2.imwrite("face" + str(s) + ".png", img)


    return img, sourceTri


def InTriangle(allMatrices, img, sourceTri):
    #print("POINT CHECKING ", points)
    count = 1

    all_abg = {}
    all_TrianglePts = {}
    print("length of matrices ", len(allMatrices))
    for i in range(len(allMatrices)):
        matrix = allMatrices[count]

        print("MATRIX ", i+1)
        print(matrix)

        x = [matrix[0][0], matrix[0][1], matrix[0][2]]
        print("X")
        print(x)

        y = [matrix[1][0], matrix[1][1], matrix[1][2]]
        print("Y")
        print(y)
        topleftx = np.min(x)
        toplefty = np.min(y)


        botrightx = np.max(x)
        botlefty = np.max(y)



        xx, yy = np.meshgrid(range(int(topleftx), int(botrightx)), range(int(toplefty), int(botlefty)))
        xx = xx.flatten()
        yy = yy.flatten()
        ones = np.ones(xx.shape, dtype = int )



        boundingbox = [topleftx, toplefty, botrightx, botlefty]

        #image = cv2.rectangle(img, (int(boundingbox[0]), int(boundingbox[1])), int((boundingbox[2]), int(boundingbox[3])), (255, 0, 0), 2)
        pt1 = int(np.float32(boundingbox[0]))
        pt2 = int(np.float32(boundingbox[1]))
        pt3 = int(np.float32(boundingbox[2]))
        pt4 = int(np.float32(boundingbox[3]))





        #image = cv2.rectangle(img, (pt3, pt4), (pt1, pt2), (255, 0, 0), 2)
       
        cv2.imwrite("imagebox.png", img)
        boundingboxpts = np.vstack((xx, yy, ones))
        alpha, beta, gamma = np.dot(np.linalg.pinv(matrix),  boundingboxpts)
        valid_alpha = np.where(np.logical_and(alpha > -0.1, alpha <1.1))[0]
        valid_beta = np.where(np.logical_and(beta > -0.1, beta <1.1))[0]
        valid_gamma = np.where(np.logical_and(alpha + beta + gamma > -0.1, alpha + beta + gamma < 1.1))[0]


        valid_al_beta = np.intersect1d(valid_alpha, valid_beta)
        inside_pts_loc = np.intersect1d(valid_al_beta, valid_gamma)
        boundingboxpts = boundingboxpts.T
        pts_in_triangle = boundingboxpts[inside_pts_loc]
        all_alpha = alpha[inside_pts_loc]
        all_beta = beta[inside_pts_loc]
        all_gamma = gamma[inside_pts_loc]


        #print("ALPHA ", all_alpha)
        #print("BETA ", all_beta)
        #print("GAMMA ", gamma)
        

        triangle = [all_alpha, all_beta, all_gamma]
        all_TrianglePts[count] = pts_in_triangle

        all_abg[count] = triangle
        count += 1


    return all_abg, all_TrianglePts

    






def makebMat(trianglepoints):
    allMatrices = {}
    count = 1
    for corners in trianglepoints:

        #print("corners " + str(count),  corners)
        corners = trianglepoints[count]
      
        Bmat = np.array([[corners[0][0], corners[1][0], corners[2][0]], [corners[0][1], corners[1][1], corners[2][1]], [1, 1, 1]])
        allMatrices[count] = Bmat
        count += 1
    return allMatrices







#goes through image points and checks what each point is 
'''
Hashmap : points -> alpha, beta, gama
'''
def correspondance(img1, allMatrices, square, destTriangles, img2, sourceTri):

   

    count = 1

    print("Square x lower ", square[0])
    print("Square x upper", square[2])

    print("Square y lower", square[1])
    print("Square y upper", square[3])

    all_abg = InTriangle(allMatrices, img1, sourceTri)
    #print("ALL ABG")
    #print(all_abg)


    finalpoints = {}
    for i in range(1, len(destTriangles)):
        matrix = all_abg[i]
        destTriangle = destTriangles[i]
        aMat = makebMat(destTriangles)
        points = PixelPosition(aMat[i], matrix)
        finalpoints[i] = points #







def swap(image2, allMatrices, image, sourceTri, destTriangles, square):
    
    img = image2.copy()
    warpedImgS = np.zeros((img.shape), np.uint8)
    xS, yS, wS, hS = square[0]-300, square[1]-100, square[2]+200, square[3]+900
    all_abg, all_TrianglePts = InTriangle(allMatrices, image, sourceTri)
  

    for i in range(1, len(sourceTri)):
        pt1source = sourceTri[i][0]
        pt2source = sourceTri[i][1]
        pt3source = sourceTri[i][2]


        pt1dest = sourceTri[i][0]
        pt2dest = sourceTri[i][1]
        pt3dest = sourceTri[i][2]



        abg = all_abg[i]
        alpha = abg[0]
        beta = abg[1]
        gamma = abg[2]

        insideTriangle = all_TrianglePts[i]
        insideTriangle = insideTriangle[:, 0:2] 
        if np.shape(insideTriangle)[0] == 0: 
             continue


        destTriangle = destTriangles[i]
        #aMat = makebMat(destTriangle)

        aMat = np.array([[destTriangle[0][0], destTriangle[1][0], destTriangle[2][0]], [destTriangle[0][1], destTriangle[1][1], destTriangle[2][1]], [1, 1, 1]])

        aMatCoord = np.vstack((alpha, beta))
        #pry()
        aMatCoord = np.vstack((aMatCoord, gamma)) 
        warped_ptS = np.dot(aMat, aMatCoord) 


        warped_ptS = warped_ptS.T 
        warped_ptS[:, 0] = warped_ptS[:, 0]/warped_ptS[:, 2] 
        warped_ptS[:, 1] = warped_ptS[:, 1]/warped_ptS[:, 2] 
        warped_ptS = warped_ptS[:, 0:2] 
  
        width = range(0, image.shape[1]) 
        height = range(0, image.shape[0]) 
        

        # print(image1.shape) 
        # https://scipython.com/book/chapter-8-scipy/examples/scipyinterpolateinterp2d/ 
        interp1 = interpolate.interp2d(width, height, image[:, :, 0], kind='linear') 
        #print(x) 
        interp2 = interpolate.interp2d(width, height, image[:, :, 1], kind='linear') 
        #print(len(y)) 
        interp3 = interpolate.interp2d(width, height, image[:, :, 2], kind='linear') 
        # print(pt1d, pt2d, pt3d) 

        for pts, x, y in zip(insideTriangle, warped_ptS[:, 0], warped_ptS[:, 1]): 
             x -= xS 
             y -= yS 
             edge1 = interp1(x, y) 
             edge2 = interp2(x, y) 
             edge3 = interp3(x, y) 
  
             img[pts[1], pts[0]] = (edge1, edge2, edge3) 
             warpedImgS[pts[1], pts[0]] = (edge1, edge2, edge3) 

    return img, warpedImgS


        





def PixelPosition(Amat, matrix):

    points = {}


    for i in range(len(matrix)):
        alpha = matrix[0][i]
        beta = matrix[1][i]
        gamma = matrix[2][i]

        v = np.vstack((alpha, beta, gamma))
        x, y, z = np.dot(Amat, v)
    



        x = int(x/z)
        y = int(y/z)
        points[i] = [x,y] #dictionary of each coordinate in the triangle




        #print("x ", x) 
        #print("y ", y)




    return points








    '''
    for x in range(square[0], square[2]):
        for y in range(square[1], square[3]):
            if(InTriangle([x,y], allMatrices, img) != None):
                
            else:
                #print("NONE for count ", count)
                count += 1
             #print("POINT ", str(x), str(y))

             '''





             

        






    














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
