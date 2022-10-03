import cv2
import numpy as np
import math
import dlib


def main():
    print('Thin Plate Spline Initialization...')

    video_path = './Data/Data2.mp4'

    video = cv2.VideoCapture(video_path)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('./Data/Data2Output.mp4',
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             10, size)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    print('Being Swapping')

    old_box = None
    initiation = True

    while True:
        ret, frame = video.read()
        if not ret:
            print("FaceSwap finalize..")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        box = detector(frame_gray, 1)

        square_box = Box(box)

        if square_box is None or len(square_box) < 2:
            if not initiation:
                square_box = old_box

        old_box = square_box

        box1, box2 = square_box[0], square_box[1]

        WarpedFace, cara1, cara2 = FaceSwap(frame, box1, box2, predictor)
        initiation = False

        print('Video generation...')
        cv2.imwrite('./Results/face1.jpg', cara1)
        #cv2.waitKey()
        cv2.imwrite('./Results/face2.jpg', cara2)
        #cv2.waitKey()
        cv2.imshow('Swap - TPS', WarpedFace)
        result.write(np.uint8(WarpedFace))
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break


def faceDetection(facegray, box, predictor):

    pts = np.empty((68, 2), dtype=int)

    face_marks = predictor(facegray, box)

    for i in range(pts.shape[0]):
        pts[i] = (face_marks.part(i).x, face_marks.part(i).y)

    face_marks = pts
    bounder_box = cv2.boundingRect(np.float32([face_marks]))
    (x, y, w, h) = bounder_box
    shift_markers = face_marks - (x, y)
    return face_marks, bounder_box, shift_markers

def tps(source_pts, dest_pts):

    source_pts = np.array(source_pts)
    dest_pts = np.array(dest_pts)
    p = dest_pts.shape[0]
    K = np.zeros((p, p))
    P = np.zeros((p, 3))

    for i in range(p):
        for j in range(p):
            r = np.linalg.norm(dest_pts[i] - dest_pts[j])
            if r == 0:
                U = 0
            else:
                U = (r**2) * math.log(r**2)

            K[i,j] = U

        P[i, 0] = dest_pts[i, 0]
        P[i, 1] = dest_pts[i, 1]
        P[i, 2] = 1

    Pt = P.T
    Zero_mat = np.zeros([3, 3])

    L = np.vstack([np.hstack([K, P]), np.hstack([Pt, Zero_mat])])

    lam = 1e-5
    L = L + np.eye(p + 3) * lam
    L_inv = np.linalg.inv(L)

    x_set = np.zeros((source_pts.shape[0] + 3, 1))
    y_set = np.zeros((source_pts.shape[0] + 3, 1))

    x_set[0:source_pts.shape[0], :] = source_pts[:, 0].reshape(-1, 1)
    y_set[0:source_pts.shape[0], :] = source_pts[:, 1].reshape(-1, 1)

    w_x = np.matmul(L_inv, x_set)
    w_y = np.matmul(L_inv, y_set)

    weights = np.hstack((w_x, w_y))
    #print('wieghts TPS', weights)

    return weights

def warpfaces(face1, face2, weights, face_markers_2):
    
    w_x = weights[:, 0].reshape(-1, 1)
    w_y = weights[:, 1].reshape(-1, 1)

    Xi, Yi = np.indices((face2.shape[1], face2.shape[0]))
    warped_points = np.stack((Xi.ravel(), Yi.ravel(), np.ones(Xi.size))).T

    axx = w_x[w_x.shape[0] - 3]
    ayx = w_x[w_x.shape[0] - 2]
    a1x = w_x[w_x.shape[0] - 1]

    axy = w_y[w_y.shape[0] - 3]
    ayy = w_y[w_y.shape[0] - 2]
    a1y = w_y[w_y.shape[0] - 1]

    A = np.array([[axx, axy], [ayx, ayy], [a1x, a1y]]).reshape(3,2)
    actual_points = np.dot(warped_points, A)

    warped_points_x = warped_points[:,0].reshape(-1,1)
    face_markers_2_x = face_markers_2[:,0].reshape(-1,1)
    ax, bx = np.meshgrid(face_markers_2_x, warped_points_x)
    t1 = np.square(ax - bx)

    warped_points_y = warped_points[:, 1].reshape(-1,1)
    face_markers_2_y = face_markers_2[:, 1].reshape(-1,1)
    ay, by = np.meshgrid(face_markers_2_y, warped_points_y)
    t2 = np.square(ay - by)

    R = np.sqrt(t1 + t2)

    # U = np.square(R) * np.log(np.square(R))
    U = np.square(R) * np.log(R)
    U[R == 0] = 0

    MX = w_x[0:68, 0].T
    Ux = MX * U
    Ux_sum = np.sum(Ux, axis = 1).reshape(-1,1)

    MY = w_y[0:68, 0].T
    Uy = MY * U
    Uy_sum = np.sum(Uy, axis = 1).reshape(-1,1)

    actual_points = actual_points + np.hstack((Ux_sum, Uy_sum))

    X = actual_points[:, 0].astype(int)
    Y = actual_points[:, 1].astype(int)
    X[X >= face1.shape[1]] = face1.shape[1] - 1
    Y[Y >= face1.shape[0]] = face1.shape[0] - 1
    X[X < 0] = 0
    Y[Y < 0] = 0

    warped_face = np.zeros(face2.shape)
    warped_face[Yi.ravel(), Xi.ravel()] = face1[Y, X]

    return np.uint8(warped_face)



def FaceSwap(frame, box1, box2, predictor):
    WarpedFace_tmp, _,_ = Swap(frame, frame, box1, box2, predictor)
    WarpedFace, F1,F2 = Swap(frame, WarpedFace_tmp,box2, box1, predictor)
    return WarpedFace, F1, F2

def Swap(Face1,Face2, box1,box2, predictor):

    facemarks1, BoundingBox1, shifted_FaceMarks1 = faceDetection(Face1, box1, predictor)
    facemarks2, BoundingBox2, shifted_FaceMarks2 = faceDetection(Face2, box2, predictor)
    (x,y,w,h) = BoundingBox1
    crop_face1  = Face1[y:y+h,x:x+w]
    (x,y,w,h) = BoundingBox2
    crop_face2  = Face2[y:y+h,x:x+w]

    M  = tps(shifted_FaceMarks1, shifted_FaceMarks2)
    warped_face = warpfaces(crop_face1, crop_face2, M, shifted_FaceMarks2)

    mask_warped_face,_ = Mask(warped_face, shifted_FaceMarks2)
    mask_warped_face = np.int32(mask_warped_face/mask_warped_face.max())
    warped_face = warped_face * mask_warped_face

    WarpedFace = np.zeros_like(Face1)
    x,y,w,h = BoundingBox2
    WarpedFace[y:y+h, x:x+w] = WarpedFace[y:y+h, x:x+w] + warped_face

    mask, box = Mask(Face2, facemarks2)
    x,y,w,h = box
    cx, cy = (2*x+w) //2, (2*y+ h) //2
    WarpedFace = cv2.seamlessClone(np.uint16(WarpedFace), Face2, mask, tuple([cx,cy]), cv2.NORMAL_CLONE)


    Face1_print = drawMarkers(Face1, facemarks1, BoundingBox1)
    Face2_print = drawMarkers(Face2, facemarks2, BoundingBox2)

    return WarpedFace, Face1_print, Face2_print

def drawMarkers(Face1, facemarks, box):

    Face = Face1.copy()
    #(x,y,w,h) = box
    # cv2.rectangle(Face,(x,y),(x+w,y+h),(0,255,0),2)

    for (a,b) in facemarks:
        cv2.circle(Face,(a,b),2,(255,0,0),-1)
    return Face

def Mask(Face2,facemarks2):
    mask = np.zeros_like(Face2)
    hull = cv2.convexHull(np.array(facemarks2))
    mask = cv2.fillConvexPoly(mask,hull, (255, 255, 255))
    box = cv2.boundingRect(np.float32([hull.squeeze()]))
    return mask, box

def Box(boxes):

    AreaList = []
    squarebox = []

    for i, box in enumerate(boxes):

        x = box.left()
        y = box.top()
        w = box.right() - x
        h = box.bottom() - y

        area = w*h

        if area < 500:
            area = -1
        AreaList.append(area)

    inds = np.argsort(AreaList)[::-1]

    for i in inds[:2]:
        if AreaList[i] > 500:
            squarebox.append(boxes[i])
    return squarebox

if __name__ == '__main__':
    main()
