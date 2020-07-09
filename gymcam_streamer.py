# import computer vision libraries
import cv2
import numpy as np
import imutils
import pickle
# import Flask server and streamer
from flask_opencv_streamer.streamer import Streamer
from flask import Flask
from flask_cors import CORS
# import PI camera tools
from imutils.video import VideoStream
from imutils.video import FPS

# import other Python libraries
import threading
import os
import time

#import pantilt, trianglesolver and math
import pantilthat
from math import pi, sqrt, acos
from trianglesolver import solve, degree

# initialize face model    
# load our serialized face detector from disk
protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detection_model",
    "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# set preferable target to MYRIAD to run model
detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
# load our serialized face embedding model from disk
embedder = cv2.dnn.readNetFromTorch("face_embedding_model/openface_nn4.small2.v1.t7")
# set preferable target to MYRIAD to run the model
embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())   

#  initialize OpenPose Model
protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
nPoints = 15
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
# load model from disk
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
# set preferable target to MYRIAD to run the model
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# initialize video streamer
app = Flask(__name__)
cors = CORS(app, resources={r"/": {"origins": "*"}})
port = 3030
require_login = False
streamer = Streamer(port, require_login)

# initialize the video stream, then allow the camera sensor to warm up
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# stream one frame to initilize server
frame = vs.read()
streamer.update_frame(frame)
if not streamer.is_streaming:
    streamer.start_streaming()

# start the FPS throughput estimator
fps = FPS().start()

camAngle = 63.3333
camWidth = 680
camHeight = 480
pan_threshold = 100
tilt_threshold = 90
previousPanAngle = 0
previousTiltAngle = -20

pantilthat.pan(previousPanAngle)
pantilthat.tilt(previousTiltAngle)

def pantilt_camera(startX, startY, endX, endY):
    global previousPanAngle
    global previousTiltAngle
    # pan the camera to locate the face in the middle of the frame
    w = endX - startX
    newBase = (camWidth/2)-(startX+(w/2))
    absNewBase = abs(newBase)
    if (absNewBase > pan_threshold):
        a,b,c,A,B,C = solve(a=camWidth/2, A=(camAngle/2)*degree, B=90*degree)
        ll = b
        ml = sqrt((b*b)-((camWidth/2)*(camWidth/2)))
        newll = sqrt((ml*ml)+(newBase*newBase))
        #print(ml, newBase, newll)
        a,b,c,A,B,C = solve(a=ml, b=newll, c=absNewBase)
        new_pan_angle = C*(180/pi)
        print(newBase, previousPanAngle, new_pan_angle)
        if newBase > 0:
            previousPanAngle = previousPanAngle + new_pan_angle
        elif newBase < 0:
            previousPanAngle = previousPanAngle - new_pan_angle
        if (previousPanAngle > 90):
            previousPanAngle = 90
        if (previousPanAngle < -90):
            previousPanAngle = -90
        pantilthat.pan(previousPanAngle) 
    # tilt the camera to locate the face in the upper section
    if (startY < tilt_threshold/4):
        previousTiltAngle = previousTiltAngle - 10
        if (previousTiltAngle < -90):
            previousTiltAngle = -90        
        pantilthat.tilt(previousTiltAngle)        
    if (startY > tilt_threshold):
        tilt_delta = startY - 5
        tilt_percent = tilt_delta / camHeight
        new_titl_angle = (camAngle * (camAngle/100)) * tilt_percent
        print(new_titl_angle)
        previousTiltAngle = previousTiltAngle + new_titl_angle
        if (previousTiltAngle > 90):
            previousTiltAngle = 90
        pantilthat.tilt(previousTiltAngle)

def thread_face(frame):
    # construct a blob from the image
    (h, w) = frame.shape[:2]    
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    print('face - start')
    detections = detector.forward()
    print('face - finish')
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections
        if confidence * 100 > 30:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(cv2.resize(face,
                (96, 96)), 1.0 / 255, (96, 96), (0, 0, 0),
                swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            print('face embedd - start')
            vec = embedder.forward()
            print('face embedd - start')            

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            
            print(name)
            print(startX,startY,endX,endY)
            if (name == 'jad'): 
                pantilt_camera(startX,startY,endX,endY)


def thread_pose(frame):
    # add OpenPose 
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)    
    net.setInput(inpBlob)
    print('pose - start')    
    output = net.forward()
    print('pose - finish')    
    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > 0.1 : 
            cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=680)
    frame = imutils.rotate(frame, angle = 180)

    # run the face detection thread and pan/tilt the camera
    face_thread = threading.Thread(target = thread_face(frame= frame))
    face_thread.start()
    # run the pose estimation model thread
    pose_thread = threading.Thread(target=thread_pose(frame= frame))
    pose_thread.daemon = True
    pose_thread.start()


    # update the FPS counter
    fps.update()

    # show the output frame
    # cv2.imshow("Frame", frame)
    streamer.update_frame(frame)
    if not streamer.is_streaming:
        streamer.start_streaming()
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()