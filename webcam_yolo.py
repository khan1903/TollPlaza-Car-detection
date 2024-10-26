from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
# import torch
# print("Packages Imported")
# print(torch.cuda.is_available())
# # for webCam
vid = cv2.VideoCapture(0,cv2.CAP_DSHOW)   #but it kept showing a Warning which said: `anonymous-namespace'::SourceReaderCB::~SourceReaderCB terminating async callback and the webcam wouldn't open.
                                                # I found a workaround for it in the internet which said I needed to write the code as cv2.CAP_DSHOW
                                                # it is use only in webcam
# # for Video Input
# vid = cv2.VideoCapture("../resources/desert_animated.mp4")
# fps = int(vid.get(cv2.CAP_PROP_FPS))
# print('frames per second =', fps)
# vid.set(3,1920)
# vid.set(4, 1080)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output = cv2.VideoWriter("desert_animated.mp4", fourcc, 30, (1280, 720))
### make detection model
model = YOLO("../Yolo-Weight/yolov8x.pt")
classNames = ["human", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "plant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
detection = []

def main():

    print("Main Function")
    while True:
        succes, frame = vid.read()

        # print(frame.shape)
        if not succes:
            break
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        print('frames per second =', fps)
        # frame = cv2.resize(frame, (1280, 720))
            ### detection model ###
        detection_model(frame,fps)
        # output.write(frame)
        cv2.imshow("VIDEO OUTPUT",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
def detection_model(img,fps):
    print("Detection Model Implemention")

    result = model(img, stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # cvzone.cornerRect(img, (x1, y1, w, h),l=10,t=2)
            #confidence
            conf = (math.ceil(box.conf[0]*100))/100

            #class name
            cls = int(box.cls[0])  #it give class id in float so convert in integer

            # cvzone.putTextRect(img,f'{classNames[cls]}  {conf}',(max(0,x1),max(35,y1)),1,1)
            #
            # print(conf)
            current_class = classNames[cls]
            # if current_class == "car" or current_class =="truck":    #and conf > 0.3:
            cvzone.putTextRect(img,f'{current_class}  {conf} ',(max(0,x1),max(35,y1)),1,1,offset=3)
            cv2.putText(img, f"fps ={fps}",(0,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),3)
            cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=2)
            currentArray = np.array([x1, y1, x2, y2, conf])

                # output = cv2.VideoWriter(
                #     "output.avi", cv2.VideoWriter_fourcc(*'MPEG'), 30, (1080, 1920))
                # print(currentArray)
                # detections = np.vstack((detections, currentArray))
                # # for val in detections:
                # #     print(detections[val])


main()


vid.release()
cv2.destroyAllWindows()