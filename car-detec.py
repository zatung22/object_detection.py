import numpy as np
import cv2 
from random import randint
import datetime

# ฟังก์ชั่นสุ่มสี
def getRandomColors(n):
    colors = []
    for i in range(n):
        colors.append((randint(0,n-1)*(255//n),randint(0,n-1)*(255//n),randint(0,n-1)*(255//n)))
    return colors

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
''
classColors = getRandomColors(20)
print(classColors)

cap = cv2.VideoCapture("C:/car/LosAngeles.mp4")

#Load the Caffe model 
# net = cv2.dnn.readNetFromCaffe("C:/car/vgg_ssd.prototxt", "C:/car/vgg_ssd.caffemodel")
net = cv2.dnn.readNetFromCaffe("C:/car/vgg_ssd.prototxt", "C:/car/vgg_ssd.caffemodel")


net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Open a file to save the report
file = open('detection_report.txt', 'w')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # MobileNet requires fixed dimensions for input image(s)
    #blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 117, 123), False)

    #Set to network the input blob 
    net.setInput(blob)

    #Prediction of network
    detections = net.forward()

    #Size of frame resize (300x300)
    # Size of frame
    height = frame.shape[0]  
    width = frame.shape[1] 

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #Confidence of prediction 
        if confidence > 0.3: # Filter prediction 
            class_id = int(detections[0, 0, i, 1]) # Class label
            
            # Scale detection frame
            xLeftBottom = int(width * detections[0, 0, i, 3]) 
            yLeftBottom = int(height * detections[0, 0, i, 4])
            xRightTop   = int(width * detections[0, 0, i, 5])
            yRightTop   = int(height * detections[0, 0, i, 6])

            # Draw location
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),classColors[class_id])

            # Draw label and confidence
            label = classNames[class_id] + ": " + str(confidence)

            # Get current time
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Add the time to the label
            label_with_time = current_time + " - " + label

            labelSize, baseLine = cv2.getTextSize(label_with_time, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            yLeftBottom = max(yLeftBottom, labelSize[1])
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                    (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                    (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label_with_time, (xLeftBottom, yLeftBottom),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            # Write the output to the file
            file.write(f"{current_time} - Class: {label}\n")

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:  # Break with ESC 
        break

class_counts = {key: 0 for key in classNames.keys()}  # Initialize counts for each class to zero

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # MobileNet requires fixed dimensions for input image(s)
    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 117, 123), False)

    # Set input blob to network
    net.setInput(blob)

    # Prediction
    detections = net.forward()

    # Size of frame
    height = frame.shape[0]  
    width = frame.shape[1] 

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Confidence of prediction
        if confidence > 0.3:  # Filter prediction
            class_id = int(detections[0, 0, i, 1])  # Class label
            
            # Increment count for detected class
            class_counts[class_id] += 1

            # Scale detection frame
            xLeftBottom = int(width * detections[0, 0, i, 3]) 
            yLeftBottom = int(height * detections[0, 0, i, 4])
            xRightTop = int(width * detections[0, 0, i, 5])
            yRightTop = int(height * detections[0, 0, i, 6])

            # Draw location
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), classColors[class_id])

            # Draw label and confidence
            label = classNames[class_id] + ": " + str(confidence)
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            label_with_time = current_time + " - " + label
            labelSize, baseLine = cv2.getTextSize(label_with_time, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            yLeftBottom = max(yLeftBottom, labelSize[1])
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                    (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                    (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label_with_time, (xLeftBottom, yLeftBottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            # Write the output to the file
            file.write(f"{current_time} - Class: {label}\n")

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:  # Break with ESC
        break

# Print the counts of each class
for class_id, count in class_counts.items():
    print(f"Class '{classNames[class_id]}': {count}")

file.close()



