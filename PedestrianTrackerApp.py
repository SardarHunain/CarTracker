import cv2
#image
img = "401_Gridlock.JPG"

video = cv2.VideoCapture('carVideo.mp4')

#creat opencv image
img = cv2.imread(img)

#convert to grey scale
blackNwhite = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#pre trained classifier
classifierFile = "cars.xml"
pedestrianFile = 'pedestrian.xml'
#create car classifier
car_tracker = cv2.CascadeClassifier(classifierFile)
pedestrianTracker = cv2.CascadeClassifier(pedestrianFile)
#run forever until car stops
while True:
    #read the current frame
    (readSuccessful, frame) = video.read()

    if readSuccessful:
        #must convert to grey
        greyscaledFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect cars
    cars = car_tracker.detectMultiScale(greyscaledFrame)
    pedestrians = pedestrianTracker.detectMultiScale(greyscaledFrame)
    # draw rectangles around cars
    for (x, y, w, z) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + z), (0, 255, 255), 2)

    for (x, y, w, z) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + z), (0, 255, 255), 2)


    cv2.imshow('Car Detector',frame)
    cv2.waitKey(1)


#Detect Cars
cars = car_tracker.detectMultiScale(blackNwhite)

#draw rectangles around cars
for(x,y,w,z) in cars:
    cv2.rectangle(img,(x,y),(x+w,y+z),(0,0,255),2)

#display image with faces spotted
cv2.imshow("clever programmer car detector",img)


#dont autoclose
cv2.waitKey()

