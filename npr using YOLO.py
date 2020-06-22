import cv2
import numpy as np
import pytesseract
# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

# ADD classes to classes ds
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

#Layer Names
layer_names = net.getLayerNames()
#output layer
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#Giving each class different colors
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#Load image
img = cv2.imread("41.jpg")
#img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape


'''
Detecting the objects
'''
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

#pass the bob to network & out will get the detection
net.setInput(blob)
outs = net.forward(output_layers)

def ocr_text(x,y,w,h):
    image= img[y:y+h ,x:x+w]
    cv2.imshow('image',image)
    cv2.waitKey(0)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayscale',gray)
    cv2.waitKey(0)

    noise = cv2.medianBlur(gray,3)
    cv2.imshow('Median Blur',noise)
    cv2.waitKey(0)

    gray = cv2.bilateralFilter(gray,11,17,17)
    cv2.imshow('Bilateral filter',gray)
    cv2.waitKey(0)

    edged = cv2.Canny(gray,170,200)
    cv2.imshow('Canny Edges',edged)
    cv2.waitKey(0)

    cnts , new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img1= image.copy()
    cv2.drawContours(img1,cnts,-1,(0,255,0),3)
    cv2.imshow("6 --- All Contour",img1)
    cv2.waitKey(0)

    cnts = sorted(cnts, key=cv2.contourArea,reverse=True)[:40]

    img2 = image.copy()
    cv2.drawContours(img2,cnts,-1,(0,255,0),3)
    cv2.imshow("7 ---  Top 30 Contour",img2)
    cv2.waitKey(0)

    NumberPlateCnt = None # initially its None
    for c in cnts :
        peri = cv2.arcLength(c,True)  # Calculating Perimeter of each contour
        approx = cv2.approxPolyDP(c,0.02*peri , True) # How many edges are there for each contour

        if(len(approx)) == 4 :        # Select the Contour with 4 corners
            NumberPlateCnt = approx   # This is our approx. number plate contour

            # Crop these contour and store it in Cropped image folder
            x,y,w,h = cv2.boundingRect(c) # this will find the co ordinates for plate
            new_img = image[y:y+h ,x:x+w] # create new image

            break
    cv2.imshow('y',new_img)
    cv2.waitKey(0)
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    text = pytesseract.image_to_string(new_img)




    return text


#Showing info on screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.4:
            # Object detected
            #getting coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

print(class_ids)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:

        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])

        if label in ['car','bus','truck','motorbike']:
            print('label')
            color = colors[i]

            try:
                text = ocr_text(x,y,w,h)
            except:
                text= ""
            print(text)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " "+text, (x, y + 30), font, 3, color, 3)




cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
