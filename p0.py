import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

"""
preprocessing the image file
make a function
"""
# Read Image
image = cv2.imread('w.jpg')

# resize the image
#image= imutils.resize(image,width=500)

#display the image
cv2.imshow('0--- orignal image',image)
cv2.waitKey(0)

# convert it to greyscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("1 ---  Gray scale image",gray)
cv2.waitKey(0)

#Noise Removal
noise = cv2.medianBlur(gray,3)
cv2.imshow("2 --- medianBlur Blur",noise)
cv2.waitKey(0)


#bilateralFilter
gray = cv2.bilateralFilter(gray,11,17,17)
cv2.imshow("3 --- bilateralFilter", gray)
cv2.waitKey(0)


"""
Threshold
thre = cv2.threshold(gray,62, 255, cv2.THRESH_BINARY)
cv2.imshow("4 --- Threshold",thre)
cv2.waitKey(0)
TypeError: Expected Ptr<cv::UMat> for argument 'mat'
 use img2 = np.array(img) convert numpy object later
"""




#Canny Edge Detection
"""
Find edges , smooth , Gaussian blur
"""
edged = cv2.Canny(gray,170,200)
cv2.imshow("5 --- Canny Edges",edged)
cv2.waitKey(0)

# find Con based on edges
cnts , new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# first argument is source image, second is contour retrieval mode, and the third is contours approximation.
# output provides no. of contour and second is heirachy

#create copy of orignal image to drawContours
img1= image.copy()
cv2.drawContours(img1,cnts,-1,(0,255,0),3)
cv2.imshow("6 --- All Contour",img1)
cv2.waitKey(0)




"""
Sort contour based on their area keeep minimum  required area is 30
anything under this willl not be considered
"""
cnts = sorted(cnts, key=cv2.contourArea,reverse=True)[:30]

NumberPlateCnt = None # initially its None


"""
Now will draw top 30 contours on image , first we will copy image file
"""
img2 = image.copy()
cv2.drawContours(img2,cnts,-1,(0,255,0),3)
cv2.imshow("7 ---  Top 30 Contour",img2)
cv2.waitKey(0)

"""
loop over all these contours to find te best possible approximate
contour with 4 corners
"""
count =0
idx =0# to store the cropped images


for c in cnts :
    peri = cv2.arcLength(c,True)  # Calculating Perimeter of each contour
    approx = cv2.approxPolyDP(c,0.02*peri , True) # How many edges are there for each contour
    if(len(approx)) == 4 :        # Select the Contour with 4 corners
        NumberPlateCnt = approx   # This is our approx. number plate contour

        # Crop these contour and store it in Cropped image folder
        x,y,w,h = cv2.boundingRect(c) # this will find the co ordinates for plate
        new_img = image[y:y+h ,x:x+w] # create new image
        cv2.imwrite(str(idx)+".png",new_img) # store new image
        print(idx)
        idx=idx+1
        cv2.drawContours(image,[NumberPlateCnt],-1,(0,255,0),3)
        cv2.imshow("8 --- final image",image)
        cv2.waitKey(0)



# Drawing the selected con on the image

l =[]

for i in range(idx):
    cropped_image_loc = str(i)+".png"
    cv2.imshow("Cropped Image ",cv2.imread(cropped_image_loc))

    # Use tesseract to convert image to string
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    text = pytesseract.image_to_string(cropped_image_loc)
    l.append(text)

    cv2.waitKey(0)


print(l)
