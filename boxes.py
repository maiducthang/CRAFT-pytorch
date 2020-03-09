import cv2
import numpy as np
import pytesseract
from PIL import Image
with open('results/res_dethi2.txt', 'r') as f:
    file = f.read().splitlines() 


def crop_text(image, coor):
    poly = np.array(box).astype(np.int32).reshape((-1, 1, 2))
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    coor =  np.array([coor], np.int32)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    cv2.fillPoly(mask, coor, (255))
    res = cv2.bitwise_and(image, image, mask = mask)
    rect = cv2.boundingRect(coor) # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    im = image.copy()
    im = cv2.polylines(im, coor, True, (0,255,0))
    return cropped, im

image = cv2.imread('data/dethi2.jpg')
image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# (thresh, image) = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

arr = []
for line in file:
    coor = line.split(',')
    coor = [str(i) for i in coor]
    arr.append(np.array(coor).reshape(4, 2).astype('uint32'))

for box in arr:
    crop_img, img = crop_text(image, box)
    # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(crop_img)
    print(pytesseract.image_to_string(im_pil, lang='vie'))
    cv2.imshow('img', img)
    cv2.imshow('crop image', crop_img)
    cv2.waitKey(0)

cv2.destroyAllWindows

