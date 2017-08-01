import cv2

img = 'data/IMG/center_2017_07_28_14_04_24_471.jpg'
image = cv2.imread(img)
image_flip = cv2.flip(image,1)
cv2.imwrite('image_flip.png',image_flip)