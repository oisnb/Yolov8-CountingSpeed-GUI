import cv2
img=cv2.imread("../directionImg/1111.jpg")
img=img[445:518,290:490]
cv2.imshow("qq",img)
cv2.waitKey(0)
cv2.destroyAllWindows()