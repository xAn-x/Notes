new_img=cv2.bitwise_and(img,img,mask=mask)
cv2.imshow("Image",new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()