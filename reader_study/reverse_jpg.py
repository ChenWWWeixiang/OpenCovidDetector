import cv2,os
for item in os.listdir('sig_img'):
    I=cv2.imread(os.path.join('sig_img',item))
    cv2.imwrite('sig_img/'+item,I[1,:,:])