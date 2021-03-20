import cv2
from cam_test import Predictor

from PIL import Image


vid = cv2.VideoCapture(2)
vid.set(cv2.CAP_PROP_FPS, 10)
while(True):
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read()
    # Display the resulting frame 
     

    resized = cv2.resize(frame, (200,200), interpolation = cv2.INTER_LINEAR)
    resized =  resized[:,:,::-1]

    im = Image.fromarray(resized)
    out = Predictor(im)

    cv2.imshow('frame', out)
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(100) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 