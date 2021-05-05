import cv2 as cv 
import numpy as np
import dlib


def rect_to_bb(rect):
    '''
    fuction which converts rectangle predicted by dlib to bounding box which can used in opencv
    '''
    x = rect.left()
    y = rect.top()
    h = rect.bottom()-y
    w = rect.right()-x

    return (x,y,w,h)

def shape_to_np(shape):
    '''
    function which converts the facial coordinates to numpy array
    '''
    coords = np.zeros((68,2),dtype = "int")
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def smile(shape):
    left = shape[48]
    right = shape[54]

    # avg of points in the middle of the mouth 
    mid = (shape[51]+shape[62]+shape[66]+shape[57])/4
   
    #perpendicular dist b/w mid and line joing the left and right 
    dist = np.abs(np.cross(right-left,left-mid))/np.linalg.norm(right-left)
    return dist


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor.dat")
cam = cv.VideoCapture(0)

counter = 0
selfie_no = 0
smile_threshold = 4

while(cam.isOpened()):
    ret, frame = cam.read()
    frame = cv.flip(frame,1)
    frame = cv.resize(frame,None,fx=1,fy=1,interpolation=cv.INTER_AREA)

    #detect faces 
    rects = detector(frame,1)
    for r in rects:
        (x,y,w,h) = rect_to_bb(r)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        shape = predictor(frame,r)
        shape = shape_to_np(shape)
        # getting points of only mouth (i.e 49 to 68)
        mouth = shape[48: ]
        for m in mouth:
            cv.circle(frame,tuple(m),1,(255,255,255),-1)
        #findind smile params
        smile_param = smile(shape)
        cv.putText(frame,'SP : '+str(round(smile_param,2)),(300,30), 
            cv.FONT_HERSHEY_SIMPLEX, 0.7 , (0,0,255), 2)
        
        #detecting smiles
        if smile_param > smile_threshold:
            cv.putText(frame,"Smile Detected",(300,60), 
                cv.FONT_HERSHEY_SIMPLEX, 0.7 , (0,0,255), 2)
            
            counter +=1
            #take selfie if 15 continuous frames detect smiles 
            if counter >= 15:
                selfie_no +=1
                filename = "selfie_no_"+str(selfie_no)+".png"
                ret, image =  cam.read()
                cv.imwrite(filename,image)
                print("Selfie " +  str(selfie_no)   +" is taken ...")
                counter = 0
        else :
            counter = 0       

    
    cv.imshow("frame",frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv.destroyAllWindows()