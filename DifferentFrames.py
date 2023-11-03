# tham khao trang web https://debuggercafe.com/moving-object-detection-using-frame-differencing-with-opencv/
# https://github.com/a00ayad00/Motion-Detection-using-frame-differencing-with-opencv/blob/main/Motion%20Detection.py
import cv2

# cv2.startWindowThread()
# img=cv2.imread('Input.png')
# cv2.imshow('Window',img)
# cv2.waitKey(0)

vid = r"E:\VietBao\2023\Motion-Detection-Survey\video_3.mp4"

cap = cv2.VideoCapture(vid)

frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height= cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# fourcc = cv2.VideoWriter_fourcc(*"X264")
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
path = r"E:\VietBao\2023\Motion-Detection-Survey\videotmp.MP4"
out = cv2.VideoWriter(path, fourcc, 10, (int(frame_width), int(frame_height)))

done, CurrentFrame = cap.read()
done, NextFrame = cap.read()

while cap.isOpened():
    if done==True:
        diff = cv2.absdiff(CurrentFrame, NextFrame)
        
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        blured_img = cv2.GaussianBlur(gray, (5, 5), 0)
        
        threshold, binary_img = cv2.threshold(blured_img, 35, 255, cv2.THRESH_BINARY)
        
        dilated = cv2.dilate(binary_img, None, iterations=12)
        
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)
        
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            
            if cv2.contourArea(contour) < 1000:
                continue
            
            cv2.rectangle(CurrentFrame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
        cv2.imshow("frame", CurrentFrame)
        # cv2.imshow("binary_img", cv2.flip(binary_img, 1))
        # cv2.imshow("dilated_img", dilated)
        
        out.write(CurrentFrame)
        
        CurrentFrame = NextFrame
        
        done, NextFrame = cap.read()
        
        if cv2.waitKey(30) == ord("g"):
            break
    else: break
cv2.destroyAllWindows()
cap.release()
out.release()
