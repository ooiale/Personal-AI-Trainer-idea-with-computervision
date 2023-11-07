import cv2
import time
import numpy as np
import poseModule as pm

#12 14 16
#11 13 15



def main():
    cap = cv2.VideoCapture('videos/pushups.mp4')
    pTime = 0
    cTime = 0
    angle = 0
    count = 0
    pushUpCheck1 = False
    pushUpCheck2 = False
    personalBest = 3
    color = (0,0,255)
    detector = pm.poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        h, w, c = img.shape

        img = detector.findPose(img)
        lmList = detector.getPosition(img, draw=False)

        if len(lmList) != 0:
            angle = int(detector.getAngle(img, 11, 13, 15))
            #cv2.putText(img, str(int(angle)), (45, 400),
            #           cv2.FONT_HERSHEY_COMPLEX, 5, (255, 0, 0), 5)
            


            
            anglePercentage = np.interp(angle, (69, 166), (100, 0))
            angleBar = np.interp (angle, (69, 166), (w, 300))
            angleBar = min(angleBar, w-300)

            cv2.rectangle(img, (300, h - 150), (w - 300, h-50), (255,0,0),3, 3)
            cv2.rectangle(img, (300, h-150), (int(angleBar), h-50),(255,0,0),cv2.FILLED)


            if count > personalBest:
                color = (0,255,0)
                personalBest = count
            cv2.putText(img, str(count), (90, h-50), cv2.FONT_HERSHEY_COMPLEX, 5, color, 5)
            cv2.putText(img, f'PB:{str(personalBest)}', (30, h-10), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)


            cv2.putText(img, f'{str(int(anglePercentage))}%', (880, h-85), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 4)

            if anglePercentage > 85:
                pushUpCheck1 = True
            
            if pushUpCheck1:
                if anglePercentage < 5:
                    pushUpCheck2 = True

            if pushUpCheck1 and pushUpCheck2:
                count = count + 1
                pushUpCheck2 = False
                pushUpCheck1 = False




        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        #cv2.putText(img, str(int(fps)), (70, 50),
        #            cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)

        img = cv2.resize(img, (800, 600))
        cv2.imshow("image", img)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == 27:  # Break the loop when the 'Esc' key is held
            break

    cap.release()
    cv2.destroyAllWindows()

main()