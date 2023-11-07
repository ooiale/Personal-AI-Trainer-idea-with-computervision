import cv2
import mediapipe as mp
import time
import math

class poseDetector():

    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackingCon=0.5):

        self.mode = mode
        #self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode = self.mode, smooth_landmarks = self.smooth, min_detection_confidence = self.detectionCon, min_tracking_confidence = self.trackingCon )
        #self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if (self.results.pose_landmarks):
            if draw:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition (self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w) , int(lm.y * h) #pixel values
                if draw:
                    cv2.circle(img, (cx,cy), 10, (255,0,0), cv2.FILLED)
                self.lmList.append([id, cx,cy])
        return self.lmList

    def getAngle (self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        if draw:
            cv2.circle(img, (x1,y1), 20, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x2,y2), 20, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x3,y3), 20, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x1,y1), 10, (0,0,255), 2)
            cv2.circle(img, (x2,y2), 10, (0,0,255), 2)
            cv2.circle(img, (x3,y3), 10, (0,0,255), 2)

        vector1 = (x1 - x2, y1 - y2)
        vector2 = (x3 - x2, y3 - y2)

        # Calculate the angle in radians
        angle_radians = math.atan2(vector1[1], vector1[0]) - math.atan2(vector2[1], vector2[0])

        # Ensure the angle is within the range [0, 2*pi]
        if angle_radians < 0:
            angle_radians += 2 * math.pi

        # Convert the angle to degrees
        angle_degrees = math.degrees(angle_radians)

        return angle_degrees
        


def main():
    cap = cv2.VideoCapture('videos/running.mp4')
    pTime = 0
    cTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()

        img = detector.findPose(img)
        lmList = detector.getPosition(img, draw=False)
        print(lmList[14])
        cv2.circle(img, (lmList[14][1] , lmList[14][2]), 10, (0,255,0), cv2.FILLED)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)

        img = cv2.resize(img, (800, 600))
        cv2.imshow("image", img)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == 27:  # Break the loop when the 'Esc' key is held
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
