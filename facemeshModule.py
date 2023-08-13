import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self,staticMode=False,maxFaces=3,minDetectionCon=0.5,minTrackCon=0.5):
        self.staticMode=staticMode
        self.maxFaces=maxFaces
        self.minDetectionCon=minDetectionCon
        self.minTrackCon=minTrackCon
        self.mpDraw=mp.solutions.drawing_utils
        self.mpFaceMesh=mp.solutions.face_mesh
        self.faceMesh=self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,self.minDetectionCon,self.minTrackCon)
        self.drawSpecs=self.mpDraw.DrawingSpec(thickness=1,circle_radius=2)

    def findFaceMesh(self,img,draw=True):
        self.imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.faceMesh.process(self.imgRGB)
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,self.drawSpecs,self.drawSpecs)
            face=[]
            for id,lm in enumerate(faceLms.landmark):
                ih,iw,ic=img.shape
                x,y=int(lm.x*iw),int(lm.y*ih)
                print(id,x,y)
                face.append([x,y])
            faces = [face]
        return img,faces             



    
def main():
    cap=cv2.VideoCapture("/Users/soyamdas/Desktop/Face Edges Tracker/sample2.MOV")
    pTime=0
    detector=FaceMeshDetector(maxFaces=3)
    while True:
        success, img = cap.read()
        img,faces= detector.findFaceMesh(img)
        if len(faces)!=0:
            print(faces[0])
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,f"FPS:{int(fps)}",(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        cv2.imshow("Image",img)
        cv2.waitKey(1)




if __name__=='__main__':
    main()