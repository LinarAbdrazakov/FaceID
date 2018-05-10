# -*- coding: utf-8 -*-
# Autor: Abdrazakov Linar
# Date mode: 27 dec. 2017

import numpy as np
import dlib
import cv2
from scipy.spatial import distance
import time

from subprocess import Popen, PIPE

def voice_text(text):
    procces = Popen(['say'], stdin = PIPE)
    procces.stdin.write(text.encode('utf-8'))
    procces.stdin.close()

class FaceID(object):

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()
        #self.cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
        self.predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
        self.face_rec = dlib.face_recognition_model_v1("data/dlib_face_recognition_resnet_model_v1.dat")

    def add_id_face(self, name):
        _, image = self.cap.read()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #faces = self.cascade.detectMultiScale(gray_image, 1.3, 5)
        faces = self.detector(image, 1)
        if len(faces) == 1:
            #x, y, w, h = faces[0]
            #rect = Rect(x, y, w, h)
            face = faces[0]
            shape = self.predictor(gray_image, face)
            descriptor = self.face_rec.compute_face_descriptor(image, shape)
            #print descriptor.dtype
            with open("Names_users.txt", 'a') as file_users:
                file_users.write(name+'\n')
            with open("ID_users/"+name+".txt", 'w') as id_file:
                id_file.write(self.vector_to_str(descriptor))
            cv2.imwrite("Photo_users/"+name+".jpg", image)
            print "Succes add face user!"
        else:
            print "Error!!!"
            print "To image", len(faces), "people!"

    def get_id_faces(self):
        _, image = self.cap.read()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(image, 1)
        for face in faces:
            shape = self.predictor(gray_image, face)
            descriptor = self.vector_to_m(self.face_rec.compute_face_descriptor(image, shape))
            it_user = False
            with open("Names_users.txt") as file_users:
                names = file_users.read()
                min_dist = 1.0
                Name_user = "Not user"
                for name in names.split('\n'):
                    if len(name) == 0: break
                    with open("ID_users/"+name+".txt") as file_decriptor:
                        descriptor_user = self.str_to_m(file_decriptor.read())
                        #print descriptor_user, descriptor
                        dist = distance.euclidean(descriptor_user, descriptor)
                        if(dist < 0.6 and dist < min_dist):
                            min_dist = dist
                            Name_user = name
        
            print Name_user, min_dist
            voice_text("Hello, " + Name_user)
        
        cv2.imshow("Camera", image)
        k = cv2.waitKey(1)
        return k
                            
    def vector_to_m(self, vec):
        m = []
        for el in vec:
            m.append(el)
        return np.array(m)

    def vector_to_str(self, vec):
        line = ''
        for el in vec:
            line += str(el) + ' '
        return line

    def str_to_m(self, line):
        m = []
        for el in line.split():
            m.append(float(el))
        return np.array(m)


if __name__ == "__main__":
    FaceID = FaceID()
    #print "Your name: "
    #name = raw_input()
    #FaceID.add_id_face(name)
    while True:
        time.sleep(3)
        k = FaceID.get_id_faces()
        if(k == ord('q')): break
#print "OK"
