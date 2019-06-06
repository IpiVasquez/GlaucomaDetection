#!/usr/bin/env python3
import cv2
import mahotas as mt
import numpy as np

#Descriptores de forma
#c1: perimetro
#c2: area
#c3: compacidad
#c4: centroide x
#c5: centroide y
#c6: longitud eje a
#c7: longitud eje b
#c8: angulo entre ejes
#c9: Box ax
#c10: Box ay
#c11: Box bx
#c12: Box by


def phorm(img):
    features = np.zeros(12)
    #perimetro
    canny = cv2.Canny(img, 50, 150)
    features[0] = canny.sum()
    #Area
    features[1] = img.sum()
    #compacidad
    features[2] = features[0]**2 / features[1]
    #centroide x, y
    cenX=0
    cenY=0
    npc=0
    #Box contend
    ax=0
    ay=0
    bx=0
    by=0
    cx=0
    cy=0
    dx=0
    dy=0
    ban = 0
    for xc in range(len(img)):
        for yc in range(len(img[xc])):
           if(img[xc,yc]==1):
               cenX = cenX + xc
               cenY = cenY + yc
               npc = npc +1
               if(ban == 0):
                    by=yc
                    bx=xc
                    ax=xc
                    ay=yc
                    ban=1
               else:
                    dy = yc
                    dx = xc
                    cx = xc
                    cy = yc
    features[3] = int(cenX / npc)
    features[4] = int(cenY / npc)    
    
    features[8] = ax
    features[9] = bx
    features[10] = cx
    features[11] = dx

    #c6-7: Longitud de ejes
    features[5] = np.sqrt((ax - cx)**2 + (ay - cy)**2)
    features[6] = np.sqrt((bx - dx)**2 + (by - dy)**2)
    
    #c8: angulo entre ejes
    #distancia del centroide al punto b
    cen2b = np.sqrt((features[3] - bx)**2 + (features[5] - by)**2)
    cen2c = np.sqrt((features[3] - cx)**2 + (features[5] - cy)**2)
    features[7] = np.arctan(cen2c/cen2b)*np.pi/2

    return features


