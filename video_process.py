#https://gist.github.com/victorbstan/2130079

import cv2.cv as cv
from cv import highgui
import pygame
import sys

camera = highgui.cvCreateCameraCapture(-1)

def get_image():
    im = highgui.cvQueryFrame(camera)
    return opencv.adaptors.Ip12PIL(im)

fps = 30.0
pygame.init()
window = pygame.display.set_mode((640,480))
pygame.display.set_caption("Demo")
screen = pygame.display.get_surface()

while True:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
            sys.exit(0)
    im = get_image()
    pg_img = pygame.image.frombuffer(im.tostring(), im.size, im.mode)
    screen.blit(pg_img, (0,0))
    pygame.display.flip()
    pygame.time.delay(int(1000 * 1.0/fps))
