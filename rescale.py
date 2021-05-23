import cv2 as cv


def changeRes(width, height):
  # Live video (or camera)
  capture.set(3, width)
  capture.set(4, height)


def rescaleFrame(frame, scale=0.75):
  # Images, video & live video (or camera)
  height = int(frame.shape[0] * scale)
  width = int(frame.shape[1] * scale)
  dimension = (width, height)

  return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)


capture = cv.VideoCapture('Videos/dog.mp4')

while True:
  isTrue, frame = capture.read()
  frame_resized = rescaleFrame(frame, scale=0.2)

  cv.imshow('Video', frame)
  cv.imshow('Video Resized', frame_resized)

  if cv.waitKey(20) & 0xFF == ord('d'):
    break

cv.waitKey(0)
