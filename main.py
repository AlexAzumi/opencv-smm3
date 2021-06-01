import sys
import cv2

try:
  videoParam = str(sys.argv[1])
except:
  videoParam = 'video_1'

capture = None
region = None

# Apply the corresponding video
if (videoParam == 'video_1'):
  capture = cv2.VideoCapture("highway_1.m4v")
  region = [500, 720, 500, 1280]

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=40)

while True:
  ret, frame = capture.read()
  height, width, _ = frame.shape

  # Extract "region of interest"
  roi = frame[region[0]: region[1], region[2]: region[3]]

  # Object detection
  mask = object_detector.apply(roi)
  _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  for cnt in contours:
    # Calculate area and remove small elements
    area = cv2.contourArea(cnt)

    if area > 100:
      # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
      x, y, w, h = cv2.boundingRect(cnt)
      cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

  cv2.imshow("roi", roi)
  cv2.imshow("Frame", frame)
  cv2.imshow("Mask", mask)

  # Wait for a key press
  key = cv2.waitKey(30)
  if key == 27:
    break

capture.release()
cv2.destroyAllWindows()
