import os
import sys
import cv2
import psycopg2
from dotenv import load_dotenv
from tracker import *

# Load environment variables
load_dotenv()

# Create tracker object
tracker = EuclideanDistTracker()


def connectToDatabase():
  connection = None
  try:
    # Create connection
    print('Connecting to PostgreSQL database...')
    connection = psycopg2.connect(
        host=os.getenv('DATABASE_HOST'),
        database=os.getenv('DATABASE_NAME'),
        user=os.getenv('DATABASE_USER'),
        password=os.getenv('DATABASE_PASSWORD'),
        port=os.getenv('DATABASE_PORT'))

    # Create a cursor
    cursor = connection.cursor()

    # Execute a statement
    print('PostgreSQL database version:')
    cursor.execute('SELECT version()')

    # Display the PostgreSQL database server version
    db_version = cursor.fetchone()
    print(db_version)

    return connection
  except:
    print('Cannot connect to database')
    return None


def saveInDatabase(connection):
  try:
    connection.cursor().execute(
        'INSERT INTO "public"."vehicles" (time, cv_id) VALUES (CURRENT_TIMESTAMP)')
    # Save changes
    connection.commit()
  except Exception as e:
    print('Error while saving in database')
    print(e)


def main():
  savedVehicles = []
  # Connect to the PostgreSQL database
  dbConnection = connectToDatabase()

  try:
    videoParam = str(sys.argv[1])
  except:
    videoParam = 'video_1'

  capture = None
  region = None

  # Apply the corresponding video
  if videoParam == 'video_1':
    capture = cv2.VideoCapture("highway_1.mp4")
    region = [340, 720, 500, 800]
  elif videoParam == 'video_2':
    capture = cv2.VideoCapture("highway_2.m4v")
    region = [500, 720, 500, 1280]

  # Object detection from Stable camera
  object_detector = cv2.createBackgroundSubtractorMOG2(
      history=100, varThreshold=40)

  while True:
    ret, frame = capture.read()

    # Extract "region of interest"
    roi = frame[region[0]: region[1], region[2]: region[3]]

    # 1. Object detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
      # Calculate area and remove small elements
      area = cv2.contourArea(cnt)

      if area > 100:
        # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(cnt)

        detections.append([x, y, w, h])

    # 2. Object traking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
      x, y, w, h, id = box_id
      cv2.putText(roi, str(id), (x, y - 15),
                  cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
      cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
      # Save in database
      if id not in savedVehicles:
        saveInDatabase(dbConnection)
        savedVehicles.append(id)

    # cv2.imshow("roi", roi)
    # cv2.imshow("Mask", mask)
    cv2.imshow("Computer Vision | Luxtlalli", frame)

    # Wait for a key press
    key = cv2.waitKey(30)
    if key == 27:
      break

  capture.release()
  cv2.destroyAllWindows()


# Call main function
main()
