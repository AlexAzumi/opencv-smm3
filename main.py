import os
import sys
import cv2
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


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

    return cursor
  except:
    print('Cannot connect to database')
    return None


# Connect to the PostgreSQL database
dbCursor = connectToDatabase()

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

  try:
    height, width, _ = frame.shape
  except:
    print('\nEnd of the video')
    break

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

  # cv2.imshow("roi", roi)
  # cv2.imshow("Mask", mask)
  cv2.imshow("Computer Vision | Luxtlalli", frame)

  # Wait for a key press
  key = cv2.waitKey(30)
  if key == 27:
    break

capture.release()
cv2.destroyAllWindows()
