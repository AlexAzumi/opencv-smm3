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
    region = [600, 720, 600, 1280]
  elif videoParam == 'video_3':
    capture = cv2.VideoCapture("highway_3.mp4")
    region = [150, 360, 0, 500]

  # Trained XML classifier describes some features of some objects we want to detect
  car_cascade = cv2.CascadeClassifier("carx.xml")

  while True:
    ret, frame = capture.read()

    # Extract "region of interest"
    roi = frame[region[0]: region[1], region[2]: region[3]]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.1, 2)

    for (x, y, w, h) in cars:
      cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 255), 3)

    #cv2.imshow("Region of interest", roi)
    cv2.imshow("Computer Vision | Luxtlalli", frame)

    # Wait for a key press
    key = cv2.waitKey(30)
    if key == 27:
      break

  capture.release()
  cv2.destroyAllWindows()


# Call main function
main()
