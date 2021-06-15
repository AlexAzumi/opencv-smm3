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
    print('PostgreSQL: Connecting to database...')
    connection = psycopg2.connect(
        host=os.getenv('DATABASE_HOST'),
        database=os.getenv('DATABASE_NAME'),
        user=os.getenv('DATABASE_USER'),
        password=os.getenv('DATABASE_PASSWORD'),
        port=os.getenv('DATABASE_PORT'))

    print("PostgreSQL: CONNECTED!")

    return connection
  except:
    print('PostgreSQL: An error ocurred while connecting to the database')
    return None


def saveInDatabase(connection):
  try:
    connection.cursor().execute(
        'INSERT INTO "public"."vehicles" (time) VALUES (CURRENT_TIMESTAMP)')
    # Save changes
    connection.commit()
  except Exception as e:
    print('PostgreSQL: Error while saving in database')
    print(e)


def main():
  videoParam = ''
  videoIndex = 0
  savedVehicles = []
  # Connect to the PostgreSQL database
  dbConnection = connectToDatabase()

  try:
    videoParam = str(sys.argv[1]) or '-video'
    videoIndex = int(sys.argv[2]) or 0
  except:
    videoParam = '-video'

  capture = None
  region = None

  # Apply the corresponding video
  if videoParam == '-video' and videoIndex == 0:
    capture = cv2.VideoCapture("example_1.mp4")
    region = [580, 720, 580, 1280]
  elif videoParam == '-video' and videoIndex == 1:
    capture = cv2.VideoCapture("example_2.mp4")
    region = [340, 720, 500, 800]
  elif videoParam == '-video' and videoIndex == 2:
    capture = cv2.VideoCapture("example_3.mp4")
    region = [150, 360, 0, 500]
  else:
    print("App: Video not found. Exiting program...")
    exit(0)

  # Trained XML classifier that describes some features of some objects we want to detect
  car_cascade = cv2.CascadeClassifier("cars.xml")

  while True:
    ret, frame = capture.read()

    # Extract "region of interest"
    roi = frame[region[0]: region[1], region[2]: region[3]]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.1, 2)

    # Object tracking
    tracked_objects = tracker.update(cars)

    for (x, y, w, h, id) in tracked_objects:
      cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 255), 3)
      cv2.putText(roi, str(id), (x, y - 15),
                  cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
      # Save in database
      if id not in savedVehicles and dbConnection is not None:
        saveInDatabase(dbConnection)
        savedVehicles.append(id)

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
