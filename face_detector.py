import cv2
from datetime import datetime
import pandas
import os
from random import randrange


status_list = [None, None]
times = []
df = pandas.DataFrame(columns=["Start", "End"])

# load the cascade
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
trained_upper_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
trained_eye_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

webcam = cv2.VideoCapture(1)

while True:
	status = 0
	successful_frame_read, frame = webcam.read()

	grayscaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	face_coordinates = trained_face_data.detectMultiScale(grayscaled_image)

	for (x, y, w, h) in face_coordinates:
		status = 1
		cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)
		eye_coordinates = trained_eye_data.detectMultiScale(grayscaled_image, scaleFactor=2.0, minNeighbors=16)
		for (x, y, w, h) in eye_coordinates:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, randrange(256), 0), 1)
			upper_coordinates = trained_upper_data.detectMultiScale(grayscaled_image)
			for (x, y, w, h) in upper_coordinates:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 3)
			

	status_list.append(status)

	if status_list[-1] == 1 and status_list[-2] == 0:
		times.append(datetime.now())
	if status_list[-1] == 0 and status_list[-2] == 1:
		times.append(datetime.now())
	if status_list[-1] == 1 and status_list[-2] == 0:
		print("Motion Detected")
		print('\a\a')						  # This part of code only works on mac
		os.system('say "Motion Detected"')    # This part of code only works on mac
		
	cv2.imshow("Face Detector", frame)

	key = cv2.waitKey(1)
	if key == 81 or key == 113:
		if status == 1:
			times.append(datetime.now())
		break

print(status_list)

for i in range(0, len(times), 2):
    df = df.append({"Start": times[i], "End": times[i + 1]}, ignore_index=True)

df.to_csv("Time.csv")

webcam.release()

print("\nCode Completed\n")
