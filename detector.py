# Imports
import time
import cv2
import dlib
import playsound
import imutils
from imutils import face_utils
from imutils.video import VideoStream
import numpy as np
from scipy.spatial import distance
from threading import Thread


# Assignments
drowsy_frame_count = 0
wake_up = False


# Thresholds setter
def set_thresholds(eye_aspect_ratio, frame_threshold):

	global drowsy_ear_threshold, drowsy_frame_threshold
	drowsy_ear_threshold = eye_aspect_ratio
	drowsy_frame_threshold = frame_threshold


# Alert driver on being drowsy
def alert_driver():

	print "\tDrowsiness detected, waking the driver up.."
	while wake_up:
		playsound.playsound("audio/alert.mp3")
	print "\tDriver is now awake again!"


# Update driver status on screen
def update_driver_status(current_frame, status):

	cv2.putText(current_frame, status , (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 76, 0), )


# Compute eye's Eye Aspect Ratio (EAR)
def compute_ear(driver_eye):

	# Calculate the eye's vertical lines euclidean distance - Landmark points:(2,6) and (3,5)
	vertical_distance_1 = distance.euclidean(driver_eye[1], driver_eye[5])
	vertical_distance_2 = distance.euclidean(driver_eye[2], driver_eye[4])

	# Calculate the eye's horizontal line euclidean distance - Landmark points: (1,4) 
	horizontal_distance = distance.euclidean(driver_eye[0], driver_eye[3])
 
	# Eye Aspect ratio computation
	eye_aspect_ratio = (vertical_distance_1+vertical_distance_2)/(2.0*horizontal_distance)
	
	return eye_aspect_ratio


# Create frontal face detector
def create_detector():

	global detector
	detector = dlib.get_frontal_face_detector()
	print "=>	Frontal face detector has been created successfully.\n"


# Load facial shape predictor
def load_predictor(predictor_path=None):

	global predictor

	# Default predictor loaded unless parameter passed
	if predictor_path is None:
		predictor_path = "predictors/face.dat"

	try:
		predictor = dlib.shape_predictor(predictor_path)
	except:
		print "=>	Facial shape predictor creation failed"
		exit()
	
	print "=>	Facial shape predictor has been loaded successfully.\n"


# Visualize eye
def visualize_eye(frame,left_eye,right_eye):

	left_eye_border = cv2.convexHull(left_eye)
	right_eye_border = cv2.convexHull(right_eye)
	cv2.drawContours(frame, [left_eye_border], -1, (0, 255, 0), 1)
	cv2.drawContours(frame, [right_eye_border], -1, (0, 255, 0), 1)


# Fetch eyes landmarks
def fetch_eye_landmarks():

	global left_eye_start, left_eye_end, right_eye_start, right_eye_end
	(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]	# Left eye indexes
	(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]	# Right eye indexes
	print "=>	Eye landmarking indexes haveindexes been loaded successfully.\n\
	Left eye start index = {}\n\tLeft eye end index = {}\n\tRight eye start index = {}\n\tRight eye end index = {}\n"\
	.format(left_eye_start, left_eye_end, right_eye_start, right_eye_end)


# Dynamically compute drowsiness parameters | Optimization
def compute_drowsy_ear(spectacles=True):
	# To be implemented
	if spectacles:	# Automated spectacles detection to be implemented
		drowsy_ear = 0.20
	else:
		drowsy_ear = 0.25

	return drowsy_ear


# Start webcam monitoring
def start_monitoring():

	print "=>	Real-time monitoring has started..\n"
	video_stream = VideoStream(0).start()
	time.sleep(1.0)
	global drowsy_frame_count, drowsy_ear_threshold, drowsy_frame_threshold, wake_up
 	detected_ear = 0

 	# Monitor indefinitely
	while True:

		# Pre-process frame
		frame = imutils.resize(video_stream.read(), width=600)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Detect face(s) found in frame
 		detected_facial_areas = detector(gray_frame, 0)
 		cv2.putText(frame, "Driver status: ", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 76, 0), 2)

 		# Detection starts here
 		for facial_area in detected_facial_areas:

 				# Predict facial features in the facial area
 				facial_shape = face_utils.shape_to_np(predictor(gray_frame, facial_area))

 				# Slice the eyes and compute their EAR (eye aspect ratio) values
 				detected_left_eye = facial_shape[left_eye_start:left_eye_end]
				detected_right_eye = facial_shape[right_eye_start:right_eye_end]
				left_eye_ear = compute_ear(detected_left_eye)
				right_eye_ear = compute_ear(detected_right_eye)

				# Compute the average EAR of the detected eyes
				detected_ear = (left_eye_ear + right_eye_ear)/2.0

				# Visualize the detected features
				visualize_eye(frame, detected_left_eye,detected_right_eye)

				# Check for drowsiness
				if detected_ear < drowsy_ear_threshold: # Drowsy

					# Count drowsy frames
					drowsy_frame_count += 1
 					
 					# Check if threshold exceeded
					if drowsy_frame_count >= drowsy_frame_threshold:
						if not wake_up:
							wake_up = True
							alarm_thread = Thread(target=alert_driver)
							alarm_thread.deamon = True
							alarm_thread.start()

						# Display drowsiness alert
						update_driver_status(frame, "Drowsy")
 				
				else: # Not drowsy
					drowsy_frame_count = 0
					wake_up = False
					update_driver_status(frame, "Awake")
		
		# Display EAR value (for tuning purposes)
		cv2.putText(frame, "EAR value : {:.4f}".format(detected_ear), (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 111, 130), 2)
		
		# Display output
		cv2.imshow("Drowsiness Detector", frame)

		# Exiting using ESC
		pressed_key = cv2.waitKey(1) & 0xFF
		if pressed_key == 27:
			break

	# Exit
	cv2.destroyAllWindows()
	video_stream.stop()


# Run
if __name__ == "__main__":

	drowsy_ear = compute_drowsy_ear(spectacles=True) # Normal EAR relative to the driver's posture, eye shape and eye spectacles detection
	set_thresholds(eye_aspect_ratio = drowsy_ear, frame_threshold = 25) # Greater EAR or lesser frame threshold = Greater sensitivity
	create_detector()	# Detector
	load_predictor(predictor_path="predictors/face.dat")	# Predictor
	fetch_eye_landmarks()	# Eye landmarks
	start_monitoring()	# Start monimotring