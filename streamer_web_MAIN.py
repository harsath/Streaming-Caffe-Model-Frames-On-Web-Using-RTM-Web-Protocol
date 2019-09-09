#Model Source Google's Caffe trained models : 140000 training images
#Model Author: Harsath
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from flask import Flask
import threading
import imutils
from flask import Response
from flask import Flask
from flask import render_template


outputFrame = None
lock = threading.Lock()
app = Flask(__name__)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())
# ap = argparse.ArgumentParser()
ap.add_argument("-i", "--ip", type=str, required=True,
	help="ip address of the device")
ap.add_argument("-o", "--port", type=int, required=True,
	help="ephemeral port number of the server (1024 to 65535)")
ap.add_argument("-f", "--frame-count", type=int, default=32,
	help="# of frames used to construct the background model")
args = vars(ap.parse_args())


@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")


# load our serialized model from disk
print("Caffe Model loading on CUDA")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
 
# initialize the video stream and allow the camera sensor to warm up
print("About to Stream")
vs = VideoStream(src=0).start()
time.sleep(2.0)

def face_detect():
	global vs, outputFrame, lock
	# loop over the frames from the video stream
	while True:
	    # grab the frame from the threaded video stream and resize it
	    # to have a maximum width of 400 pixels
	    frame = vs.read()
	    frame = imutils.resize(frame, width=400)
	 
	    # grab the frame dimensions and convert it to a blob
	    (h, w) = frame.shape[:2]
	    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
	 
	    # pass the blob through the network and obtain the detections and
	    # predictions
	    net.setInput(blob)
	    detections = net.forward()
	    # print(detections.shape[2])
	    count = 0    
	    
	    # loop over the detections
	    for i in range(0, detections.shape[2]):
	        # extract the confidence (i.e., probability) associated with the
	        # prediction
	        confidence = detections[0, 0, i, 2]
	        # print(confidence)
	        #print(confidence * 100)

	 
	        # filter out weak detections by ensuring the `confidence` is
	        # greater than the minimum confidence
	        if confidence < args["confidence"]:
	            continue
	        count += 1 
	        # compute the (x, y)-coordinates of the bounding box for the
	        # object
	        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
	        (startX, startY, endX, endY) = box.astype("int")
	 
	        # draw the bounding box of the face along with the associated
	        # probability
	        text = f"Counts : {str(count)}"
	        y = startY - 10 if startY - 10 > 10 else startY + 10
	        cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
	        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
	        cv2.resize(frame, (1000,1000))
	    with lock:
	    	outputFrame = frame.copy()
	    # show the output frame
	    # cv2.imshow("Frame", frame)
	    # key = cv2.waitKey(1) & 0xFF
	 
	    # if the `q` key was pressed, break from the loop
	    # if key == ord("q"):
 
def generate():
	global outputFrame, lock

	
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")
# cv2.destroyAllWindows()
# vs.stop()
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments

	# start a thread that will perform motion detection
	t = threading.Thread(target=face_detect)
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()






