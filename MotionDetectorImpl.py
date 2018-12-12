"""
This project uses open-cv for motion/no motion detection
An action is performed if no motion is detected for X amount of seconds
"""
import datetime
import json

import cv2

conf = json.load(open("conf.json"))

print "[INFO] Initializing..."
# initialize the camera and grab a reference to the raw camera capture
video_capture = cv2.VideoCapture(1)

avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0

text_motion_detected = "Motion detected"
text_no_motion_detected = "No motion"

lastMotionDetectionTime = None


def performActionIfNoMotionForXSeconds(frame):
    """
    Perform an action using the frame if there is no motion for X seconds
    :param frame:
    :return:
    """
    global lastMotionDetectionTime

    if lastMotionDetectionTime is None:
        return

    # time difference since the last motion was detected
    timeDiff = (datetime.datetime.now() - lastMotionDetectionTime).seconds  # type: datetime

    if timeDiff > conf["min_no_motion_seconds"]:
        lastMotionDetectionTime = None
        print "perform action on this frame", frame.shape, timeDiff


while True:
    # capture frames from the camera
    ret, frame = video_capture.read()
    timestamp = datetime.datetime.now()
    text = text_no_motion_detected

    # resize the frame, convert it to grayscale, and blur it
    scale = 0.50
    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the average frame is None, initialize it
    if avg is None:
        print ("[INFO] starting background model...")
        avg = gray.copy().astype("float")
        continue

    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # threshold the delta image, dilate the thresholded image to fill
    # in holes, then find contours on thresholded image
    thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < conf["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = text_motion_detected
        lastMotionDetectionTime = datetime.datetime.now()
        print "motion detected"

    # draw the text and timestamp on the frame
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, "Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # check to see if motion is detected
    if text == text_motion_detected:
        # check to see if enough time has passed between uploads
        if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
            # increment the motion counter
            motionCounter += 1

            # check to see if the number of frames with consistent motion is high enough
            if motionCounter >= conf["min_motion_frames"]:
                print ("Motion detection is fully satisfied")
                # Upload the frame or write to the disk etc.
                # path = timestamp.strftime("%b-%d_%H_%M_%S" + ".jpg")
                # cv2.imwrite(path, frame)
                lastUploaded = timestamp
                motionCounter = 0

    # otherwise, no motion is detected
    else:
        motionCounter = 0

    # check to see if the frames should be displayed to screen
    if conf["show_video"]:
        # display the video feed
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    # finally process frame if needed
    performActionIfNoMotionForXSeconds(frame)
