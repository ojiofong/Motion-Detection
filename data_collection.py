import cv2
import time


def preprocessImage(image):
    h, w, channels = image.shape
    y1, y2 = 0, h
    x1, x2 = 45, w
    crop_img = image[y1:y2, x1:x2].copy()
    return crop_img


if __name__ == '__main__':

    count = 0
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 232)

    ret, image = cap.read()

    while ret:
        count = count + 1
        ret, image = cap.read()
        image = preprocessImage(image)

        # Load the image
        dir = "/Users/oofong/Projects/Hackathon/HackathonProject/kitchen_bot/sink_photos/clean/"
        fileName = "temp{}.jpg".format(count)
        imagePath = dir + fileName
        cv2.imwrite(imagePath, image)
        image = cv2.imread(imagePath)
        exit("done bro")

        print ("got frame", image.shape, count)
        time.sleep(1)

        if count >= 300:
            exit("Finished. Got {} samples".format(count))

        cv2.imshow("Window", image)
        # time.sleep(1)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

        continue
