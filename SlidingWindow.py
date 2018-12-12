import cv2


def sliding_window(image, stepSize, windowSize):
    """
    Slide a window across an image

    :param image: The image to slide window across
    :param stepSize: Pixels to skip in both (x, y) direction. The smaller the higher the computation. Avoid 1 as value
    :param windowSize: e.g. (winW, winH) defines the width and height of the window
    :return:
    """
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


if __name__ == '__main__':

    # Load the image
    imagePath = "images/test.jpg"
    image = cv2.imread(imagePath)

    # Optionally scale down the image pixel size for faster sliding
    scale = 0.5
    image = cv2.resize(image, None, fx=scale, fy=scale)

    # Define the window width/height relative to the image width's size
    relativeSize = image.shape[0] / 3
    (winW, winH) = (relativeSize, relativeSize)
    # (winW, winH) = (128, 128)

    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(image, stepSize=32, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # Process window frame if needed here for e.g. using a ML classifier
        # processWindow(window)

        # Optionally preview the window over the original image
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
