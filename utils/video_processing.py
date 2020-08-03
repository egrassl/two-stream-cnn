import cv2
import numpy as np


OPTFLOW = cv2.DualTVL1OpticalFlow_create()


def read_video(src):
    # Extract video frames
    video = cv2.VideoCapture(src)
    frames = []
    success = 1
    while success:
        success, frame = video.read()

        # Appends frame only if it is not empty
        if frame is not None:
            frames.append(frame)

    return frames


def calculate_flow(frame1, frame2, bound=15):
    # Get motion flow
    umat1 = cv2.UMat(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)).get()
    umat2 = cv2.UMat(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)).get()
    flow = OPTFLOW.calc(umat1, umat2, None)

    # Transform data back to image format
    assert flow.dtype == np.float32
    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
    return flow


def extract_motion_from_frames(frames, init_frame, end_frame, nb_frames, width=350):
    motion_frames = [calculate_flow(frame_resize(frames[i]), frame_resize(frames[i+1])) for i in range(init_frame, end_frame)]

    # Stacked array has shape of width * height * 2 * nb_frames
    stacked_flow = np.zeros(motion_frames[0][:, :, 1].shape + (2 * nb_frames,), dtype=np.uint8)

    for i in range(0, nb_frames):

        # Stack horizontal and vertical motion channels
        stacked_flow[:, :, i] = motion_frames[i][:, :, 0]
        stacked_flow[:, :, i + nb_frames] = motion_frames[i][:, :, 1]

    return stacked_flow


def frame_resize(image, width=350, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized