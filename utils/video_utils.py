import cv2

def read_video(path):
    """
    Reads a video from file and returns it as a list of frames.

    Parameters
    ----------
    path : str
        The path to the video file.

    Returns
    -------
    frames : list of numpy arrays
        A list of frames, where each frame is a numpy array of shape
        (height, width, channels) and dtype uint8.
    """
    
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def write_video(frames, path, fps=24):
    """
    Writes a list of frames to a video file.

    Parameters
    ----------
    frames : list of numpy arrays
        A list of frames, where each frame is a numpy array of shape
        (height, width, channels) and dtype uint8.
    path : str
        The path to save the video file.
    fps : int, optional
        Frames per second for the output video. Default is 30.
    """
    
    if not frames:
        raise ValueError("The list of frames is empty.")
    
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()