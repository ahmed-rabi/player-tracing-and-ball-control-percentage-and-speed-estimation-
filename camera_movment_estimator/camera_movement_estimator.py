import pickle
import cv2
import numpy as np
import os
class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame_copy = frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 500), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f'Movement X: {x_movement:.2f}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f'Movement Y: {y_movement:.2f}',
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            output_frames.append(frame)
        return output_frames
    
    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for obj, obj_tracks in tracks.items():
            for frame_num, frame_data in enumerate(obj_tracks):
                for tarck_id, track_data in frame_data.items():
                    position = track_data.get('position', [0, 0])
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    tracks[obj][frame_num][tarck_id]['position_adjusted'] = position_adjusted
                    
    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        camera_movement = [[0, 0]] * len(frames)  # Initialize with no movement
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            new_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, status, error = cv2.calcOpticalFlowPyrLK(
                old_gray, new_gray, old_features, None, **self.lk_params
            )

            max_distance = 0
            camera_movement_x = camera_movement_y = 0, 0 
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_piont = new.ravel()
                old_features_point = old.ravel()
                distance = np.linalg.norm(new_features_piont - old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = new_features_piont[0] - old_features_point[0], new_features_piont[1] - old_features_point[1]

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(new_gray, **self.features)
            old_gray = new_gray.copy()
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
        return camera_movement
