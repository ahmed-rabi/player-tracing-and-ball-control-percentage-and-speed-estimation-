

import cv2
import numpy as np


class SpeedAndDistanceEstimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24  # Assuming a frame rate of 24 FPS


    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        for obj, obj_tracks in tracks.items():
            if obj=='ball' or obj=='referees':
                continue
            number_of_frames = len(obj_tracks)

            for frame_num in range(0, number_of_frames, self.frame_window):
                if frame_num not in obj_tracks:
                    continue
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in obj_tracks[frame_num].items():
                    if track_id not in obj_tracks[last_frame]:
                        continue

                    if 'position' not in obj_tracks[frame_num][track_id]:
                        continue

                    current_position = np.array(obj_tracks[frame_num][track_id]['position_transformed'])
                    previous_position = np.array(obj_tracks[last_frame][track_id]['position_transformed'])

                    if current_position is None or previous_position is None:
                        continue

                    distance = np.linalg.norm(current_position - previous_position)
                    speed = distance / (self.frame_window / self.frame_rate)
                    speed = speed * 3.6  # Convert m/s to km/h
                    
                    if obj not in total_distance:
                        total_distance[obj] = {}
                    if track_id not in total_distance[obj]:
                        total_distance[obj][track_id] = 0

                    total_distance[obj][track_id] += distance

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in obj_tracks[frame_num_batch]:
                            continue
                        obj_tracks[frame_num_batch][track_id]['speed'] = speed
                        obj_tracks[frame_num_batch][track_id]['distance'] = total_distance[obj][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for obj, obj_tracks in tracks.items():
                if obj == 'ball' or obj == 'referees':
                    continue
                for _, track_data in obj_tracks[frame_num].items():
                    if 'speed' in track_data:
                        speed = track_data.get('speed', None)
                        distance = track_data.get('distance', None)
                        if speed is None or distance is None:
                            continue
                        bbox = track_data['bbox']
                        position = int((bbox[0] + bbox[2]) / 2), int(bbox[3])
                        position = list(position)
                        position[1] += 40
                        position = tuple(map(int, position))
                        cv2.putText(frame, f'Speed: {speed:.2f} km/h', position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f'Distance: {distance:.2f} m', (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 0), 2)
            output_frames.append(frame)
        return output_frames
