from ultralytics import YOLO
import supervision as sv
import pickle
import os
import pandas as pd
import cv2
import numpy as np

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    def add_positions_to_tracks(self, tracks):
        
        for obj, obj_data in tracks.items():
            for frame_num, frame_data in enumerate(obj_data):
                    for track_id, track_info in frame_data.items():
                            bbox = track_info['bbox']
                            if obj == 'ball':
                                positions = [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]
                            else:
                                positions = int((bbox[0] + bbox[2]) / 2), int(bbox[3])
                            tracks[obj][frame_num][track_id]['positions'] = positions


    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1:{'bbox':x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
    
    def detect_frames(self, frames):
        """
        Detect objects in a list of frames using the YOLO model.
        Args:
            frames (list): List of frames to process.

        Returns:
            detections: List of detection results for each frame.
        """

        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            results = self.model.predict(batch, conf=0.1)
            detections.extend(results)
        return detections

    def get_object_tracks(self, frame, read_from_stub=False, stub_path=None):
        """
        Get object tracks from a single frame.
        Args:
            frame: A single video frame.

        Returns:
            tracks: List of tracked objects in the frame.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        tracks = {
            'players':[],
            'referees':[],
            'ball':[],
        }
        detections = self.detect_frames(frame)

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)

            for object_id, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_id] = cls_names_inv['player']
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}

                if class_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}

            for frame_detection in detection_supervision:

                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox': bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        
        return tracks
    
    def draw_triangle(self, frame, bbox, color=(0, 0, 255), label=None):
        """
        Draw a triangle on the frame based on the bounding box.
        Args:
            frame: A single video frame.
            bbox: Bounding box coordinates in the format [x1, y1, x2, y2].
            color: Color of the triangle.
            label: Optional label to display.

        Returns:
            Annotated frame with the triangle drawn.
        """
        x1, y1, x2, y2 = map(int, bbox)
        center = (x1 + x2) // 2
        
        points = [
            (center, y1),
            (x1 - 5, y1 - 10),
            (x2 + 5, y1 - 10)
        ]
        
        cv2.drawContours(frame, [np.array(points)], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [np.array(points)], 0,(0, 0, 0), 2)
        
        return frame

    def draw_ellipse(self, frame, bbox, color=(0, 255, 0), label=None):
        """
        Draw an ellipse on the frame based on the bounding box.
        Args:
            frame: A single video frame.
            bbox: Bounding box coordinates in the format [x1, y1, x2, y2].
            color: Color of the ellipse.
            label: Optional label to display.

        Returns:
            Annotated frame with the ellipse drawn.
        """
        x1, y1, x2, y2 = map(int, bbox)
        center = ((x1 + x2) // 2,  y2)
        
        cv2.ellipse(frame, center, axes=(x2 - x1, int(0.35*(x2 - x1))), angle=0.0, 
                    startAngle=-45, 
                    endAngle=235, color=color,
                    lineType=cv2.LINE_4, 
                    thickness=2)
        if label:
            center = (x1 + x2) // 2 
            rect_width = 40
            rect_height = 20
            x1_rect = center - rect_width // 2
            x2_rect = center + rect_width // 2
            y1_rect = (y2 - rect_height // 2) + 15
            y2_rect = (y2 + rect_height // 2) + 15
            x1_text = x1_rect + 12
            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)
            cv2.putText(frame, label, (x1_text, y2_rect - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        (0,0,0), 1)
        return frame
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        Draw the team ball control information on the frame.
        Args:
            frame: A single video frame.
            frame_num: The current frame number.
            team_ball_control: List of team IDs controlling the ball.

        Returns:
            Annotated frame with team ball control information.
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        
        team_1 = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2 = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team1 = team_1 / (team_1 + team_2 + 1e-6)  # Avoid division by zero
        team2 = team_2 / (team_1 + team_2 + 1e-6)  # Avoid division by zero
        # Draw the team ball control information
        cv2.putText(frame, f'Team 1 Ball Control: {team1}', (1400, 870), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, f'Team 2 Ball Control: {team2}', (1400, 900), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame

    def draw_annotations(self, frame, tracks, team_ball_control):
        output = []

        for frame_num, frame_data in enumerate(frame):
            annotated_frame = frame_data.copy()

            for player_id, player_data in tracks['players'][frame_num].items():
                bbox = player_data['bbox']
                color = player_data.get('team_color', (0, 0, 255))
                annotated_frame = self.draw_ellipse(annotated_frame, bbox, color=color, 
                                                    label=f'{player_id}')
                if player_data.get('has_ball'):
                    annotated_frame = self.draw_triangle(annotated_frame, bbox, color=color)
            for referee_id, referee_data in tracks['referees'][frame_num].items():
                bbox = referee_data['bbox']
                annotated_frame = self.draw_ellipse(annotated_frame, bbox, color=(255, 0, 0))

            if tracks['ball'][frame_num]:
                ball_data = tracks['ball'][frame_num][1]
                bbox = ball_data['bbox']
                annotated_frame = self.draw_triangle(annotated_frame, bbox, color=(0, 0, 255))
            
            self.draw_team_ball_control(annotated_frame, frame_num, team_ball_control)

            output.append(annotated_frame)
        return output
