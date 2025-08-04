import cv2
import numpy as np

class ViewTransformer():
    def __init__(self):
        court_width = 68
        court_length = 23.32

        self.pixel_verticies = np.array([
            [110, 1035],
            [265,275],
            [910, 260],
            [1640, 915]
        ])

        self.target_verticies = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [0, court_length]
        ])

        self.pixel_verticies = self.pixel_verticies.astype(np.float32)
        self.target_verticies = self.target_verticies.astype(np.float32)

        self.prespective_transform_matrix = cv2.getPerspectiveTransform(self.pixel_verticies, self.target_verticies)
    
    def transform_point(self, point):
        p = int(point[0]), int(point[1])
        is_inside = cv2.pointPolygonTest(self.pixel_verticies, p, False) >= 0
        if not is_inside:
            return None
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.prespective_transform_matrix)
        
        return transformed_point.reshape(-1, 2)
    
    def add_transformed_position_to_tracks(self, tracks):
        for obj, obj_tracks in tracks.items():
            for frame_num, track in enumerate(obj_tracks):
                for track_id, track_data in track.items():
                    position = track_data['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze()
                    tracks[obj][frame_num][track_id]['position_transformed'] = position_transformed


                    