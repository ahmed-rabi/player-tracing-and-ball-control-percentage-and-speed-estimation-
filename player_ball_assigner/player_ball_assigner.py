
import numpy as np

class PlayerBallAssigner:
    def __init__(self):
        self.max_player_distance = 70  # Maximum distance to consider a player as the ball holder
    def calculate_distance(self, bbox1, bbox2):
        """
        Calculate the Euclidean distance between the centers of two bounding boxes.
        Args:
            bbox1: Bounding box of the first object.
            bbox2: Bounding box of the second object.

        Returns:
            distance: Euclidean distance between the centers of the two bounding boxes.
        """
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        return np.linalg.norm(np.array(center1) - np.array(center2))
    def assign_ball(self, players, ball_bbox):
        """
        Assign the ball to the player closest to it.
        Args:
            players: List of player bounding boxes.
            ball_bbox: Bounding box of the ball.

        Returns:
            player_id: ID of the player holding the ball, or None if no player is close enough.
        """
        closest_player_id = None
        min_distance = float('inf')

        for player_id, player_bbox in players.items():
            distance = self.calculate_distance(player_bbox['bbox'], ball_bbox)
            if distance < self.max_player_distance and distance < min_distance:
                min_distance = distance
                closest_player_id = player_id

        return closest_player_id

    
        