import cv2
import numpy as np
from sklearn.cluster import KMeans
class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_teams_dict = {}

    def get_player_color(self, frame, bbox):
        """
        Get the color of a player based on their bounding box.
        Args:
            frame: The video frame.
            bbox: Bounding box of the player.

        Returns:
            color: The dominant color of the player.
        """
        x1, y1, x2, y2 = map(int, bbox)
        player_region = frame[y1:y2, x1:x2]
        top_half = player_region[:player_region.shape[0] // 2, :, :]
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(top_half.reshape(-1, 3))
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half.shape[0], top_half.shape[1])
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        dominant_color = kmeans.cluster_centers_[player_cluster]
        return dominant_color

    def assign_team_color(self, frame, player_detections):
        
        players_color = []
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            players_color.append(player_color)

        kmeans = KMeans(n_clusters=2)
        kmeans.fit(players_color)
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_teams_dict:
            return self.player_teams_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1
        self.player_teams_dict[player_id] = team_id
        return team_id