from utils.video_utils import read_video, write_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movment_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator 

def main():
    video_path = 'D:\\football_analysis\\videos\\08fd33_4.mp4'
    output_path = 'D:\\football_analysis\\output_videos\\output.avi'

    frames = read_video(video_path)
    tracker = Tracker(model_path='models/best.pt')
    tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path='stubs/tracks.pkl')
    # get object positions
    tracker.add_positions_to_tracks(tracks)
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    # estimate speed and distance
    speed_estimator = SpeedAndDistanceEstimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)
    # camera movement estimation
    camera_estimator = CameraMovementEstimator(frames[0])
    camera_movment_per_frame = camera_estimator.get_camera_movement(frames,
                                                                    read_from_stub=True,
                                                                    stub_path='stubs/camera_movement.pkl')
    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movment_per_frame)
    # view transformation
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):

        for player_id, player_data in player_track.items():
            bbox = player_data['bbox']
            team_id = team_assigner.get_player_team(frames[frame_num], bbox, player_id)

            tracks['players'][frame_num][player_id]['team_id'] = team_id
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team_id]
    # assign ball to player
    player_assigner = PlayerBallAssigner()
    team_ball_control = [None]
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', [])
        player_id = player_assigner.assign_ball(player_track, ball_bbox)
        
        if player_id is not None:
            tracks['players'][frame_num][player_id]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][player_id]['team_id'])
        else:
            team_ball_control.append(team_ball_control[-1])

        team_ball_control.append(player_id)

    # draw camera movement
    
    output = tracker.draw_annotations(frames, tracks, np.array(team_ball_control))
    output = camera_estimator.draw_camera_movement(output, camera_movment_per_frame)
    speed_estimator.draw_speed_and_distance(output, tracks)
    write_video(output, output_path)

if __name__ == '__main__':
    main()