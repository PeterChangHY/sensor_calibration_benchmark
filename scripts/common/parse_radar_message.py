#!/usr/bin/env python
def parse_radar_track(msg):
    radar_tracks = []

    for i in range(len(msg.tracks)):
        for p in msg.tracks[i].track_shape.points:
            radar_track = {'timestamp': msg.header.stamp.to_sec(),\
                       'track_id': msg.tracks[i].track_id,\
                       'velocity': [msg.tracks[i].linear_velocity.x, msg.tracks[i].linear_velocity.y, msg.tracks[i].linear_velocity.z],\
                       'acceleration': [ msg.tracks[i].linear_acceleration.x, msg.tracks[i].linear_acceleration.y, msg.tracks[i].linear_acceleration.z],\
                       'point': [ (p.x, p.y, p.z) for p in msg.tracks[i].track_shape.points]}
        radar_tracks.append(radar_track)
    
    return radar_tracks
