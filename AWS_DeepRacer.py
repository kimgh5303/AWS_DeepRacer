import math
import numpy as np

# REWARD 1.0 ~ 1e-3
MAX_REWARD = 1.0
MIN_REWARD = 1e-3

# NEXT_DIRECTION 1.0 ~ 0
NEXT_DIRECTION = 0

def set_reward(score_table, value):
    if value <= score_table['stds'][0] or score_table['stds'][-1] <= value:
        return MIN_REWARD

    cnt = 0
    for i in score_table['stds']:
        if value >= i:
            cnt += 1
        else:
            if score_table['rewards'][cnt] == score_table['rewards'][cnt - 1]:
                return score_table['rewards'][cnt]
                
            grad = 1.0 * (score_table['rewards'][cnt] - score_table['rewards'][cnt - 1]) / (score_table['stds'][cnt] - score_table['stds'][cnt - 1])
            bias = score_table['rewards'][cnt] - grad * score_table['stds'][cnt]

            return grad * value + bias

def check_ontrack(params):
    '''모든 휠이 트랙 위에 있는가?'''
    return MAX_REWARD if params['all_wheels_on_track'] else MIN_REWARD

my_way_score_table = {}
my_way_score_table['stds'] = [0, 10, 30, 180]
my_way_score_table['rewards'] = [MAX_REWARD, MAX_REWARD, MAX_REWARD / 2, MIN_REWARD]

def check_my_way(params):
    '''차량이 포인터를 따라 가는가?'''
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']

    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]
    
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    track_direction = math.degrees(track_direction)
    
    direction_diff = abs(track_direction - heading)
    direction_diff = 360 - direction_diff if direction_diff > 180 else direction_diff
    NEXT_DIRECTION = direction_diff

    return set_reward(my_way_score_table, direction_diff)

speed_score_table = {}

def check_speed(params):
    if 30 < NEXT_DIRECTION:
        speed_score_table['stds'] = [0, 1, 2, 3, 4]
        speed_score_table['rewards'] = [MAX_REWARD * 0.8, MAX_REWARD * 0.8, MAX_REWARD / 2, MIN_REWARD, MIN_REWARD]
    else:
        speed_score_table['stds'] = [0, 1, 2, 3, 4]
        speed_score_table['rewards'] = [MIN_REWARD, MIN_REWARD, MAX_REWARD / 2, MAX_REWARD, MAX_REWARD]

    return set_reward(speed_score_table, params['speed'])

steering_score_table = {}

def check_steering(params):
    steering_score_table['stds'] = [0, 5, 15, 30]
    steering_score_table['rewards'] = [MAX_REWARD, MAX_REWARD, MAX_REWARD / 2, MIN_REWARD]

    return set_reward(steering_score_table, params['steering_angle'])

center_table = {}

def check_center(params):
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']

    center_table['stds'] = np.array([0, 0.1, 0.5, 1]) * track_width
    center_table['rewards'] = [MAX_REWARD, MAX_REWARD, MAX_REWARD / 2, MIN_REWARD]

    return set_reward(center_table, distance_from_center)

def reward_function(params):
    reward = 0

    reward += check_ontrack(params)
    reward += check_my_way(params)
    reward += check_speed(params)
    reward += check_steering(params)
    reward += check_center(params)

    return reward
