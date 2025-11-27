import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

def curvature_from_three_points(p_prev, p_current, p_next):
    x1, y1 = p_prev
    x2, y2 = p_current
    x3, y3 = p_next

    dx21 = x2 - x1
    dy21 = y2 - y1
    dx32 = x3 - x2
    dy32 = y3 - y2
    dx31 = x3 - x1
    dy31 = y3 - y1


    a = np.hypot(dx21, dy21)
    b = np.hypot(dx32, dy32)
    if a < 1e-5 or b < 1e-5:
        return 0.0

    c = np.hypot(dx31, dy31)
        
    area2 = abs(dx21 * dy31 - dy21 * dx31)
    
    return 2 * area2 / (a * b * c)
        

def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:

    #INPUT
    #state: [xpos, ypos, steering angle, speed, heading]
    #desired: [steer angle, velocity]
    #parameters: car parameters

    #OUTPUT
    #[input steering rate, input acceleration]

    
    current_steer = state[2]  
    current_velocity = state[3]  
    
    desired_steer = desired[0] 
    desired_velocity = desired[1] 

    if not hasattr(lower_controller, 'prev_steer_error'):
        lower_controller.prev_steer_error = 0.0
    
    # Compute steering error
    steer_error = desired_steer - current_steer
    steer_error_rate = (steer_error - lower_controller.prev_steer_error) * 10 # dt = 0.1
    
    # Steering
    Kp_steer = 2.0  
    Kd_steer = 0.5  
    
    # velocity
    Kp_velocity = 20  # velocity gain
    
    # Compute control inputs
    u_steer = Kp_steer * steer_error + Kd_steer * steer_error_rate
    u_accel = Kp_velocity * (desired_velocity - current_velocity)
    
    # Clip to limits (from parameters)
    u_steer = np.clip(u_steer, parameters[7], parameters[9])  # max steering velocity
    u_accel = np.clip(u_accel, parameters[8], parameters[10])  # max acceleration
    
    lower_controller.prev_steer_error = steer_error

    return np.array([u_steer, u_accel])

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    #INPUT:
    #state: [xpos, ypos, steering angle, speed, heading]
    #parameters: car parameters
    #racetrack: input racetrack

    #OUTPUT
    #desired: [steer angle, velocity]

    #parse state
    x, y, steer, v, yaw = state
    N = len(racetrack.centerline)

    # ----------------------------
    # 1. Find nearest waypoint
    # ----------------------------
    
    car_pos = state[0:2]
    dists_sq = np.sum((racetrack.centerline - car_pos)**2, axis=1)
    idx = np.argmin(dists_sq)


    # ----------------------------
    # 2. Adaptive lookahead based on upcoming curvature
    # ----------------------------

    upcoming_curv = racetrack.curvatures[(idx+2) % N]
    
    if upcoming_curv > 0.03:  # Sharp turn threshold
        idx_next = (idx + 3) % N
    else:  # Gentle turn 
        idx_next = (idx + 2) % N

    p_current = racetrack.centerline[idx]
    p_next = racetrack.centerline[idx_next]

    # ----------------------------
    # 3. Path tracking errors
    # ----------------------------
    # Path direction vector
    path_dx = p_next[0] - p_current[0]
    path_dy = p_next[1] - p_current[1]
    path_yaw = np.arctan2(path_dy, path_dx)

    # Vector from car to closest path point
    map_dx = p_current[0] - x
    map_dy = p_current[1] - y

    # Cross-track error sign using cross-product
    cte = np.sin(path_yaw - np.arctan2(map_dy, map_dx)) * np.sqrt(dists_sq[idx])

    heading_error = path_yaw - yaw
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))  # normalize

    # ----------------------------
    # 4. Stanley steering control 
    # ----------------------------
    k = 0.5
    v_eps = max(v, 0.1) # To avoid division by zero
    cross_term = np.arctan2(k * cte, v_eps)

    #clip
    steer_cmd = np.clip(heading_error-cross_term, parameters[1], parameters[4])

    # ----------------------------
    # 5. Speed control 
    # ----------------------------

    # Look ahead further to anticipate upcoming turns
    lookahead_count = int(v * 0.5)+1  # Increase this to look further ahead
    lookback_count = 5
    
    # Collect curvatures over the lookahead window
    
    lookahead_indices = (idx + np.arange(-lookback_count, lookahead_count)) % N
    curvatures = racetrack.curvatures[lookahead_indices]
    
    
    curvatures_pow = curvatures ** 5
    weights = np.exp(-0.1 * np.arange(len(curvatures)))  # Exponential decay
    max_curv = (np.average(curvatures_pow, weights=weights))**0.2

    # Compute cornering speed limit based on maximum upcoming curvature
    a_lat_max = parameters[10]
    v_corner = np.sqrt(a_lat_max / (max(max_curv, 1e-6)))
    v_target = min(100, v_corner)
    
    decel_rate = 0.8  
    if v_target < v:
        vel_cmd = v + decel_rate * (v_target - v)
    else:
        vel_cmd = v_target

    return np.array([steer_cmd, vel_cmd])

    