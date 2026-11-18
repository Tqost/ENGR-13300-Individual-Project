import numpy as np

def pid_controller(
        kp,
        ki, 
        kd, 
        dt, 
        error, 
        prev_error, 
        integral, 
        integral_limit=50.0
        ):
    integral += error * dt

    if integral > integral_limit: 
        integral = integral_limit
    elif integral < -integral_limit:
        integral = -integral_limit

    derivative = (error - prev_error) / dt
    output = kp * error + ki * integral + kd * derivative
    
    return output, integral
