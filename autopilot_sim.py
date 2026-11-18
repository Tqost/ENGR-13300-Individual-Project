import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
import numpy as np
from pid import pid_controller


def get_valid_input(prompt, low, high):
    while True:
        try: 
            value = float(input(prompt))
            if low <= value <= high:
                return value
            print(f"Value is not within low and high bounds, please try again")
        except ValueError: 
            print(f"Error: Please enter a number between {low} and {high}")


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def update_dynamics(
        state, 
        elevator, 
        bank, 
        vert_gust, 
        wind_x, 
        wind_y, 
        airspeed, 
        dt
        ):
    x, y, altitude, pitch, pitch_rate, heading = state
    pitch_accel = -2.0 * pitch_rate - 4.0 * pitch + 6.0 * elevator + vert_gust
    pitch_rate += pitch_accel * dt
    pitch += pitch_rate * dt
    altitude += airspeed * np.sin(pitch) * dt

    heading += (9.81 / airspeed) * np.tan(bank) * dt
    heading = wrap_angle(heading)

    ground_speed = airspeed * np.cos(pitch)
    x += (ground_speed * np.cos(heading) + wind_x) * dt
    y += (ground_speed * np.sin(heading) + wind_y) * dt

    return np.array([x, y, altitude, pitch, pitch_rate, heading])


def fly(
        state, 
        seconds, 
        dt=0.05, 
        elevator=0.0, 
        bank=0.0, 
        wind_x=0.0, 
        wind_y=0.0, 
        airspeed=20.0
        ):
    steps = int(seconds / dt)
    history = np.zeros((steps, 6))
    for i in range(steps):
        state = update_dynamics(
            state, 
            elevator, 
            bank, 
            0.0, 
            wind_x, 
            wind_y, 
            airspeed, 
            dt
            )
        history[i] = state
    return history


def simulate_landing(start_state, runway_heading, airspeed, dt=0.05, max_time=120.0):
    state = start_state.copy()
    max_steps = int(max_time / dt)
    history = np.zeros((max_steps, 6))

    alt_integral, alt_prev_error = 0.0, 0.0
    alt_kp, alt_ki, alt_kd = 0.05, 0.002, 0.1
    max_elevator = np.radians(25)
    glide_slope = np.tan(np.radians(3))

    steps_flown = 0
    for step in range(max_steps):
        x, y, altitude = state[0], state[1], state[2]

        along = x * np.cos(runway_heading) + y * np.sin(runway_heading)

        target_alt = glide_slope * max(-along, 0.0)
        alt_error = target_alt - altitude

        elevator, alt_integral = pid_controller(
            kp=alt_kp, 
            ki=alt_ki, 
            kd=alt_kd, 
            dt=dt,
            error=alt_error, 
            prev_error=alt_prev_error, 
            integral=alt_integral
        )
        alt_prev_error = alt_error

        if elevator > max_elevator:
            elevator = max_elevator
        elif elevator < -max_elevator:
            elevator = -max_elevator

        state = update_dynamics(state, elevator, 0.0, 0.0, 0.0, 0.0, airspeed, dt)
        history[step] = state
        steps_flown = step + 1

        if state[2] <= 0:
            break

    return history[:steps_flown]

    
def animate_landing(
        history, 
        rwy_heading=0.0, 
        rwy_length=300.0, 
        rwy_width=20.0, 
        dt=0.05, 
        skip=2
        ):
    x, y, alt = history[:, 0], history[:, 1], history[:, 2]
    c, s = np.cos(rwy_heading), np.sin(rwy_heading)
    along = x * c + y * s

    corners = [(0, -rwy_width / 2), (rwy_length, -rwy_width / 2),
               (rwy_length, rwy_width / 2), (0, rwy_width / 2)]
    runway_pts = [(a * c - b * s, a * s + b * c) for a, b in corners]

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(12, 5))

    ax_top.add_patch(Polygon(runway_pts, color="gray"))
    ax_top.set_xlim(min(x.min(), -rwy_length) - 50, max(x.max(), rwy_length) + 50)
    ax_top.set_ylim(min(y.min(), -rwy_length) - 50, max(y.max(), rwy_length) + 50)
    ax_top.set_aspect("equal")
    ax_top.set_title("Top view")
    top_trail, = ax_top.plot([], [], "b-", linewidth=1)
    top_plane, = ax_top.plot([], [], "ro")

    ax_side.axhline(0, color="gray")
    ax_side.set_xlim(min(along.min(), 0) - 50, max(along.max(), rwy_length) + 50)
    ax_side.set_ylim(-5, alt.max() + 10)
    ax_side.set_title("Side view")
    ax_side.set_xlabel("Distance along runway (m)")
    ax_side.set_ylabel("Altitude (m)")
    side_trail, = ax_side.plot([], [], "b-", linewidth=1)
    side_plane, = ax_side.plot([], [], "ro")

    def update(frame):
        top_trail.set_data(x[:frame], y[:frame])
        top_plane.set_data([x[frame]], [y[frame]])
        side_trail.set_data(along[:frame], alt[:frame])
        side_plane.set_data([along[frame]], [alt[frame]])
        fig.suptitle(f"t = {frame * dt:5.1f} s   alt = {alt[frame]:5.1f} m")
        return top_trail, top_plane, side_trail, side_plane

    anim = FuncAnimation(fig, update, frames=range(0, len(x), skip),
                         interval=dt * 1000 * skip, blit=False)

    plt.show()
    return anim

def main():
    # runway_heading = get_valid_input("Enter runway heading (0 - 359): ", 0, 359)
    # gust_strength = get_valid_input("Enter a gust strenght (0 - 10 mph): ", 0, 10)

    start = np.array([-600.0, 0.0, 60.0, 0.0, 0.0, 0.0])
    history = fly(state=start, seconds=40, elevator=-0.06, bank=0.0, wind_y=0.0)
    below_ground = np.where(history[:, 2] <= 0)[0]
    if len(below_ground) > 0:
        history = history[:below_ground[0] + 1]
    anim = animate_landing(history, rwy_heading=0.0, dt=0.05)


if __name__ == "__main__":
    main()