from pid import pid_controller
import matplotlib.pyplot as plt

target = 10.0
dt = 0.05
value, integral, prev_error = 0.0, 0.0, 0.0
times, values = [], []

for step in range(1000):
    error = target - value
    output, integral = pid_controller(kp=1.0, ki=0.1, kd=0.05, dt=dt,
                                      error=error, prev_error=prev_error,
                                      integral=integral)
    prev_error = error            # must be set by you after every call
    value += output * dt          # the "plane" just moves by the output
    times.append(step * dt)
    values.append(value)

print("Final value (expect close to 10):", round(value, 3))

plt.plot(times, values, label="value")
plt.axhline(target, color="gray", linestyle="--", label="target")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.show()