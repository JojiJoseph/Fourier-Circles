import cv2
import numpy as np
from collections import deque
import argparse

parser = argparse.ArgumentParser(description="Animate Fourier Circles")
parser.add_argument(
    "-N", type=int, default=None, help="Number of circles to approximate the curve"
)

args = parser.parse_args()
N = args.N  # Number of circles to approximate the curve. TODO

img = np.zeros((512, 512, 3), np.uint8)  # Create a black image
cv2.namedWindow("Draw a closed curve")  # Create a window
points = []  # List of points

# state variables
pressed: bool = False  # Mouse button pressed
start_animation: bool = False  # Start animation


# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global pressed, start_animation
    if start_animation:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        if not pressed:
            points.append((x, y))
            pressed = True
    if event == cv2.EVENT_MOUSEMOVE:
        if pressed:
            points.append((x, y))
    if event == cv2.EVENT_LBUTTONUP:
        pressed = False
        start_animation = True


cv2.setMouseCallback("Draw a closed curve", mouse_callback)  # Set mouse callback

while True:
    if not start_animation:
        for point in points:
            cv2.circle(img, point, 2, (0, 255, 0), -1)
    cv2.imshow("Draw a closed curve", img)
    cv2.waitKey(1)
    if start_animation:
        break


target_curve = np.array([x - 256 + (y - 256) * 1j for x, y in points])
if N is None:
    N = len(target_curve)
last_queue: deque = deque(
    maxlen=int(N * 0.75)
)  # Last queue to draw the reconstructed curve
input_to_fourier = np.interp(
    np.linspace(0, 1, N), np.linspace(0, 1, len(target_curve)), target_curve
)

# Fourier transform
fourier = np.fft.fft(input_to_fourier) / N


t = 0
writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (512, 512))


def draw_complex_curve(img, curve, color):
    for xy in curve:
        x = xy.real + 256
        y = xy.imag + 256
        cv2.circle(img, (int(x), int(y)), 2, color, -1)


def draw_curve(img, curve, color):
    for x, y in curve:
        cv2.circle(img, (int(x), int(y)), 2, color, -1)


while True:
    img = np.zeros((512, 512, 3), np.uint8)  # Create a black image
    # Draw info text
    cv2.putText(
        img, f"N = {N}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
    cv2.putText(img, f"Target", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(
        img, f"Reconstructed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
    )

    draw_complex_curve(img, target_curve, (0, 255, 0))
    draw_curve(img, last_queue, (0, 0, 255))
    # Draw circles
    x, y = 256, 256
    for i in range(N):
        radius, phase = np.abs(fourier[i]), np.angle(fourier[i])
        angular_frequency = 2 * np.pi * i / N
        n = t
        next_x = x + radius * np.cos(angular_frequency * n + phase)
        next_y = y + radius * np.sin(angular_frequency * n + phase)
        cv2.circle(img, np.int0((x, y)), int(radius), (255, 255, 255), 1)
        cv2.arrowedLine(
            img, np.int0((x, y)), np.int0((next_x, next_y)), (0, 255, 255), 1
        )
        x = next_x
        y = next_y
    last_queue.append((x, y))
    cv2.imshow("Output", img)
    writer.write(img)
    t = (t + 1) % N
    key = cv2.waitKey(30) & 0xFF
    if key in [ord("q"), 27]:
        break

writer.release()
