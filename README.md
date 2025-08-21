# Keyboard-free-car-racing

**Webcam-based virtual steering wheel using MediaPipe & OpenCV to control car racing games without a keyboard.**

---

## Introduction

Keyboard-free-car-racing turns two-hand gestures captured by your webcam into steering input for car racing games. It overlays a virtual steering wheel on the camera feed, scales/rotates it based on the position of your hands, computes a steering angle, and emits arrow-key events so you can play keyboard-controlled racing games without touching the keyboard.


**Demo video**



```markdown
![demo.gif](docs/demo.gif)

or

[Watch demo on YouTube](https://youtube.com/your-demo-link)
```

Replace the link with your GIF or video URL.

---

## How to use my code

First, run the program:

```bash
python mediapipe_car_game.py
```

Then, open any keyboard-controlled car racing game and use your two hands in front of the webcam as a steering wheel. The program will send arrow key events to the OS, which most games will accept as input.

**Notes:**

* Keep the webcam facing you and ensure both hands are visible for the best experience.
* If the game requires focus, click on the game window after starting the script so it receives the emulated key events.

---

## Requirements

* **Python:** 3.8+ (recommended).
* **Python packages:** 

```
mediapipe
opencv-python
numpy
pynput
```
