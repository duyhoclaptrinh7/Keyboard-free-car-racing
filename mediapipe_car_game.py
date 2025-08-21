import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import numpy as np
import math
from pynput.keyboard import Key, Controller
kb = Controller()

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None:
        x_offset = (bg_w - fg_w) // 2
    if y_offset is None:
        y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


wheel_image = cv2.imread("steering_wheel.png", cv2.IMREAD_UNCHANGED)
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width, _ = image.shape
    if results.multi_hand_landmarks:
      hand_centers = []
      # Đảm bảo tay phải luôn ở trước tay trái trong mảng hand_centers
      # tạo list tạm để lưu paired: [(label, center), ...]
      pairs = []

      if results.multi_hand_landmarks:
          for handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
              label = handedness.classification[0].label  # "Left" hoặc "Right"
              cx = int(hand_landmarks.landmark[9].x * width)
              cy = int(hand_landmarks.landmark[9].y * height)
              pairs.append((label, (cx, cy)))

      # sau đó sort hoặc ghép cố định Right trước Left
      for label, center in pairs:
          if label == "Right":
              hand_centers.append(center)
      for label, center in pairs:
          if label == "Left":
              hand_centers.append(center)
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
      if len(hand_centers) == 2:
        center_x = (hand_centers[0][0] + hand_centers[1][0]) // 2
        center_y = (hand_centers[0][1] + hand_centers[1][1]) // 2
        radius = int(math.sqrt((hand_centers[0][0] - hand_centers[1][0]) ** 2 + (hand_centers[0][1] - hand_centers[1][1]) ** 2) / 2)
        angle = math.degrees(math.atan2(hand_centers[1][1] - hand_centers[0][1], hand_centers[0][0] - hand_centers[1][0]))
        print(angle)
        add_transparent_image(image, cv2.resize(rotate_image(wheel_image, angle), (radius * 2, radius * 2)), center_x - radius, center_y - radius)
        # Điều khiển bàn phím dựa trên góc
        if angle > -9 and angle < 9:
          kb.release(Key.left)
          kb.release(Key.right)
          kb.release(Key.down)
          kb.press(Key.up)
        elif angle <= -9 and angle >= -12:
          kb.release(Key.right)
          kb.release(Key.down)
          kb.press(Key.up)
          kb.press(Key.left)
        elif angle < -12:
          kb.release(Key.right)
          kb.release(Key.up)
          kb.release(Key.down)
          kb.press(Key.left)
        elif angle >= 9 and angle <= 12:
          kb.release(Key.left)
          kb.release(Key.down)
          kb.press(Key.up)
          kb.press(Key.right)
        elif angle > 12:
          kb.release(Key.left)
          kb.release(Key.up)
          kb.release(Key.down)
          kb.press(Key.right)
      elif len(hand_centers) == 1:
        kb.release(Key.left)
        kb.release(Key.right)
        kb.release(Key.up)
        kb.release(Key.down)
      else:
        kb.release(Key.left)
        kb.release(Key.right)
        kb.release(Key.up)
        kb.press(Key.down)  
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()