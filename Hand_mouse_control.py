import cv2
import mediapipe as mp
import pyautogui
import time
import math

class HandsMouseControl:

    def __init__(self, width_detection=300, height_detection=200, rect_x=150, rect_y=150, min_detection_confidence=0.5, min_tracking_confidence=0.5, capture_input=0, window_title='hand_detection') -> None:
        # Initialize detection parameters
        self.width_detection = width_detection
        self.height_detection = height_detection
        self.rect_x = rect_x
        self.rect_y = rect_y
        self.screen_width, self.screen_height = pyautogui.size()

        # Initialize Mediapipe modules
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Configure video capture
        self.capture_input = capture_input
        self.window_title = window_title

        # Configure confidence thresholds for hand detection and tracking
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Variables for mouse click tracking
        self.time_last_click_L = None
        self.time_last_click_R = None
        self.state_down = False

    def left_click(self):
        # Function to perform a left click
        if self.time_last_click_L is None:
            # If it's the first click, record the time and perform a click
            self.time_last_click_L = time.time()
            pyautogui.click()
        elif self.time_last_click_L >= 2 and not self.state_down:
            # If the time between clicks is greater than 2 seconds and the button is not already down, hold the click
            pyautogui.mouseDown()
            self.state_down = True
            print('down')

    def right_click(self):
        # Function to perform a right click
        if self.time_last_click_R is None:
            # If it's the first right click, record the time and perform a right click
            self.time_last_click_R = time.time()
            pyautogui.rightClick()

    def gesture_left_click(self, lm, shape):
        # Function to handle left click based on finger positions
        thumb_tip = lm.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        x_thumb, y_thumb = int(thumb_tip.x * shape[1]), int(thumb_tip.y * shape[0])

        index_finger = lm.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x_index, y_index = int(index_finger.x * shape[1]), int(index_finger.y * shape[0])

        # Calculate distance between thumb and index
        distance = math.sqrt((x_index - x_thumb) ** 2 + (y_index - y_thumb) ** 2)

        if distance <= 30:
            # If the distance is less than 30 pixels, perform a left click
            self.left_click()
        elif self.state_down:
            # If the button is down, release it
            pyautogui.mouseUp()
            self.state_down = False
            self.time_last_click_L = None
            print('up')

    def gesture_right_click(self, lm, shape):
        # Function to handle right click based on finger positions
        thumb_tip = lm.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        x_thumb, y_thumb = int(thumb_tip.x * shape[1]), int(thumb_tip.y * shape[0])

        middle_finger = lm.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        x_middle, y_middle = int(middle_finger.x * shape[1]), int(middle_finger.y * shape[0])

        # Calculate distance between thumb and middle finger
        distance = math.sqrt((x_middle - x_thumb) ** 2 + (y_middle - y_thumb) ** 2)
        print(distance)
        if distance <= 30:
            # If the distance is less than 30 pixels, perform a right click
            self.right_click()
        else:
            self.time_last_click_R = None

    def start(self, drawing=True, show=True):
        # Main function to start video capture and hand tracking
        capture = cv2.VideoCapture(self.capture_input)

        with self.mp_hands.Hands(min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence) as hands:
            while capture.isOpened():
                img, frame = capture.read()

                # Convert BGR image to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process frame for hand tracking
                result = hands.process(rgb_frame)

                # If there is a hand in the image
                if result.multi_hand_landmarks:
                    for lm in result.multi_hand_landmarks:
                        # Get index finger position
                        middle_landmark = lm.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                        x_pos, y_pos = int(middle_landmark.x * frame.shape[1]), int(middle_landmark.y * frame.shape[0])

                        # Get cursor position
                        cursor_x = int((x_pos - self.rect_x) / self.width_detection * pyautogui.size()[0])
                        cursor_y = int((y_pos - self.rect_y) / self.height_detection * pyautogui.size()[1])

                        # Move cursor if the position is valid
                        if 0 <= cursor_x < self.screen_width and 0 <= cursor_y < self.screen_height:
                            pyautogui.moveTo(self.screen_width - cursor_x, cursor_y, 0.15)

                        # Handle clicks based on gestures
                        self.gesture_left_click(lm, frame.shape)
                        self.gesture_right_click(lm, frame.shape)

                        # Display additional elements (drawings, information)
                        if drawing:
                            self.mp_drawing.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)
                            cv2.circle(frame, (x_pos, y_pos), 10, (255, 0, 0), -1)
                            cv2.putText(frame, f'x:{cursor_x} y:{cursor_y}', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1,
                                        (0, 0, 0), 2)
                            cv2.rectangle(frame, (self.rect_x, self.rect_y),
                                          (self.rect_x + self.width_detection, self.rect_y + self.height_detection),
                                          (0, 255, 0), 2)

                # Display resized frame
                if show:
                    frame_resize = cv2.resize(frame, (1200, 950))
                    cv2.imshow('Hand Tracking', frame_resize)

                # Exit loop by pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    control = HandsMouseControl()
    control.start()
