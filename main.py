import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui

pyautogui.FAILSAFE = False

class SimpleGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
            model_complexity=0
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Mode toggle
        self.pinch_mode = False
        self.last_mode_toggle = 0
        self.mode_switch_timeout = 3.0  # 3 second timeout after mode switch
        
        # Cursor for pinch mode
        self.cursor_x = 320
        self.cursor_y = 240
        self.is_pinching = False
        
        # FPS
        self.fps_counter = 0
        self.fps_start = time.time()
        self.current_fps = 0
        
    def finger_up(self, landmarks, tip, pip):
        return landmarks[tip].y < landmarks[pip].y
    
    def count_fingers(self, landmarks):
        fingers = [
            self.finger_up(landmarks, 4, 3),   # thumb
            self.finger_up(landmarks, 8, 6),   # index
            self.finger_up(landmarks, 12, 10), # middle
            self.finger_up(landmarks, 16, 14), # ring
            self.finger_up(landmarks, 20, 18)  # pinky
        ]
        return sum(fingers)
    
    def detect_pinch(self, landmarks):
        thumb = landmarks[4]
        index = landmarks[8]
        distance = ((thumb.x - index.x)**2 + (thumb.y - index.y)**2)**0.5
        return distance < 0.05
    
    def detect_a_ok(self, landmarks):
        """Detect A-OK gesture (thumb tip touching index tip, other fingers extended)"""
        # Get thumb tip (4) and index finger tip (8)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate distance between thumb and index tips
        distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        
        # Check if thumb and index are close (forming the "O")
        if distance < 0.08:  # Relaxed threshold
            # Check if other fingers are extended
            middle_up = self.finger_up(landmarks, 12, 10)
            ring_up = self.finger_up(landmarks, 16, 14)
            pinky_up = self.finger_up(landmarks, 20, 18)
            
            # A-OK requires at least 1 of the other 3 fingers to be up (very relaxed)
            other_fingers_count = sum([middle_up, ring_up, pinky_up])
            return other_fingers_count >= 1
        
        return False
    
    def get_cursor_pos(self, landmarks, hand_label):
        # Get thumb tip (landmark 4) and index finger tip (landmark 8)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate midpoint between thumb and index finger
        mid_x = (thumb_tip.x + index_tip.x) / 2
        mid_y = (thumb_tip.y + index_tip.y) / 2
        
        # For left hand, adjust coordinates since thumb/index positions are mirrored
        if hand_label == "Left":
            mid_x = 1 - mid_x
        
        # Map to frame size (640x480)
        x = int(mid_x * 640)
        y = int(mid_y * 480)
        
        return max(0, min(x, 639)), max(0, min(y, 479))
    
    def check_mode_toggle(self, all_hands):
        # Always print hand count for debugging
        print(f"Hand count: {len(all_hands)}")
        
        if len(all_hands) != 2:
            return
        
        left_hand = None
        right_hand = None
        
        for landmarks, label in all_hands:
            print(f"Processing {label} hand")
            if label == "Left":
                left_hand = landmarks
            else:
                right_hand = landmarks
        
        if not left_hand or not right_hand:
            print("Missing left or right hand data")
            return
        
        # Check: both hands A-OK
        left_a_ok = self.detect_a_ok(left_hand)
        right_a_ok = self.detect_a_ok(right_hand)
        
        # Always print A-OK status
        print(f"A-OK Status: Left={left_a_ok}, Right={right_a_ok}")
        
        if left_a_ok and right_a_ok:
            now = time.time()
            print(f"*** BOTH HANDS A-OK! Time since last: {now - self.last_mode_toggle:.1f}s ***")
            if now - self.last_mode_toggle > 1.5:  # Reduced cooldown for easier testing
                self.pinch_mode = not self.pinch_mode
                self.last_mode_toggle = now
                mode = "PINCH" if self.pinch_mode else "GESTURE"
                print(f"**** MODE SWITCHED TO {mode} ****")
            else:
                print("Still in cooldown period")
    
    def open_claude_chat(self):
        try:
            import subprocess
            subprocess.run(['open', '-a', 'Google Chrome', 'https://claude.ai/chat/a592e8ae-110c-49b9-b4e1-52947ba91351'], check=True)
            print("Opened Claude chat in Chrome")
        except:
            try:
                script = '''tell application "Google Chrome"
                    activate
                    open location "https://claude.ai/chat/a592e8ae-110c-49b9-b4e1-52947ba91351"
                end tell'''
                subprocess.run(['osascript', '-e', script], check=True)
                print("Opened Claude chat via AppleScript")
            except:
                print("Failed to open Claude chat")
    
    def open_spotify(self):
        try:
            import subprocess
            subprocess.run(['open', '-a', 'Spotify'], check=True)
            print("Opened Spotify app")
        except:
            try:
                subprocess.run(['open', '-a', 'Google Chrome', 'https://open.spotify.com'], check=True)
                print("Opened Spotify web in Chrome")
            except:
                print("Failed to open Spotify")
    
    def open_browser(self):
        try:
            import subprocess
            subprocess.run(['open', '-a', 'Google Chrome'], check=True)
            print("Opened Chrome")
        except:
            print("Failed to open Chrome")
    
    def close_tab(self):
        try:
            import subprocess
            subprocess.run(['osascript', '-e', 'tell application "Google Chrome" to close active tab of front window'], check=True)
            print("Closed tab")
        except:
            pyautogui.hotkey('cmd', 'w')
            print("Sent Cmd+W")
    
    def take_screenshot(self):
        try:
            pyautogui.hotkey('cmd', 'shift', '3')
            print("Screenshot taken")
        except:
            print("Screenshot failed")
    
    def trigger_mission_control(self):
        try:
            pyautogui.hotkey('ctrl', 'up')
            print("Mission Control")
        except:
            print("Mission Control failed")
    
    def detect_gestures(self, landmarks):
        fingers = [
            self.finger_up(landmarks, 4, 3),   # thumb
            self.finger_up(landmarks, 8, 6),   # index
            self.finger_up(landmarks, 12, 10), # middle
            self.finger_up(landmarks, 16, 14), # ring
            self.finger_up(landmarks, 20, 18)  # pinky
        ]
        finger_count = sum(fingers)
        
        gestures = []
        
        # A-OK gesture
        if self.detect_a_ok(landmarks):
            gestures.append('A-OK')
        
        # Pinky up: only pinky (and optionally thumb)
        elif fingers[4] and not fingers[1] and not fingers[2] and not fingers[3]:
            if finger_count == 1 or (finger_count == 2 and fingers[0]):
                gestures.append('Pinky Up')
        
        # Hang Loose: thumb + pinky only
        elif fingers[0] and fingers[4] and not fingers[1] and not fingers[2] and not fingers[3]:
            if finger_count == 2:
                gestures.append('Hang Loose')
        
        # Upside down peace: ring + pinky
        elif fingers[3] and fingers[4] and not fingers[1] and not fingers[2]:
            gestures.append('Upside Down Peace')
        
        # Three fingers: index + middle + ring
        elif fingers[1] and fingers[2] and fingers[3] and not fingers[4]:
            gestures.append('Three Fingers')
        
        # Point: only index (and optionally thumb)
        elif fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
            if finger_count == 1 or (finger_count == 2 and fingers[0]):
                gestures.append('Point')
        
        # Peace: index + middle only
        elif fingers[1] and fingers[2] and not fingers[0] and not fingers[3] and not fingers[4]:
            if finger_count == 2:
                gestures.append('Peace')
        
        # Open hand
        elif finger_count >= 4:
            gestures.append('Open Hand')
        
        # Fist
        elif finger_count == 0:
            gestures.append('Fist')
        
        return gestures
    
    def execute_gesture(self, gesture):
        # Check if we're in timeout period after mode switch
        if time.time() - self.last_mode_toggle < self.mode_switch_timeout:
            return  # Ignore gestures during timeout
        
        if gesture == "Point":
            self.open_claude_chat()
        elif gesture == "A-OK":
            self.open_spotify()
        elif gesture == "Three Fingers":
            self.open_browser()
        elif gesture == "Pinky Up":
            self.close_tab()
        elif gesture == "Hang Loose":
            self.take_screenshot()
        elif gesture == "Upside Down Peace":
            self.trigger_mission_control()
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Camera error")
            return
        
        print("Simple Gesture Detector")
        print("Gesture Mode:")
        print("  Point (index finger) = Claude Chat")
        print("  A-OK (thumb+index circle) = Spotify")
        print("  Three Fingers = Chrome Browser")
        print("  Pinky Up = Close Tab")
        print("  Hang Loose (thumb+pinky) = Screenshot")
        print("  Upside Down Peace (ring+pinky) = Mission Control")
        print("")
        print("Mode Toggle: Both hands A-OK = Switch to Pinch Mode")
        print("Press q to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # FPS
            self.fps_counter += 1
            if time.time() - self.fps_start >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start = time.time()
            
            all_hands = []
            self.is_pinching = False
            
            if results.multi_hand_landmarks and results.multi_handedness:
                # Collect hand data
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    all_hands.append((hand_landmarks.landmark, label))
                
                # Check for mode toggle
                self.check_mode_toggle(all_hands)
                
                # Process based on mode
                if self.pinch_mode:
                    # PINCH MODE
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        label = handedness.classification[0].label
                        
                        # Update cursor with proper midpoint calculation
                        self.cursor_x, self.cursor_y = self.get_cursor_pos(hand_landmarks.landmark, label)
                        
                        # Check pinch
                        if self.detect_pinch(hand_landmarks.landmark):
                            self.is_pinching = True
                            print(f"PINCH at ({self.cursor_x}, {self.cursor_y})")
                        
                        # Draw simple landmarks
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(100, 100, 255), thickness=1),
                            self.mp_drawing.DrawingSpec(color=(100, 100, 255), thickness=1)
                        )
                else:
                    # GESTURE MODE
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        # Detect all gestures
                        gestures = self.detect_gestures(hand_landmarks.landmark)
                        
                        for gesture in gestures:
                            print(f"Detected: {gesture}")
                            self.execute_gesture(gesture)
                            time.sleep(0.5)  # Simple cooldown
                        
                        # Draw full landmarks
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
                            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                        )
            
            # Draw UI
            mode_text = "PINCH MODE" if self.pinch_mode else "GESTURE MODE"
            mode_color = (0, 0, 255) if self.pinch_mode else (0, 255, 0)
            
            cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
            cv2.putText(frame, f"FPS: {self.current_fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if self.pinch_mode:
                # Draw cursor
                color = (0, 0, 255) if self.is_pinching else (0, 255, 0)
                cv2.circle(frame, (self.cursor_x, self.cursor_y), 15, color, -1)
                cv2.putText(frame, f"({self.cursor_x}, {self.cursor_y})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Gesture Detector', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = SimpleGestureDetector()
    detector.run()
