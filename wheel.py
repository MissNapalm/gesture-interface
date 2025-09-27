import cv2
import mediapipe as mp
import time
import numpy as np
import math

class HandTracker:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
            model_complexity=0
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Gesture detection
        self.three_finger_gesture = False
        
        # Knob/wheel properties
        self.knob_active = False
        self.knob_center = None
        self.knob_radius = 80
        self.knob_angle = 0
        self.last_finger_angle = None
    
    def is_finger_extended(self, landmarks, tip_id, pip_id):
        """Check if a finger is extended by comparing tip and PIP joint positions"""
        return landmarks[tip_id].y < landmarks[pip_id].y
    
    def detect_three_finger_gesture(self, landmarks):
        """Detect if thumb, index, and middle fingers are extended while ring and pinky are folded"""
        # More robust thumb detection - check if thumb tip is significantly away from palm
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]
        
        # Calculate if thumb is extended outward from the hand
        thumb_extended = abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x) * 1.2
        
        # Check if index finger is extended
        index_extended = self.is_finger_extended(landmarks, 8, 6)  # Index tip vs Index PIP
        
        # Check if middle finger is extended
        middle_extended = self.is_finger_extended(landmarks, 12, 10)  # Middle tip vs Middle PIP
        
        # Check if ring finger is folded
        ring_folded = not self.is_finger_extended(landmarks, 16, 14)  # Ring tip vs Ring PIP
        
        # Check if pinky is folded
        pinky_folded = not self.is_finger_extended(landmarks, 20, 18)  # Pinky tip vs Pinky PIP
        
        # ALL conditions must be true - this prevents peace sign from triggering
        return thumb_extended and index_extended and middle_extended and ring_folded and pinky_folded
    
    def get_hand_center(self, landmarks):
        """Get the center point of the hand"""
        # Use the middle of the hand (landmark 9 - middle finger MCP)
        return landmarks[9]
    
    def calculate_finger_angle(self, landmarks):
        """Calculate angle of index finger relative to hand center"""
        hand_center = self.get_hand_center(landmarks)
        index_tip = landmarks[8]
        
        # Calculate angle from hand center to index finger tip
        dx = index_tip.x - hand_center.x
        dy = index_tip.y - hand_center.y
        angle = math.atan2(dy, dx)
        return angle
    
    def update_knob_angle(self, current_angle):
        """Update knob rotation based on finger movement"""
        if self.last_finger_angle is not None:
            # Calculate angle difference
            angle_diff = current_angle - self.last_finger_angle
            
            # Handle angle wrap-around
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            elif angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # Update knob angle
            self.knob_angle += angle_diff * 2  # Multiply for more sensitivity
            
            # Keep angle in reasonable range
            self.knob_angle = self.knob_angle % (2 * math.pi)
        
        self.last_finger_angle = current_angle
    
    def draw_knob(self, frame):
        """Draw the virtual knob/wheel"""
        if not self.knob_active or self.knob_center is None:
            return
        
        frame_h, frame_w = frame.shape[:2]
        center_x = int(self.knob_center.x * frame_w)
        center_y = int(self.knob_center.y * frame_h)
        
        # Draw outer circle (knob body)
        cv2.circle(frame, (center_x, center_y), self.knob_radius, (100, 100, 100), 3)
        cv2.circle(frame, (center_x, center_y), self.knob_radius - 10, (200, 200, 200), 2)
        
        # Draw tick marks around the knob
        for i in range(12):  # 12 tick marks like a clock
            tick_angle = i * (2 * math.pi / 12)
            outer_x = center_x + int((self.knob_radius - 5) * math.cos(tick_angle))
            outer_y = center_y + int((self.knob_radius - 5) * math.sin(tick_angle))
            inner_x = center_x + int((self.knob_radius - 15) * math.cos(tick_angle))
            inner_y = center_y + int((self.knob_radius - 15) * math.sin(tick_angle))
            cv2.line(frame, (outer_x, outer_y), (inner_x, inner_y), (150, 150, 150), 2)
        
        # Draw knob pointer/indicator
        pointer_length = self.knob_radius - 20
        pointer_x = center_x + int(pointer_length * math.cos(self.knob_angle))
        pointer_y = center_y + int(pointer_length * math.sin(self.knob_angle))
        
        # Pointer line
        cv2.line(frame, (center_x, center_y), (pointer_x, pointer_y), (0, 0, 255), 4)
        
        # Pointer tip circle
        cv2.circle(frame, (pointer_x, pointer_y), 8, (0, 0, 255), -1)
        
        # Center dot
        cv2.circle(frame, (center_x, center_y), 6, (0, 255, 0), -1)
        
        # Display angle value
        angle_degrees = int(math.degrees(self.knob_angle) % 360)
        cv2.putText(frame, f"Angle: {angle_degrees}°", (center_x - 40, center_y + self.knob_radius + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    def draw_info(self, frame):
        """Draw FPS and gesture information"""
        cv2.putText(frame, f"FPS: {self.current_fps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show gesture detection status
        if self.three_finger_gesture:
            cv2.putText(frame, "KNOB ACTIVE!", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Extend thumb, index & middle fingers to create knob", (10, frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Move index finger to rotate knob - Press 'q' to quit", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Hand Tracker with Virtual Knob Started!")
        print("Extend thumb, index, and middle fingers to create a virtual knob")
        print("Move your index finger to rotate the knob")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Reset gesture state
            self.three_finger_gesture = False
            
            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks and connections
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    # Detect three-finger gesture
                    if self.detect_three_finger_gesture(hand_landmarks.landmark):
                        self.three_finger_gesture = True
                        self.knob_active = True
                        self.knob_center = self.get_hand_center(hand_landmarks.landmark)
                        
                        # Calculate and update knob rotation based on index finger position
                        finger_angle = self.calculate_finger_angle(hand_landmarks.landmark)
                        self.update_knob_angle(finger_angle)
                        
                        print(f"Knob active - Angle: {int(math.degrees(self.knob_angle) % 360)}°")
                    else:
                        # Reset knob when gesture is lost
                        if self.knob_active:
                            self.knob_active = False
                            self.last_finger_angle = None
                            print("Knob deactivated")
            
            # Draw knob if active
            self.draw_knob(frame)
            
            # Draw info overlay
            self.draw_info(frame)
            
            # Calculate FPS
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            # Show frame
            cv2.imshow('Hand Tracker', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = HandTracker()
    tracker.run()
