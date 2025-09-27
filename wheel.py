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
        self.knob_radius = 120
        self.knob_angle = 0
        self.last_finger_angle = None
        
        # Stability system
        self.gesture_stability_count = 0
        self.required_stability = 3
    
    def is_finger_extended(self, landmarks, tip_id, pip_id):
        """Check if a finger is extended by comparing tip and PIP joint positions"""
        return landmarks[tip_id].y < landmarks[pip_id].y
    
    def detect_three_finger_gesture(self, landmarks):
        """Detect if thumb, index, and middle fingers are extended while ring and pinky are folded"""
        # More lenient thumb detection - just check if it's somewhat away from palm
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]
        
        # Much more relaxed thumb detection - just needs to be slightly extended
        thumb_extended = abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x) * 0.8
        
        # More relaxed finger extension detection
        index_extended = self.is_finger_extended(landmarks, 8, 6)
        middle_extended = self.is_finger_extended(landmarks, 12, 10)
        
        # Much more lenient folded finger detection - allow some extension
        ring_folded = landmarks[16].y > landmarks[14].y - 0.02
        pinky_folded = landmarks[20].y > landmarks[18].y - 0.02
        
        # Main requirement: thumb, index, and middle should be clearly extended
        primary_fingers = thumb_extended and index_extended and middle_extended
        secondary_fingers = ring_folded and pinky_folded
        
        return primary_fingers and secondary_fingers
    
    def get_hand_center(self, landmarks):
        """Get the center point of the hand"""
        return landmarks[9]
    
    def calculate_finger_angle(self, landmarks):
        """Calculate angle of index finger relative to hand center"""
        hand_center = self.get_hand_center(landmarks)
        index_tip = landmarks[8]
        
        dx = index_tip.x - hand_center.x
        dy = index_tip.y - hand_center.y
        angle = math.atan2(dy, dx)
        return angle
    
    def update_knob_angle(self, current_angle):
        """Update knob rotation based on finger movement"""
        if self.last_finger_angle is not None:
            angle_diff = current_angle - self.last_finger_angle
            
            # Handle angle wrap-around
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            elif angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            self.knob_angle += angle_diff * 2
            self.knob_angle = self.knob_angle % (2 * math.pi)
        
        self.last_finger_angle = current_angle
    
    def draw_knob(self, frame):
        """Draw the virtual knob/wheel"""
        if not self.knob_active or self.knob_center is None:
            return
        
        frame_h, frame_w = frame.shape[:2]
        center_x = int(self.knob_center.x * frame_w) - 60
        center_y = int(self.knob_center.y * frame_h)
        
        # Keep knob within frame boundaries
        margin = self.knob_radius + 20
        center_x = max(margin, min(center_x, frame_w - margin))
        center_y = max(margin, min(center_y, frame_h - margin))
        
        # Draw outer circle
        cv2.circle(frame, (center_x, center_y), self.knob_radius, (100, 100, 100), 3)
        cv2.circle(frame, (center_x, center_y), self.knob_radius - 15, (200, 200, 200), 2)
        
        # Draw tick marks
        for i in range(12):
            tick_angle = i * (2 * math.pi / 12)
            outer_x = center_x + int((self.knob_radius - 8) * math.cos(tick_angle))
            outer_y = center_y + int((self.knob_radius - 8) * math.sin(tick_angle))
            inner_x = center_x + int((self.knob_radius - 22) * math.cos(tick_angle))
            inner_y = center_y + int((self.knob_radius - 22) * math.sin(tick_angle))
            cv2.line(frame, (outer_x, outer_y), (inner_x, inner_y), (150, 150, 150), 2)
        
        # Draw pointer
        pointer_length = self.knob_radius - 30
        pointer_x = center_x + int(pointer_length * math.cos(self.knob_angle))
        pointer_y = center_y + int(pointer_length * math.sin(self.knob_angle))
        
        cv2.line(frame, (center_x, center_y), (pointer_x, pointer_y), (0, 0, 255), 5)
        cv2.circle(frame, (pointer_x, pointer_y), 10, (0, 0, 255), -1)
        cv2.circle(frame, (center_x, center_y), 8, (0, 255, 0), -1)
        
        # Display angle
        angle_degrees = int(math.degrees(self.knob_angle) % 360)
        cv2.putText(frame, f"Angle: {angle_degrees}°", (center_x - 50, center_y + self.knob_radius + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def draw_info(self, frame):
        """Draw FPS and gesture information"""
        cv2.putText(frame, f"FPS: {self.current_fps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.three_finger_gesture:
            cv2.putText(frame, "KNOB ACTIVE!", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
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
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            self.three_finger_gesture = False
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    gesture_detected = self.detect_three_finger_gesture(hand_landmarks.landmark)
                    
                    if gesture_detected:
                        self.gesture_stability_count = min(self.gesture_stability_count + 1, self.required_stability)
                        
                        if self.gesture_stability_count >= self.required_stability:
                            self.three_finger_gesture = True
                            self.knob_active = True
                            self.knob_center = self.get_hand_center(hand_landmarks.landmark)
                            
                            finger_angle = self.calculate_finger_angle(hand_landmarks.landmark)
                            self.update_knob_angle(finger_angle)
                            
                            if self.fps_counter % 10 == 0:
                                print(f"Knob active - Angle: {int(math.degrees(self.knob_angle) % 360)}°")
                    else:
                        self.gesture_stability_count = max(self.gesture_stability_count - 1, 0)
                        
                        if self.gesture_stability_count <= 0:
                            if self.knob_active:
                                self.knob_active = False
                                self.last_finger_angle = None
                                print("Knob deactivated")
            else:
                self.gesture_stability_count = max(self.gesture_stability_count - 1, 0)
                if self.gesture_stability_count <= 0 and self.knob_active:
                    self.knob_active = False
                    self.last_finger_angle = None
                    print("Knob deactivated - no hands")
            
            self.draw_knob(frame)
            self.draw_info(frame)
            
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            cv2.imshow('Hand Tracker', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = HandTracker()
    tracker.run()
