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
            max_num_hands=2,  # Changed to detect both hands
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
        self.knob_angle = math.pi  # Start at 180 degrees (middle position)
        self.last_finger_angle = None
        
        # Stability system
        self.gesture_stability_count = 0
        self.required_stability = 3
        
        # Timeline properties
        self.timeline_position = 0.5  # Position along timeline (0.0 to 1.0)
        self.timeline_height = 60
        self.timeline_margin = 20
        
        # Wheel mode properties
        self.wheel_mode = False
        self.touch_threshold = 50  # Distance threshold for touch detection in pixels
        self.wheel_persistent = False  # Once wheel appears, it stays until hands drop
    
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
            
            # Update timeline position based on knob angle
            # Map angle (0 to 2π) to timeline position (0.0 to 1.0)
            self.timeline_position = (self.knob_angle / (2 * math.pi)) % 1.0
        
        self.last_finger_angle = current_angle
    
    def draw_hand_numbers(self, frame, landmarks, hand_label):
        """Draw numbers on fingertips and knuckles"""
        frame_h, frame_w = frame.shape[:2]
        
        if hand_label == "Right":
            # Draw 1 and 2 on index and middle fingertips
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            
            # Index finger tip - "1"
            tip_x = int(index_tip.x * frame_w)
            tip_y = int(index_tip.y * frame_h)
            cv2.putText(frame, "1", (tip_x - 10, tip_y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
            cv2.putText(frame, "1", (tip_x - 10, tip_y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Middle finger tip - "2"
            tip_x = int(middle_tip.x * frame_w)
            tip_y = int(middle_tip.y * frame_h)
            cv2.putText(frame, "2", (tip_x - 10, tip_y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
            cv2.putText(frame, "2", (tip_x - 10, tip_y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        elif hand_label == "Left":
            # Draw 1, 2, 3, 4 on knuckles (MCP joints)
            knuckle_landmarks = [5, 9, 13, 17]  # Index, Middle, Ring, Pinky MCP joints
            knuckle_labels = ["1", "2", "3", "4"]
            
            for i, knuckle_id in enumerate(knuckle_landmarks):
                knuckle = landmarks[knuckle_id]
                knuckle_x = int(knuckle.x * frame_w)
                knuckle_y = int(knuckle.y * frame_h)
                
                cv2.putText(frame, knuckle_labels[i], (knuckle_x - 10, knuckle_y + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
                cv2.putText(frame, knuckle_labels[i], (knuckle_x - 10, knuckle_y + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    def detect_touch(self, right_landmarks, left_landmarks):
        """Detect if right hand fingertips (1 or 2) are touching left hand knuckles"""
        frame_h, frame_w = 640, 480  # Approximate frame dimensions for calculation
        
        # Right hand fingertips (index=8, middle=12)
        right_index_tip = (int(right_landmarks[8].x * frame_w), int(right_landmarks[8].y * frame_h))
        right_middle_tip = (int(right_landmarks[12].x * frame_w), int(right_landmarks[12].y * frame_h))
        
        # Left hand knuckles (MCP joints: 5, 9, 13, 17)
        left_knuckles = []
        for knuckle_id in [5, 9, 13, 17]:
            knuckle_pos = (int(left_landmarks[knuckle_id].x * frame_w), int(left_landmarks[knuckle_id].y * frame_h))
            left_knuckles.append(knuckle_pos)
        
        # Check if any fingertip is close to any knuckle
        for fingertip in [right_index_tip, right_middle_tip]:
            for knuckle in left_knuckles:
                distance = math.sqrt((fingertip[0] - knuckle[0])**2 + (fingertip[1] - knuckle[1])**2)
                if distance < self.touch_threshold:
                    return True
        
        return False
    
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
        
        # Display angle and timeline position
        angle_degrees = int(math.degrees(self.knob_angle) % 360)
        timeline_percent = int(self.timeline_position * 100)
        cv2.putText(frame, f"Angle: {angle_degrees}° | Timeline: {timeline_percent}%", 
                   (center_x - 80, center_y + self.knob_radius + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_timeline(self, frame):
        """Draw the timeline at the bottom of the frame"""
        frame_h, frame_w = frame.shape[:2]
        
        # Timeline dimensions
        timeline_y = frame_h - self.timeline_height - 10
        timeline_start_x = self.timeline_margin
        timeline_end_x = frame_w - self.timeline_margin
        timeline_width = timeline_end_x - timeline_start_x
        
        # Draw timeline background
        cv2.rectangle(frame, 
                     (timeline_start_x, timeline_y), 
                     (timeline_end_x, timeline_y + 30), 
                     (60, 60, 60), -1)
        
        # Draw timeline border
        cv2.rectangle(frame, 
                     (timeline_start_x, timeline_y), 
                     (timeline_end_x, timeline_y + 30), 
                     (150, 150, 150), 2)
        
        # Draw timeline tick marks
        num_ticks = 10
        for i in range(num_ticks + 1):
            tick_x = timeline_start_x + int(i * timeline_width / num_ticks)
            tick_height = 15 if i % 5 == 0 else 8  # Larger ticks every 5 marks
            cv2.line(frame, 
                    (tick_x, timeline_y + 30), 
                    (tick_x, timeline_y + 30 - tick_height), 
                    (200, 200, 200), 1)
            
            # Add percentage labels at major ticks
            if i % 5 == 0:
                label = f"{int(i * 10)}%"
                cv2.putText(frame, label, (tick_x - 10, timeline_y + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Calculate and draw timeline cursor position
        cursor_x = timeline_start_x + int(self.timeline_position * timeline_width)
        
        # Draw timeline cursor (triangular pointer)
        cursor_points = np.array([
            [cursor_x, timeline_y - 5],
            [cursor_x - 8, timeline_y - 15],
            [cursor_x + 8, timeline_y - 15]
        ], np.int32)
        cv2.fillPoly(frame, [cursor_points], (0, 255, 255))
        cv2.polylines(frame, [cursor_points], True, (0, 200, 200), 2)
        
        # Draw vertical line from cursor to timeline
        cv2.line(frame, (cursor_x, timeline_y - 5), (cursor_x, timeline_y + 30), (0, 255, 255), 2)
        
        # Timeline title
        cv2.putText(frame, "Timeline", (timeline_start_x, timeline_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_info(self, frame):
        """Draw FPS and gesture information"""
        cv2.putText(frame, f"FPS: {self.current_fps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.wheel_mode:
            cv2.putText(frame, "WHEEL MODE ACTIVE!", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            if self.three_finger_gesture:
                cv2.putText(frame, "KNOB ACTIVE!", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.putText(frame, "Make 3-finger gesture to control knob", (10, frame.shape[0] - 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Drop both hands to exit wheel mode - Press 'q' to quit", (10, frame.shape[0] - 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Touch fingertips (1,2) to knuckles to activate wheel", (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Hand Tracker with Virtual Knob and Timeline Started!")
        print("Extend thumb, index, and middle fingers to create a virtual knob")
        print("Move your index finger to rotate the knob and control the timeline")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            self.three_finger_gesture = False
            
            if results.multi_hand_landmarks and results.multi_handedness:
                right_hand_landmarks = None
                left_hand_landmarks = None
                
                # First pass: identify and draw all hands
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    
                    # Only draw skeleton for right hand, or for left hand if not in wheel mode
                    if hand_label == "Right" or not self.wheel_mode:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                        )
                    
                    # Draw hand numbers
                    self.draw_hand_numbers(frame, hand_landmarks.landmark, hand_label)
                    
                    # Store hand landmarks for touch detection
                    if hand_label == "Right":
                        right_hand_landmarks = hand_landmarks.landmark
                    elif hand_label == "Left":
                        left_hand_landmarks = hand_landmarks.landmark
                
                # Check for touch to activate wheel mode
                if not self.wheel_mode and right_hand_landmarks and left_hand_landmarks:
                    if self.detect_touch(right_hand_landmarks, left_hand_landmarks):
                        self.wheel_mode = True
                        self.wheel_persistent = True
                        print("Wheel mode activated!")
                
                # Always process three-finger gesture for right hand if in wheel mode
                if self.wheel_mode and right_hand_landmarks:
                    gesture_detected = self.detect_three_finger_gesture(right_hand_landmarks)
                    
                    if gesture_detected:
                        self.gesture_stability_count = min(self.gesture_stability_count + 1, self.required_stability)
                        
                        if self.gesture_stability_count >= self.required_stability:
                            self.three_finger_gesture = True
                            self.knob_active = True
                            self.knob_center = self.get_hand_center(right_hand_landmarks)
                            
                            finger_angle = self.calculate_finger_angle(right_hand_landmarks)
                            self.update_knob_angle(finger_angle)
                            
                            if self.fps_counter % 10 == 0:
                                timeline_percent = int(self.timeline_position * 100)
                                print(f"Knob: {int(math.degrees(self.knob_angle) % 360)}° | Timeline: {timeline_percent}%")
                    else:
                        self.gesture_stability_count = max(self.gesture_stability_count - 1, 0)
                        # Don't deactivate knob - just set gesture to false
                        self.three_finger_gesture = False
            else:
                # No hands detected - deactivate wheel mode and reset everything
                if self.wheel_mode:
                    self.wheel_mode = False
                    self.wheel_persistent = False
                    self.knob_active = False
                    self.three_finger_gesture = False
                    self.gesture_stability_count = 0
                    self.last_finger_angle = None
                    print("Wheel mode deactivated - no hands detected")
            
            # Always draw knob and timeline if wheel mode was ever activated (persistent)
            if self.wheel_persistent:
                self.draw_knob(frame)
                self.draw_timeline(frame)
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
