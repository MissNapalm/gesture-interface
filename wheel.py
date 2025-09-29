import cv2
import mediapipe as mp
import time
import numpy as np
import math
import pygame
import threading
import os

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
        
        # Initialize pygame mixer for audio
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Load audio files
        self.load_audio_files()
        
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
        self.show_wheel = False  # Controls when to actually draw the wheel
        
        # Audio state tracking
        self.wheel_sound_played = False
        self.startup_sound_played = False
        self.last_knob_angle = self.knob_angle
        self.whoosh_cooldown = 0  # Prevent too frequent whoosh sounds
        self.touch_sound_played = False
        
        # Visual effects
        self.blur_enabled = False  # No visual filters
        self.blur_strength = 45   
        self.blue_overlay = False  # No blue overlay
        
        # Center vertical bar properties
        self.bar_width = 0  # Width of the vertical bar
        self.bar_color = (0, 0, 0)  # Black color (BGR)
    
    def load_audio_files(self):
        """Load audio files if they exist"""
        self.sounds = {}
        audio_files = {
            'startup': 'music.mp3',
            'wheel': 'active.mp3',
            'powerdown': 'powerdown.mp3',
            'whoosh': 'whoosh.mp3',
            'active': 'swipe.mp3'
        }
        
        for sound_name, filename in audio_files.items():
            if os.path.exists(filename):
                try:
                    self.sounds[sound_name] = pygame.mixer.Sound(filename)
                    print(f"Loaded {filename}")
                except pygame.error as e:
                    print(f"Could not load {filename}: {e}")
            else:
                print(f"Audio file {filename} not found - continuing without this sound")
    
    def play_sound(self, sound_name):
        """Play a sound effect in a separate thread to avoid blocking"""
        def play():
            if sound_name in self.sounds:
                try:
                    self.sounds[sound_name].play()
                except pygame.error as e:
                    print(f"Error playing {sound_name}: {e}")
        
        # Play sound in separate thread to avoid blocking the main loop
        sound_thread = threading.Thread(target=play)
        sound_thread.daemon = True
        sound_thread.start()
    
    # draw_center_bar removed: no center bar will be drawn
    
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
    
    def detect_ok_gesture(self, landmarks):
        """Detect OK gesture (thumb and index finger touching, other fingers extended)"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate distance between thumb and index finger tips
        distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        
        # Check if thumb and index are close (forming circle)
        fingers_touching = distance < 0.05  # Adjust threshold as needed
        
        # Check if middle, ring, and pinky are extended
        middle_extended = self.is_finger_extended(landmarks, 12, 10)
        ring_extended = self.is_finger_extended(landmarks, 16, 14)
        pinky_extended = self.is_finger_extended(landmarks, 20, 18)
        
        other_fingers_up = middle_extended and ring_extended and pinky_extended
        
        return fingers_touching and other_fingers_up
    
    def detect_three_up_thumb_down(self, landmarks):
        """Detect three fingers up (index, middle, ring) with thumb and pinky down"""
        # Check fingers are extended
        index_extended = self.is_finger_extended(landmarks, 8, 6)
        middle_extended = self.is_finger_extended(landmarks, 12, 10)
        ring_extended = self.is_finger_extended(landmarks, 16, 14)
        
        # Check thumb and pinky are down
        thumb_down = landmarks[4].y > landmarks[3].y  # Thumb tip below thumb joint
        pinky_down = not self.is_finger_extended(landmarks, 20, 18)
        
        three_fingers_up = index_extended and middle_extended and ring_extended
        
        return three_fingers_up and thumb_down and pinky_down
    
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
            
            # Store previous knob angle before updating
            previous_knob_angle = self.knob_angle
            
            self.knob_angle += angle_diff * 2
            self.knob_angle = self.knob_angle % (2 * math.pi)
            
            # Check for significant wheel movement to play whoosh sound
            angle_change = abs(self.knob_angle - previous_knob_angle)
            # Handle wrap-around for angle comparison
            if angle_change > math.pi:
                angle_change = 2 * math.pi - angle_change
            
            # Play whoosh if significant movement and cooldown has passed
            if angle_change > 0.2 and self.whoosh_cooldown <= 0:  # Lowered threshold to ~11 degrees
                self.play_sound('whoosh')
                self.whoosh_cooldown = 5  # Reduced cooldown for more responsive sound
                print(f"Whoosh! Angle change: {math.degrees(angle_change):.1f}°")
            
            # Update timeline position based on knob angle
            # Map angle (0 to 2π) to timeline position (0.0 to 1.0)
            self.timeline_position = (self.knob_angle / (2 * math.pi)) % 1.0
        
        self.last_finger_angle = current_angle
    
    def create_blue_background(self, frame_shape):
        """Create a solid blue background"""
        return np.full(frame_shape, (100, 50, 20), dtype=np.uint8)  # BGR format: blue background
    
    def apply_blur_effect(self, frame):
        """Apply blur effect to the video feed"""
        if self.blur_enabled:
            # Apply Gaussian blur - kernel size must be odd
            blur_kernel = max(1, self.blur_strength if self.blur_strength % 2 == 1 else self.blur_strength + 1)
            blurred_frame = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
            return blurred_frame
        return frame
    
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
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
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
    
    def is_right_hand_on_right_side(self, landmarks, frame_width):
        """Check if right hand is on the right side of the screen"""
        hand_center = self.get_hand_center(landmarks)
        hand_x = hand_center.x * frame_width
        return hand_x > frame_width * 0.6  # Right hand must be on right 40% of screen
    
    def draw_knob(self, frame):
        """Draw the virtual knob/wheel with retro white/green/red colors"""
        if not self.knob_active or self.knob_center is None:
            return
        
        frame_h, frame_w = frame.shape[:2]
        center_x = int(self.knob_center.x * frame_w) - 60
        center_y = int(self.knob_center.y * frame_h)
        
        # Keep knob within frame boundaries
        margin = self.knob_radius + 20
        center_x = max(margin, min(center_x, frame_w - margin))
        center_y = max(margin, min(center_y, frame_h - margin))
        
        # Draw outer circles - retro style with green outer, white inner
        cv2.circle(frame, (center_x, center_y), self.knob_radius, (0, 255, 0), 3)  # Green outer ring
        cv2.circle(frame, (center_x, center_y), self.knob_radius - 15, (255, 255, 255), 2)  # White inner ring
        
        # Draw tick marks in white
        for i in range(12):
            tick_angle = i * (2 * math.pi / 12)
            outer_x = center_x + int((self.knob_radius - 8) * math.cos(tick_angle))
            outer_y = center_y + int((self.knob_radius - 8) * math.sin(tick_angle))
            inner_x = center_x + int((self.knob_radius - 22) * math.cos(tick_angle))
            inner_y = center_y + int((self.knob_radius - 22) * math.sin(tick_angle))
            
            cv2.line(frame, (outer_x, outer_y), (inner_x, inner_y), (255, 255, 255), 2)  # White ticks
        
        # Draw red pointer
        pointer_length = self.knob_radius - 30
        pointer_x = center_x + int(pointer_length * math.cos(self.knob_angle))
        pointer_y = center_y + int(pointer_length * math.sin(self.knob_angle))
        
        cv2.line(frame, (center_x, center_y), (pointer_x, pointer_y), (0, 0, 255), 5)  # Red pointer
        cv2.circle(frame, (pointer_x, pointer_y), 10, (0, 0, 255), -1)  # Red tip
        cv2.circle(frame, (center_x, center_y), 8, (255, 255, 255), -1)  # White center
        
        # Display angle and timeline position in green
        angle_degrees = int(math.degrees(self.knob_angle) % 360)
        timeline_percent = int(self.timeline_position * 100)
        cv2.putText(frame, f"Angle: {angle_degrees}° | Timeline: {timeline_percent}%", 
                   (center_x - 80, center_y + self.knob_radius + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
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
                     (40, 30, 15), -1)  # Darker blue background
        
        # Draw timeline border
        cv2.rectangle(frame, 
                     (timeline_start_x, timeline_y), 
                     (timeline_end_x, timeline_y + 30), 
                     (255, 255, 255), 2)
        
        # Draw timeline tick marks
        num_ticks = 10
        for i in range(num_ticks + 1):
            tick_x = timeline_start_x + int(i * timeline_width / num_ticks)
            tick_height = 15 if i % 5 == 0 else 8  # Larger ticks every 5 marks
            cv2.line(frame, 
                    (tick_x, timeline_y + 30), 
                    (tick_x, timeline_y + 30 - tick_height), 
                    (255, 255, 255), 1)
            
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
        cv2.fillPoly(frame, [cursor_points], (255, 255, 255))
        cv2.polylines(frame, [cursor_points], True, (255, 255, 255), 2)
        
        # Draw vertical line from cursor to timeline
        cv2.line(frame, (cursor_x, timeline_y - 5), (cursor_x, timeline_y + 30), (255, 255, 255), 2)
        
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
            
            if not self.show_wheel:
                cv2.putText(frame, "Move right hand to right side to show wheel", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            elif self.three_finger_gesture:
                cv2.putText(frame, "KNOB ACTIVE!", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if self.show_wheel:
                cv2.putText(frame, "Make 3-finger gesture to control knob", (10, frame.shape[0] - 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Drop both hands to exit wheel mode", (10, frame.shape[0] - 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Touch fingertips (1,2) to knuckles to activate wheel", (10, frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Play startup sound
        if not self.startup_sound_played:
            self.play_sound('startup')
            self.startup_sound_played = True
        
        print("Hand Tracker with Virtual Knob, Timeline, and Center Box Started!")
        print("Touch fingertips (1,2) to knuckles to activate wheel")
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
            
            # Center bar removed
            
            self.three_finger_gesture = False
            
            if results.multi_hand_landmarks and results.multi_handedness:
                right_hand_landmarks = None
                left_hand_landmarks = None
                
                # First pass: identify and draw all hands
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    
                    # Much more aggressive skeleton hiding to reduce jitter
                    show_skeleton = False
                    
                    # Only show skeleton for right hand when NOT in wheel mode
                    if hand_label == "Right" and not self.wheel_mode:
                        show_skeleton = True
                    # For left hand, only show skeleton if not in wheel mode AND fingers are clearly extended
                    elif hand_label == "Left" and not self.wheel_mode:
                        landmarks = hand_landmarks.landmark
                        # Only show if at least 2 fingers are clearly extended
                        extended_fingers = sum([
                            self.is_finger_extended(landmarks, 8, 6),   # Index
                            self.is_finger_extended(landmarks, 12, 10), # Middle  
                            self.is_finger_extended(landmarks, 16, 14), # Ring
                            self.is_finger_extended(landmarks, 20, 18)  # Pinky
                        ])
                        if extended_fingers >= 2:
                            show_skeleton = True
                    
                    if show_skeleton:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                        )
                    
                    # Draw hand numbers on the frame
                    self.draw_hand_numbers(frame, hand_landmarks.landmark, hand_label)
                    
                    # Store hand landmarks for touch detection
                    if hand_label == "Right":
                        right_hand_landmarks = hand_landmarks.landmark
                    elif hand_label == "Left":
                        left_hand_landmarks = hand_landmarks.landmark
                
                # Check for touch to activate wheel mode
                if not self.wheel_mode and right_hand_landmarks and left_hand_landmarks:
                    if self.detect_touch(right_hand_landmarks, left_hand_landmarks):
                        # Play active sound when touch is detected
                        if not self.touch_sound_played:
                            self.play_sound('active')
                            self.touch_sound_played = True
                        
                        self.wheel_mode = True
                        self.wheel_persistent = True
                        self.wheel_sound_played = False  # Reset wheel sound flag
                        print("Wheel mode activated!")
                else:
                    # Reset touch sound flag when not touching
                    self.touch_sound_played = False
                
                # Always process three-finger gesture for right hand if in wheel mode
                if self.wheel_mode and right_hand_landmarks:
                    frame_h, frame_w = frame.shape[:2]
                    
                    # Check if right hand is on right side of screen to show wheel
                    if self.is_right_hand_on_right_side(right_hand_landmarks, frame_w):
                        if not self.show_wheel:
                            self.show_wheel = True
                            # Play wheel sound when wheel first appears
                            if not self.wheel_sound_played:
                                self.play_sound('wheel')
                                self.wheel_sound_played = True
                    
                    # Only process gesture if wheel is showing
                    if self.show_wheel:
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
                # No hands detected - deactivate wheel mode and close test window
                if self.wheel_mode:
                    # Play powerdown sound when hands are dropped
                    self.play_sound('powerdown')
                    
                    self.wheel_mode = False
                    self.wheel_persistent = False
                    self.show_wheel = False
                    self.knob_active = False
                    self.three_finger_gesture = False
                    self.gesture_stability_count = 0
                    self.last_finger_angle = None
                    self.wheel_sound_played = False  # Reset for next time
                    self.touch_sound_played = False  # Reset touch sound
                    print("Wheel mode deactivated - no hands detected")
            
            # Draw all HUD elements on the frame (AFTER the center box, so they appear on top)
            # Only draw knob and timeline if wheel should be shown
            if self.wheel_persistent and self.show_wheel:
                self.draw_knob(frame)
                self.draw_timeline(frame)
            self.draw_info(frame)
            
            # Update cooldowns
            if self.whoosh_cooldown > 0:
                self.whoosh_cooldown -= 1
            
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            cv2.imshow('Hand Tracker', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if self.test_window_open:
                    self.close_test_window()
                break
        
        cap.release()
        cv2.destroyAllWindows()
        # Clean up pygame mixer
        pygame.mixer.quit()

if __name__ == "__main__":
    tracker = HandTracker()
    tracker.run()
