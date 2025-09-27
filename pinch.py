import cv2
import mediapipe as mp
import numpy as np
import time

class HandTrackingController:
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
        
        # Get screen dimensions (for display purposes)
        self.screen_width = 1920  # Default screen width
        self.screen_height = 1080  # Default screen height
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
        
        # Cursor settings
        self.cursor_x = self.screen_width // 2
        self.cursor_y = self.screen_height // 2
        self.smoothing_factor = 0.7
        self.last_cursor_x = self.cursor_x
        self.last_cursor_y = self.cursor_y
        
        # Gesture states
        self.is_pinching = False
        self.last_pinch_state = False
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def calculate_distance(self, point1, point2):
        """Calculate distance between two landmark points"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def simple_finger_check(self, landmarks, tip_id, pip_id):
        """Check if finger is extended"""
        return landmarks[tip_id].y < landmarks[pip_id].y
    
    def detect_pinch(self, landmarks):
        """Detect if thumb and index finger are pinched together"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = self.calculate_distance(thumb_tip, index_tip)
        return distance < 0.05
    
    def get_cursor_position(self, landmarks, hand_label):
        """Get cursor position from midpoint between thumb and index finger"""
        # Only process right hand
        if hand_label != "Right":
            return self.last_cursor_x, self.last_cursor_y
            
        # Get thumb tip (landmark 4) and index finger tip (landmark 8)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate midpoint between thumb and index finger
        mid_x = (thumb_tip.x + index_tip.x) / 2
        mid_y = (thumb_tip.y + index_tip.y) / 2
        
        # Map to screen coordinates
        screen_x = int(mid_x * self.screen_width)
        screen_y = int(mid_y * self.screen_height)
        
        # Add 20px offset to the right for right hand
        screen_x += 20
        
        # Apply smoothing
        smooth_x = int(self.smoothing_factor * self.last_cursor_x + (1 - self.smoothing_factor) * screen_x)
        smooth_y = int(self.smoothing_factor * self.last_cursor_y + (1 - self.smoothing_factor) * screen_y)
        
        # Keep in bounds
        smooth_x = max(0, min(smooth_x, self.screen_width - 1))
        smooth_y = max(0, min(smooth_y, self.screen_height - 1))
        
        self.last_cursor_x = smooth_x
        self.last_cursor_y = smooth_y
        
        return smooth_x, smooth_y
    
    def handle_click(self, is_pinching):
        """Handle click state changes (visual feedback only)"""
        if is_pinching and not self.last_pinch_state:
            # Pinch started
            print(f"CLICK DOWN at ({self.cursor_x}, {self.cursor_y})")
        elif not is_pinching and self.last_pinch_state:
            # Pinch ended
            print(f"CLICK UP at ({self.cursor_x}, {self.cursor_y})")
        
        self.last_pinch_state = is_pinching
    
    def draw_cursor_overlay(self, frame):
        """Draw visual cursor overlay on the frame"""
        # Map screen coordinates back to frame coordinates for display
        frame_h, frame_w = frame.shape[:2]
        frame_x = int((self.cursor_x / self.screen_width) * frame_w)
        frame_y = int((self.cursor_y / self.screen_height) * frame_h)
        
        cursor_color = (0, 0, 255) if self.is_pinching else (0, 255, 0)
        cursor_radius = 20 if self.is_pinching else 15
        
        # Main cursor circle
        cv2.circle(frame, (frame_x, frame_y), cursor_radius, cursor_color, -1)
        cv2.circle(frame, (frame_x, frame_y), cursor_radius + 3, cursor_color, 3)
        
        # Crosshair
        line_length = 25
        cv2.line(frame, 
                (frame_x - line_length, frame_y), 
                (frame_x + line_length, frame_y), 
                cursor_color, 3)
        cv2.line(frame, 
                (frame_x, frame_y - line_length), 
                (frame_x, frame_y + line_length), 
                cursor_color, 3)
    
    def draw_info(self, frame):
        """Draw information overlay"""
        h, w = frame.shape[:2]
        
        # FPS
        cv2.putText(frame, f"FPS: {self.current_fps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Screen cursor position
        cv2.putText(frame, f"Position: ({self.cursor_x}, {self.cursor_y})", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Pinch state
        if self.is_pinching:
            cv2.putText(frame, "PINCHING!", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Instructions
        instructions = [
            "Right hand tracking demonstration",
            "Pinch thumb+index for click feedback",
            "Press 'q' to quit"
        ]
        
        start_y = h - len(instructions) * 25 - 10
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, start_y + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_hand_info(self, frame, hand_data):
        """Draw hand information"""
        for landmarks, hand_label in hand_data:
            # Only process right hand
            if hand_label != "Right":
                continue
                
            # Get hand center for label placement
            hand_center_x = int(landmarks[9].x * frame.shape[1])
            hand_center_y = int(landmarks[9].y * frame.shape[0]) - 40
            
            # Check pinch status for this hand
            is_pinching = self.detect_pinch(landmarks)
            
            # Color for right hand
            color = (255, 100, 100)
            
            # Draw hand label
            label_text = f"{hand_label}"
            if is_pinching:
                label_text += ": PINCH"
            
            cv2.putText(frame, label_text, (hand_center_x - 80, hand_center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Right Hand Tracking Controller Started!")
        print("Move your right hand to see cursor tracking")
        print("Pinch thumb + index for click feedback")
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
            
            hand_data = []
            self.is_pinching = False
            
            # Process each detected hand
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    
                    # Only process right hand
                    if hand_label != "Right":
                        continue
                    
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    hand_data.append((hand_landmarks.landmark, hand_label))
                    
                    # Update cursor position based on hand movement
                    self.cursor_x, self.cursor_y = self.get_cursor_position(hand_landmarks.landmark, hand_label)
                    
                    # Check for pinch
                    if self.detect_pinch(hand_landmarks.landmark):
                        self.is_pinching = True
                
                # Handle clicking based on overall pinch state
                self.handle_click(self.is_pinching)
            else:
                # No hands detected - release any held click
                if self.last_pinch_state:
                    print("Released click - no hands detected")
                    self.last_pinch_state = False
            
            # Draw hand info
            if hand_data:
                self.draw_hand_info(frame, hand_data)
            
            # Draw cursor overlay and info
            self.draw_cursor_overlay(frame)
            self.draw_info(frame)
            
            # Calculate FPS
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            # Show frame
            cv2.imshow('Hand Tracking Controller', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandTrackingController()
    controller.run()
