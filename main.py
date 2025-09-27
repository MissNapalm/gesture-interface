import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui

# Disable pyautogui failsafe for smoother operation
pyautogui.FAILSAFE = False

class CleanGestureDetector:
    def __init__(self):
        # Initialize MediaPipe hands with optimized settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
            model_complexity=0
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.frame_count = 0
        
        # Detection state
        self.last_detection_time = 0
        self.detection_cooldown = 0.3
        
        # Gesture action tracking
        self.last_action_time = {}
        self.action_cooldown = 2.0  # 2 seconds between actions
        
    def simple_finger_check(self, landmarks, tip_id, pip_id):
        """Optimized finger extension check"""
        return landmarks[tip_id].y < landmarks[pip_id].y
    
    def trigger_mission_control(self):
        """Trigger macOS Mission Control (Expose)"""
        try:
            print("ACTION: Opening Mission Control...")
            pyautogui.hotkey('ctrl', 'up')
            print("  - Mission Control opened!")
        except Exception as e:
            print(f"Error triggering Mission Control: {e}")
    
    def open_browser(self):
        """Open browser with multiple methods"""
        try:
            print("ACTION: Opening Browser...")
            
            # Method 1: Direct Chrome launch via open command (most reliable)
            try:
                import subprocess
                subprocess.run(['open', '-a', 'Google Chrome'], check=True)
                print("  - Chrome opened via direct launch!")
                return
            except:
                print("  - Direct Chrome launch failed, trying Spotlight...")
            
            # Method 2: Spotlight with Chrome
            pyautogui.hotkey('cmd', 'space')
            time.sleep(0.5)  # Give Spotlight time to fully open
            
            # Clear any existing search
            pyautogui.hotkey('cmd', 'a')
            time.sleep(0.1)
            
            pyautogui.typewrite('chrome')
            time.sleep(0.5)
            pyautogui.press('enter')
            print("  - Tried Chrome via Spotlight")
            
        except Exception as e:
            print(f"Chrome methods failed: {e}")
            
            # Method 3: Try Safari as absolute fallback
            try:
                import subprocess
                subprocess.run(['open', '-a', 'Safari'], check=True)
                print("  - Safari opened as fallback!")
            except:
                print("  - All browser opening methods failed")
    
    def close_window(self):
        """Close foreground window - ensuring Chrome is focused first"""
        try:
            print("ACTION: Closing Chrome tab/window...")
            
            # Method 1: Use AppleScript to directly close Chrome tab (most reliable)
            try:
                print("  - Using AppleScript to close Chrome tab...")
                import subprocess
                result = subprocess.run([
                    'osascript', '-e', 
                    'tell application "Google Chrome" to close active tab of front window'
                ], capture_output=True, text=True, check=True)
                print("  - ✓ Chrome tab closed via AppleScript!")
                return
            except subprocess.CalledProcessError as e:
                print(f"  - AppleScript failed: {e.stderr}")
            except Exception as e:
                print(f"  - AppleScript error: {e}")
            
            # Method 2: Focus Chrome first, then send close command
            try:
                print("  - Focusing Chrome application...")
                import subprocess
                subprocess.run(['osascript', '-e', 'tell application "Google Chrome" to activate'], check=True)
                time.sleep(0.3)  # Give Chrome time to come to front
                
                print("  - Sending Cmd+W to focused Chrome...")
                pyautogui.hotkey('cmd', 'w')
                print("  - ✓ Sent Cmd+W to Chrome")
                
            except Exception as e:
                print(f"  - Focus method failed: {e}")
            
        except Exception as e:
            print(f"ERROR: All close methods failed: {e}")
            print("  - Make sure Chrome is running!")
    
    def take_screenshot(self):
        """Take a screenshot using macOS shortcut"""
        try:
            print("ACTION: Taking screenshot...")
            
            # Method 1: Full screen screenshot (Cmd+Shift+3)
            pyautogui.hotkey('cmd', 'shift', '3')
            print("  - ✓ Screenshot saved to desktop!")
            
            # Optional: Also try the selection screenshot for variety
            # pyautogui.hotkey('cmd', 'shift', '4')  # Uncomment for selection mode
            
        except Exception as e:
            print(f"ERROR: Screenshot failed: {e}")
            
            # Method 2: Try alternative AppleScript approach
            try:
                print("  - Trying AppleScript method...")
                import subprocess
                subprocess.run([
                    'osascript', '-e', 
                    'do shell script "screencapture ~/Desktop/screenshot_$(date +%Y%m%d_%H%M%S).png"'
                ], check=True)
                print("  - ✓ Screenshot saved via AppleScript!")
            except Exception as e2:
                print(f"  - AppleScript method also failed: {e2}")
    
    def control_mouse_with_finger(self, landmarks):
        """Control mouse cursor with index finger position"""
        try:
            # Get index finger tip position (landmark 8)
            finger_tip = landmarks[8]
            
            # Convert finger position to screen coordinates
            # Flip x-axis since camera is mirrored
            screen_x = int((1 - finger_tip.x) * self.screen_width)
            screen_y = int(finger_tip.y * self.screen_height)
            
            # Add some boundaries to prevent cursor going off-screen
            screen_x = max(50, min(screen_x, self.screen_width - 50))
            screen_y = max(50, min(screen_y, self.screen_height - 50))
            
            # Smooth mouse movement to reduce jitter
            current_time = time.time()
            if current_time - self.last_mouse_time > 0.03:  # Limit to ~30 FPS for mouse
                pyautogui.moveTo(screen_x, screen_y, duration=0.1)
                self.last_mouse_time = current_time
                
                # Debug output occasionally
                if self.frame_count % 60 == 0:  # Every 2 seconds at 30fps
                    print(f"🎯 Mouse: ({screen_x}, {screen_y})")
            
        except Exception as e:
            if self.frame_count % 60 == 0:  # Don't spam errors
                print(f"Mouse control error: {e}")
    
    def execute_gesture_action(self, gesture, landmarks=None):
        """Execute actions based on detected gestures"""
        current_time = time.time()
        
        # Handle mouse control gesture differently (continuous action)
        if gesture == "Gun Hand":
            if landmarks:
                self.control_mouse_with_finger(landmarks)
            return  # Don't apply cooldown to mouse control
        
        # Check if enough time has passed since last action for this gesture
        if gesture in self.last_action_time:
            if current_time - self.last_action_time[gesture] < self.action_cooldown:
                return  # Too soon to trigger again
        
        # Execute action based on gesture
        if gesture == "Upside Down Peace Sign" or gesture == "Upside Down Peace Sign (Alt)":
            self.trigger_mission_control()
            self.last_action_time[gesture] = current_time
        elif gesture == "Three Fingers":
            self.open_browser()
            self.last_action_time[gesture] = current_time
        elif gesture == "Pinky Up":
            print(f"🎯 EXECUTING PINKY ACTION at {current_time}")
            self.close_window()
            self.last_action_time[gesture] = current_time
            print(f"🎯 PINKY ACTION COMPLETED")
        elif gesture == "Hang Loose":
            print(f"🤙 EXECUTING HANG LOOSE ACTION at {current_time}")
            self.take_screenshot()
            self.last_action_time[gesture] = current_time
            print(f"🤙 HANG LOOSE ACTION COMPLETED")
    
    def detect_gestures(self, landmarks):
        """Precise gesture detection with debugging"""
        fingers = {
            'thumb': self.simple_finger_check(landmarks, 4, 3),
            'index': self.simple_finger_check(landmarks, 8, 6),
            'middle': self.simple_finger_check(landmarks, 12, 10),
            'ring': self.simple_finger_check(landmarks, 16, 14),
            'pinky': self.simple_finger_check(landmarks, 20, 18)
        }
        
        # Count how many fingers are up
        fingers_up_count = sum(fingers.values())
        
        gestures = []
        
        # Debug output every 30 frames
        if self.frame_count % 30 == 0:
            finger_states = [f"{k}: {'UP' if v else 'DOWN'}" for k, v in fingers.items()]
            print(f"Debug - {', '.join(finger_states)} | Total up: {fingers_up_count}")
            
            # Special pinky debug
            if fingers['pinky']:
                other_fingers_down = not fingers['thumb'] and not fingers['index'] and not fingers['middle'] and not fingers['ring']
                print(f"  Pinky Debug - Pinky UP: {fingers['pinky']}, Others down: {other_fingers_down}, Total count: {fingers_up_count}")
        
        # PRECISE gesture detection - order matters for priority!
        
        # 1. Pinky up: ONLY pinky up (thumb can be either way), all others DOWN
        if (not fingers['index'] and not fingers['middle'] 
              and not fingers['ring'] and fingers['pinky']):
            # Allow 1 finger (just pinky) or 2 fingers (pinky + thumb)
            if fingers_up_count == 1 or (fingers_up_count == 2 and fingers['thumb']):
                gestures.append('Pinky Up')
                print(f"  PINKY DETECTED! Total fingers: {fingers_up_count}")
        
        # 2. Hang Loose: ONLY thumb + pinky up, all others DOWN
        elif (fingers['thumb'] and not fingers['index'] and not fingers['middle'] 
              and not fingers['ring'] and fingers['pinky']):
            # Must be exactly 2 fingers (thumb + pinky)
            if fingers_up_count == 2:
                gestures.append('Hang Loose')
                print(f"  HANG LOOSE DETECTED! 🤙 Total fingers: {fingers_up_count}")
        
        # 3. Upside down peace: EXACTLY ring + pinky up, thumb optional, others DOWN
        # 3. Upside down peace: EXACTLY ring + pinky up, thumb optional, others DOWN
        elif (not fingers['index'] and not fingers['middle'] 
            and fingers['ring'] and fingers['pinky']):
            # Allow 2 or 3 fingers up (ring + pinky + optional thumb)
            if fingers_up_count == 2:
                gestures.append('Upside Down Peace Sign (Alt)')
            elif fingers_up_count == 3 and fingers['thumb']:
                gestures.append('Upside Down Peace Sign')
        
        # 4. Three fingers: index + middle + ring up, pinky DOWN (thumb can be either)
        # 4. Three fingers: index + middle + ring up, pinky DOWN (thumb can be either)
        elif (fingers['index'] and fingers['middle'] and fingers['ring'] 
              and not fingers['pinky']):
            # Allow 3 or 4 fingers (with optional thumb)
            if fingers_up_count == 3 or fingers_up_count == 4:
                gestures.append('Three Fingers')
        
        # 5. Gun Hand: thumb + index up, others DOWN (removed - replaced with Pointer Control)
        # elif (fingers['thumb'] and fingers['index'] and not fingers['middle'] 
        #       and not fingers['ring'] and not fingers['pinky']):
        #     if fingers_up_count == 2:
        #         gestures.append('Gun Hand')
        
        # 5. Point: ONLY index up, all others DOWN (but this is handled by Pointer Control now)
        # 5. Point: ONLY index up, all others DOWN
        # 6. Point: ONLY index up, all others DOWN
        # 5. Point: ONLY index up, all others DOWN (but this is handled by Pointer Control now)
        elif (not fingers['thumb'] and fingers['index'] and not fingers['middle'] 
              and not fingers['ring'] and not fingers['pinky']):
            # Must be exactly 1 finger
            if fingers_up_count == 1:
                gestures.append('Point')
        
        # 6. Peace sign: EXACTLY index + middle up, others DOWN
        # 6. Peace sign: EXACTLY index + middle up, others DOWN
        # 7. Peace sign: EXACTLY index + middle up, others DOWN
        # 6. Peace sign: EXACTLY index + middle up, others DOWN
        elif (fingers['index'] and fingers['middle'] 
              and not fingers['ring'] and not fingers['pinky'] and not fingers['thumb']):
            # Must be exactly 2 fingers
            if fingers_up_count == 2:
                gestures.append('Peace Sign')
        
        # 7. Open hand: ALL or most fingers up
        # 7. Open hand: ALL or most fingers up
        # 8. Open hand: ALL or most fingers up
        # 7. Open hand: ALL or most fingers up
        elif fingers_up_count >= 4:
            gestures.append('Open Hand')
        
        # 8. Fist: NO fingers up
        elif fingers_up_count == 0:
            gestures.append('Fist')
        
        return gestures, fingers
    
    def draw_stats(self, frame, gestures, hand_count):
        """Draw minimal stats in corners"""
        h, w = frame.shape[:2]
        
        # Top-left: FPS
        cv2.putText(frame, f"FPS: {self.current_fps}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Top-right: Hands detected
        cv2.putText(frame, f"Hands: {hand_count}", (w-100, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Bottom-left: Frame count
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Bottom-right: Detection count
        cv2.putText(frame, f"Gestures: {len(gestures)}", (w-120, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Bottom-left: Active gestures (only when detected)
        if gestures:
            y_offset = h - 60  # Start a bit above the bottom
            for gesture in gestures:
                cv2.putText(frame, gesture, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                y_offset += 40
    
    def run(self):
        """Clean main loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        print("Clean Gesture Detector Started")
        print("Gestures:")
        print("  - Upside Down Peace Sign (ring + pinky) -> Mission Control")
        print("  - Three Fingers (index + middle + ring) -> Open Browser")
        print("  - Pinky Up (only pinky) -> Close Window")
        print("  - Hang Loose (thumb + pinky) 🤙 -> Take Screenshot")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            frame = cv2.flip(frame, 1)
            
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # FPS calculation
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            # Collect gestures
            all_gestures = []
            hand_count = 0
            
            if results.multi_hand_landmarks and results.multi_handedness:
                hand_count = len(results.multi_hand_landmarks)
                
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Draw clean white wireframes with thicker lines
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                    
                    hand_label = handedness.classification[0].label
                    gestures, fingers = self.detect_gestures(hand_landmarks.landmark)
                    
                    for gesture in gestures:
                        all_gestures.append(f"{gesture} ({hand_label})")
                        self.execute_gesture_action(gesture)
            
            # Console output (throttled)
            if all_gestures:
                current_time = time.time()
                if current_time - self.last_detection_time > self.detection_cooldown:
                    for gesture in all_gestures:
                        print(f"DETECTED: {gesture}")
                    self.last_detection_time = current_time
            
            # Draw minimal stats
            self.draw_stats(frame, all_gestures, hand_count)
            
            cv2.imshow('Gesture Detector', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = CleanGestureDetector()
    detector.run()
