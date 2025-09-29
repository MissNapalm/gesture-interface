import sys
import cv2
import mediapipe as mp
import time
import numpy as np
import math
import pygame
import threading
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QPainterPath, QRadialGradient, QLinearGradient

class FuturisticWheel(QWidget):
    def __init__(self):
        super().__init__()
        self.angle = math.pi
        self.timeline_position = 0.5
        self.active = False
        self.center_x = 0
        self.center_y = 0
        self.radius = 110
        self.wheel_style = 0
        self.setMinimumSize(400, 400)
        
    def set_data(self, angle, timeline_pos, active, center_x, center_y):
        self.angle = angle
        self.timeline_position = timeline_pos
        self.active = active
        self.center_x = center_x
        self.center_y = center_y
        self.update()
    
    def set_style(self, style):
        self.wheel_style = style % 5
        self.update()
    
    def paintEvent(self, event):
        if not self.active:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        cx = self.center_x if self.center_x > 0 else self.width() // 2
        cy = self.center_y if self.center_y > 0 else self.height() // 2
        
        if self.wheel_style == 0:
            self.draw_style_minimal(painter, cx, cy)
        elif self.wheel_style == 1:
            self.draw_style_technical(painter, cx, cy)
        elif self.wheel_style == 2:
            self.draw_style_orbital(painter, cx, cy)
        elif self.wheel_style == 3:
            self.draw_style_radar(painter, cx, cy)
        elif self.wheel_style == 4:
            self.draw_style_mechanical(painter, cx, cy)
    
    def draw_style_minimal(self, painter, cx, cy):
        """Style 0: Holographic HUD with floating data nodes"""
        white = QColor(255, 255, 255)
        
        # Pulsing outer glow rings
        for i in range(5):
            r = self.radius + 15 + i * 10
            opacity = 100 - (i * 20)
            painter.setPen(QPen(QColor(255, 255, 255, opacity), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), r, r)
        
        # Main segmented ring
        num_segments = 48
        for i in range(num_segments):
            if i % 3 != 1:  # Create gaps
                start_angle = (i * 360 / num_segments) - 90
                span_angle = (360 / num_segments) - 3
                painter.setPen(QPen(white, 3))
                painter.drawArc(QRectF(cx - self.radius, cy - self.radius,
                                      self.radius * 2, self.radius * 2),
                               int(start_angle * 16), int(span_angle * 16))
        
        # Floating data nodes at cardinal points
        for i in range(4):
            angle = i * math.pi / 2
            node_x = cx + int((self.radius + 35) * math.cos(angle))
            node_y = cy + int((self.radius + 35) * math.sin(angle))
            
            # Node box
            painter.setPen(QPen(white, 2))
            painter.setBrush(QColor(20, 20, 20, 220))
            painter.drawRect(node_x - 15, node_y - 8, 30, 16)
            
            # Connector line
            inner_x = cx + int(self.radius * math.cos(angle))
            inner_y = cy + int(self.radius * math.sin(angle))
            painter.setPen(QPen(QColor(255, 255, 255, 150), 1))
            painter.drawLine(inner_x, inner_y, node_x, node_y)
        
        # Inner targeting ring
        painter.setPen(QPen(QColor(255, 255, 255, 120), 2))
        painter.drawEllipse(QPointF(cx, cy), self.radius - 20, self.radius - 20)
        
        # Progress indicator (arc style)
        painter.setPen(QPen(white, 6))
        painter.drawArc(QRectF(cx - self.radius + 30, cy - self.radius + 30,
                              (self.radius - 30) * 2, (self.radius - 30) * 2),
                       -90 * 16, int(self.timeline_position * 360 * 16))
        
        self.draw_pointer_and_data(painter, cx, cy)
    
    def draw_style_technical(self, painter, cx, cy):
        """Style 1: Technical schematic with measurement grid"""
        white = QColor(255, 255, 255)
        
        # Grid system - horizontal and vertical lines
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        grid_size = 20
        for i in range(-self.radius, self.radius + 1, grid_size):
            # Vertical lines
            if cx + i - self.radius > 0 and cx + i + self.radius < self.width():
                painter.drawLine(cx + i, cy - self.radius, cx + i, cy + self.radius)
            # Horizontal lines  
            if cy + i - self.radius > 0 and cy + i + self.radius < self.height():
                painter.drawLine(cx - self.radius, cy + i, cx + self.radius, cy + i)
        
        # Outer technical frame
        frame_size = self.radius + 25
        painter.setPen(QPen(white, 3))
        painter.drawLine(cx - frame_size, cy - frame_size, cx - frame_size + 30, cy - frame_size)
        painter.drawLine(cx - frame_size, cy - frame_size, cx - frame_size, cy - frame_size + 30)
        painter.drawLine(cx + frame_size, cy - frame_size, cx + frame_size - 30, cy - frame_size)
        painter.drawLine(cx + frame_size, cy - frame_size, cx + frame_size, cy - frame_size + 30)
        painter.drawLine(cx - frame_size, cy + frame_size, cx - frame_size + 30, cy + frame_size)
        painter.drawLine(cx - frame_size, cy + frame_size, cx - frame_size, cy + frame_size - 30)
        painter.drawLine(cx + frame_size, cy + frame_size, cx + frame_size - 30, cy + frame_size)
        painter.drawLine(cx + frame_size, cy + frame_size, cx + frame_size, cy + frame_size - 30)
        
        # Main circle with measurement ticks
        painter.setPen(QPen(white, 4))
        painter.drawEllipse(QPointF(cx, cy), self.radius, self.radius)
        
        # Measurement ticks around circumference
        for i in range(36):
            angle = i * (2 * math.pi / 36)
            tick_len = 15 if i % 3 == 0 else 8
            r1 = self.radius
            r2 = self.radius - tick_len
            x1 = cx + int(r1 * math.cos(angle))
            y1 = cy + int(r1 * math.sin(angle))
            x2 = cx + int(r2 * math.cos(angle))
            y2 = cy + int(r2 * math.sin(angle))
            painter.setPen(QPen(white, 2 if i % 3 == 0 else 1))
            painter.drawLine(x1, y1, x2, y2)
        
        # Crosshair
        painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
        painter.drawLine(cx - self.radius, cy, cx + self.radius, cy)
        painter.drawLine(cx, cy - self.radius, cx, cy + self.radius)
        
        # Progress fill (semi-transparent)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 50))
        path = QPainterPath()
        path.moveTo(cx, cy)
        path.arcTo(QRectF(cx - self.radius, cy - self.radius, self.radius * 2, self.radius * 2),
                  -90, self.timeline_position * 360)
        path.lineTo(cx, cy)
        painter.drawPath(path)
        
        self.draw_pointer_and_data(painter, cx, cy)
    
    def draw_style_orbital(self, painter, cx, cy):
        """Style 2: Planetary orbit system with satellites"""
        white = QColor(255, 255, 255)
        
        # Multiple orbital paths
        orbit_radii = [self.radius, self.radius - 25, self.radius - 50, self.radius - 75]
        for i, r in enumerate(orbit_radii):
            if r > 0:
                opacity = 180 - i * 30
                thickness = 3 if i == 0 else 1
                painter.setPen(QPen(QColor(255, 255, 255, opacity), thickness))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPointF(cx, cy), r, r)
        
        # Orbiting satellites (rotate based on angle)
        num_satellites = 6
        for i in range(num_satellites):
            orbit_angle = (i * 2 * math.pi / num_satellites) + self.angle * 0.3
            orbit_r = self.radius - 25
            sat_x = cx + int(orbit_r * math.cos(orbit_angle))
            sat_y = cy + int(orbit_r * math.sin(orbit_angle))
            
            # Draw satellite
            painter.setBrush(white)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(sat_x, sat_y), 4, 4)
            
            # Connection line to center
            painter.setPen(QPen(QColor(255, 255, 255, 80), 1))
            painter.drawLine(cx, cy, sat_x, sat_y)
        
        # Progress dots on outer orbit
        dots = int(self.timeline_position * 48)
        for i in range(dots):
            angle = -math.pi/2 + (i / 48) * 2 * math.pi
            dx = cx + int(self.radius * math.cos(angle))
            dy = cy + int(self.radius * math.sin(angle))
            painter.setBrush(white)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(dx, dy), 3, 3)
        
        # Outer ring decorations
        for i in range(8):
            angle = i * (2 * math.pi / 8)
            x = cx + int((self.radius + 15) * math.cos(angle))
            y = cy + int((self.radius + 15) * math.sin(angle))
            painter.setPen(QPen(white, 2))
            painter.drawLine(cx + int(self.radius * math.cos(angle)),
                           cy + int(self.radius * math.sin(angle)), x, y)
        
        self.draw_pointer_and_data(painter, cx, cy)
    
    def draw_style_radar(self, painter, cx, cy):
        """Style 3: Advanced radar system with sweep and contacts"""
        white = QColor(255, 255, 255)
        
        # Radar screen background rings
        for i in range(1, 5):
            r = self.radius * i / 4
            painter.setPen(QPen(QColor(255, 255, 255, 80), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), r, r)
        
        # Main radar circle
        painter.setPen(QPen(white, 4))
        painter.drawEllipse(QPointF(cx, cy), self.radius, self.radius)
        
        # Radar grid lines (8 directions)
        painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
        for i in range(8):
            angle = i * (2 * math.pi / 8)
            ex = cx + int(self.radius * math.cos(angle))
            ey = cy + int(self.radius * math.sin(angle))
            painter.drawLine(cx, cy, ex, ey)
        
        # Radar sweep (animated gradient)
        for i in range(3):
            gradient = QRadialGradient(cx, cy, self.radius)
            gradient.setColorAt(0, QColor(255, 255, 255, 100 - i * 30))
            gradient.setColorAt(0.5, QColor(255, 255, 255, 50 - i * 20))
            gradient.setColorAt(1, QColor(255, 255, 255, 0))
            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            
            sweep_width = 45
            start_angle = math.degrees(self.angle) - 90 - sweep_width/2 - i * 15
            
            path = QPainterPath()
            path.moveTo(cx, cy)
            path.arcTo(QRectF(cx - self.radius, cy - self.radius, 
                             self.radius * 2, self.radius * 2),
                      start_angle, sweep_width)
            path.lineTo(cx, cy)
            painter.drawPath(path)
        
        # Random "contacts" that appear on radar
        num_contacts = int(self.timeline_position * 12)
        for i in range(num_contacts):
            contact_angle = (i * 30 + self.angle) % (2 * math.pi)
            contact_r = self.radius * (0.3 + (i % 3) * 0.25)
            contact_x = cx + int(contact_r * math.cos(contact_angle))
            contact_y = cy + int(contact_r * math.sin(contact_angle))
            
            painter.setBrush(white)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(contact_x, contact_y), 3, 3)
            
            # Contact ring
            painter.setPen(QPen(QColor(255, 255, 255, 150), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(contact_x, contact_y), 8, 8)
        
        self.draw_pointer_and_data(painter, cx, cy)
    
    def draw_style_mechanical(self, painter, cx, cy):
        """Style 4: Intricate mechanical clockwork gauge"""
        white = QColor(255, 255, 255)
        
        # Outer decorative rim with rivets
        painter.setPen(QPen(white, 8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), self.radius + 5, self.radius + 5)
        
        # Rivets around outer rim
        for i in range(16):
            angle = i * (2 * math.pi / 16)
            rivet_x = cx + int((self.radius + 5) * math.cos(angle))
            rivet_y = cy + int((self.radius + 5) * math.sin(angle))
            painter.setBrush(QColor(200, 200, 200))
            painter.setPen(QPen(white, 1))
            painter.drawEllipse(QPointF(rivet_x, rivet_y), 4, 4)
        
        # Inner mechanical ring
        painter.setPen(QPen(QColor(255, 255, 255, 180), 4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), self.radius - 15, self.radius - 15)
        
        # Fine tick marks (like a precision gauge)
        for i in range(120):
            angle = -math.pi/2 + (i / 120) * 2 * math.pi
            if i % 10 == 0:
                # Major tick
                r1 = self.radius - 5
                r2 = self.radius - 25
                painter.setPen(QPen(white, 3))
            elif i % 5 == 0:
                # Medium tick
                r1 = self.radius - 5
                r2 = self.radius - 18
                painter.setPen(QPen(white, 2))
            else:
                # Minor tick
                r1 = self.radius - 5
                r2 = self.radius - 12
                painter.setPen(QPen(QColor(255, 255, 255, 150), 1))
            
            x1 = cx + int(r1 * math.cos(angle))
            y1 = cy + int(r1 * math.sin(angle))
            x2 = cx + int(r2 * math.cos(angle))
            y2 = cy + int(r2 * math.sin(angle))
            painter.drawLine(x1, y1, x2, y2)
        
        # Progress arc (thick mechanical arm)
        painter.setPen(QPen(QColor(255, 255, 255, 220), 10))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(QRectF(cx - self.radius + 35, cy - self.radius + 35,
                              (self.radius - 35) * 2, (self.radius - 35) * 2),
                       -90 * 16, int(self.timeline_position * 360 * 16))
        
        # Gear teeth effect on inner circle
        num_teeth = 24
        for i in range(num_teeth):
            angle = i * (2 * math.pi / num_teeth)
            r1 = self.radius - 15
            r2 = self.radius - 20
            x1 = cx + int(r1 * math.cos(angle))
            y1 = cy + int(r1 * math.sin(angle))
            x2 = cx + int(r2 * math.cos(angle))
            y2 = cy + int(r2 * math.sin(angle))
            painter.setPen(QPen(QColor(255, 255, 255, 120), 2))
            painter.drawLine(x1, y1, x2, y2)
        
        self.draw_pointer_and_data(painter, cx, cy)
    
    def draw_pointer_and_data(self, painter, cx, cy):
        """Draw pointer and data (common to all styles)"""
        white = QColor(255, 255, 255)
        
        pointer_length = self.radius - 30
        pointer_x = cx + int(pointer_length * math.cos(self.angle))
        pointer_y = cy + int(pointer_length * math.sin(self.angle))
        
        painter.setPen(QPen(white, 3))
        painter.drawLine(cx, cy, pointer_x, pointer_y)
        
        painter.setBrush(white)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(pointer_x, pointer_y), 6, 6)
        
        painter.drawEllipse(QPointF(cx, cy), 8, 8)
        
        angle_deg = int(math.degrees(self.angle) % 360)
        timeline_pct = int(self.timeline_position * 100)
        
        # Use Monaco (native macOS font) or Courier as fallback
        font_value = QFont("Monaco", 14, QFont.Weight.Bold)
        painter.setFont(font_value)
        
        box_y = cy + self.radius + 45
        
        painter.setPen(QPen(white, 2))
        painter.setBrush(QColor(20, 20, 20, 200))
        painter.drawRect(cx - 100, box_y, 85, 35)
        painter.setPen(white)
        painter.drawText(QRectF(cx - 100, box_y, 85, 35),
                        Qt.AlignmentFlag.AlignCenter, f"{angle_deg:03d}°")
        
        painter.setPen(QPen(white, 2))
        painter.setBrush(QColor(20, 20, 20, 200))
        painter.drawRect(cx + 15, box_y, 85, 35)
        painter.setPen(white)
        painter.drawText(QRectF(cx + 15, box_y, 85, 35),
                        Qt.AlignmentFlag.AlignCenter, f"{timeline_pct:03d}%")

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
            model_complexity=0
        )
        
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.load_audio_files()
        
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        self.three_finger_gesture = False
        
        self.knob_active = False
        self.knob_center = None
        self.knob_angle = math.pi
        self.last_finger_angle = None
        
        self.gesture_stability_count = 0
        self.required_stability = 3
        
        self.timeline_position = 0.5
        
        self.wheel_mode = False
        self.touch_threshold = 50
        self.wheel_persistent = False
        self.show_wheel = False
        self.wheel_style = 0
        
        self.wheel_sound_played = False
        self.startup_sound_played = False
        self.last_knob_angle = self.knob_angle
        self.whoosh_cooldown = 0
        
        # New touch detection variables
        self.touch_sound_played = False
        self.is_touching = False
        self.touch_stability_count = 0
        self.required_touch_stability = 5  # Need 5 frames of touch to register
        self.red_flash_timer = 0
        self.red_flash_duration = 10  # frames
    
    def load_audio_files(self):
        """Load audio files if they exist"""
        self.sounds = {}
        audio_files = {
            'startup': 'music.mp3',
            'wheel': 'active.mp3',
            'powerdown': 'powerdown.mp3',
            'whoosh': 'whoosh.mp3',
            'active': 'charge.wav'
        }
        
        for sound_name, filename in audio_files.items():
            if os.path.exists(filename):
                try:
                    self.sounds[sound_name] = pygame.mixer.Sound(filename)
                except pygame.error:
                    pass
    
    def play_sound(self, sound_name):
        """Play a sound effect in a separate thread"""
        def play():
            if sound_name in self.sounds:
                try:
                    self.sounds[sound_name].play()
                except pygame.error:
                    pass
        
        sound_thread = threading.Thread(target=play)
        sound_thread.daemon = True
        sound_thread.start()
    
    def is_finger_extended(self, landmarks, tip_id, pip_id):
        """Check if a finger is extended"""
        return landmarks[tip_id].y < landmarks[pip_id].y
    
    def detect_three_finger_gesture(self, landmarks):
        """Detect three finger gesture"""
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]
        
        thumb_extended = abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x) * 0.8
        index_extended = self.is_finger_extended(landmarks, 8, 6)
        middle_extended = self.is_finger_extended(landmarks, 12, 10)
        ring_folded = landmarks[16].y > landmarks[14].y - 0.02
        pinky_folded = landmarks[20].y > landmarks[18].y - 0.02
        
        primary_fingers = thumb_extended and index_extended and middle_extended
        secondary_fingers = ring_folded and pinky_folded
        
        return primary_fingers and secondary_fingers
    
    def get_hand_center(self, landmarks):
        """Get the center point of the hand"""
        return landmarks[9]
    
    def calculate_finger_angle(self, landmarks):
        """Calculate angle of index finger"""
        hand_center = self.get_hand_center(landmarks)
        index_tip = landmarks[8]
        
        dx = index_tip.x - hand_center.x
        dy = index_tip.y - hand_center.y
        angle = math.atan2(dy, dx)
        return angle
    
    def update_knob_angle(self, current_angle):
        """Update knob rotation"""
        if self.last_finger_angle is not None:
            angle_diff = current_angle - self.last_finger_angle
            
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            elif angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            previous_knob_angle = self.knob_angle
            self.knob_angle += angle_diff * 2
            self.knob_angle = self.knob_angle % (2 * math.pi)
            
            angle_change = abs(self.knob_angle - previous_knob_angle)
            if angle_change > math.pi:
                angle_change = 2 * math.pi - angle_change
            
            if angle_change > 0.2 and self.whoosh_cooldown <= 0:
                self.play_sound('whoosh')
                self.whoosh_cooldown = 5
            
            self.timeline_position = (self.knob_angle / (2 * math.pi)) % 1.0
        
        self.last_finger_angle = current_angle
    
    def detect_touch(self, right_landmarks, left_landmarks, frame_w, frame_h):
        """Detect if fingertips touch knuckles"""
        right_index_tip = (int(right_landmarks[8].x * frame_w), int(right_landmarks[8].y * frame_h))
        right_middle_tip = (int(right_landmarks[12].x * frame_w), int(right_landmarks[12].y * frame_h))
        
        left_knuckles = []
        for knuckle_id in [5, 9, 13, 17]:
            knuckle_pos = (int(left_landmarks[knuckle_id].x * frame_w), int(left_landmarks[knuckle_id].y * frame_h))
            left_knuckles.append(knuckle_pos)
        
        for fingertip in [right_index_tip, right_middle_tip]:
            for knuckle in left_knuckles:
                distance = math.sqrt((fingertip[0] - knuckle[0])**2 + (fingertip[1] - knuckle[1])**2)
                if distance < self.touch_threshold:
                    return True
        
        return False
    
    def is_right_hand_on_right_side(self, landmarks, frame_width):
        """Check if right hand is on right side"""
        hand_center = self.get_hand_center(landmarks)
        hand_x = hand_center.x * frame_width
        return hand_x > frame_width * 0.6

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Futuristic Hand Tracker")
        self.setStyleSheet("background-color: #0a0a0a;")
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)
        
        self.wheel_widget = FuturisticWheel()
        self.wheel_widget.setStyleSheet("background: transparent;")
        
        self.tracker = HandTracker()
        self.cap = cv2.VideoCapture(0)
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        
        self.tracker.play_sound('startup')
        
        self.resize(1280, 720)
    
    def draw_hand_skeleton(self, frame, hand_landmarks):
        """Draw hand skeleton with MediaPipe"""
        self.mp_drawing.draw_landmarks(
            frame, 
            hand_landmarks, 
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=2)
        )
    
    def draw_circular_marker(self, frame, landmark, frame_w, frame_h, color=(255, 255, 255)):
        """Draw a circular marker at a landmark position"""
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        
        cv2.circle(frame, (x, y), 15, color, 2)
        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.circle(frame, (x, y), 3, (200, 200, 200), -1)
    
    def update_frame(self):
        ret, frame= self.cap.read()
        if not ret:
            return
        
        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.tracker.hands.process(rgb_frame)
        
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Background filters removed - showing original camera feed
        
        self.tracker.three_finger_gesture = False
        
        if results.multi_hand_landmarks and results.multi_handedness:
            right_hand_landmarks = None
            left_hand_landmarks = None
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                
                self.draw_hand_skeleton(frame, hand_landmarks)
                
                if hand_label == "Right":
                    right_hand_landmarks = hand_landmarks.landmark
                elif hand_label == "Left":
                    left_hand_landmarks = hand_landmarks.landmark
                    for knuckle_id in [5, 9, 13, 17]:
                        self.draw_circular_marker(frame, hand_landmarks.landmark[knuckle_id], frame_w, frame_h)
            
            # Draw right hand markers with red flash when touching
            if right_hand_landmarks:
                marker_color = (0, 0, 255) if self.tracker.red_flash_timer > 0 else (255, 255, 255)
                self.draw_circular_marker(frame, right_hand_landmarks[8], frame_w, frame_h, marker_color)
                self.draw_circular_marker(frame, right_hand_landmarks[12], frame_w, frame_h, marker_color)
            
            # Detect touch
            touch_detected = False
            if right_hand_landmarks and left_hand_landmarks:
                touch_detected = self.tracker.detect_touch(right_hand_landmarks, left_hand_landmarks, frame_w, frame_h)
            
            # Update touch stability counter
            if touch_detected:
                self.tracker.touch_stability_count += 1
            else:
                self.tracker.touch_stability_count = 0
            
            # Stable touch registered
            if self.tracker.touch_stability_count >= self.tracker.required_touch_stability:
                if not self.tracker.is_touching:
                    # Touch just became stable
                    self.tracker.is_touching = True
                    self.tracker.red_flash_timer = self.tracker.red_flash_duration
                    self.tracker.play_sound('active')
                    
                    if not self.tracker.wheel_mode:
                        # First touch - activate wheel mode
                        self.tracker.wheel_mode = True
                        self.tracker.wheel_persistent = True
                        self.tracker.show_wheel = False
                    else:
                        # In wheel mode - hide wheel and prepare for style change
                        self.tracker.show_wheel = False
                        self.tracker.wheel_style = (self.tracker.wheel_style + 1) % 5
                        self.wheel_widget.set_style(self.tracker.wheel_style)
                        self.tracker.wheel_sound_played = False
            else:
                # Not touching or not stable enough
                if self.tracker.is_touching and self.tracker.touch_stability_count == 0:
                    # Just released touch
                    self.tracker.is_touching = False
            
            # Show wheel when right hand is on right side (only if wheel_mode is active and wheel is hidden)
            if self.tracker.wheel_mode and right_hand_landmarks and not self.tracker.show_wheel:
                if self.tracker.is_right_hand_on_right_side(right_hand_landmarks, frame_w):
                    self.tracker.show_wheel = True
                    if not self.tracker.wheel_sound_played:
                        self.tracker.play_sound('wheel')
                        self.tracker.wheel_sound_played = True
            
            # Handle three-finger gesture for rotation
            if self.tracker.wheel_mode and self.tracker.show_wheel and right_hand_landmarks:
                gesture_detected = self.tracker.detect_three_finger_gesture(right_hand_landmarks)
                
                if gesture_detected:
                    self.tracker.gesture_stability_count = min(self.tracker.gesture_stability_count + 1, self.tracker.required_stability)
                    
                    if self.tracker.gesture_stability_count >= self.tracker.required_stability:
                        self.tracker.three_finger_gesture = True
                        self.tracker.knob_active = True
                        self.tracker.knob_center = self.tracker.get_hand_center(right_hand_landmarks)
                        
                        finger_angle = self.tracker.calculate_finger_angle(right_hand_landmarks)
                        self.tracker.update_knob_angle(finger_angle)
                else:
                    self.tracker.gesture_stability_count = max(self.tracker.gesture_stability_count - 1, 0)
                    self.tracker.three_finger_gesture = False
        else:
            # No hands detected - reset everything
            if self.tracker.wheel_mode:
                self.tracker.play_sound('powerdown')
                self.tracker.wheel_mode = False
                self.tracker.wheel_persistent = False
                self.tracker.show_wheel = False
                self.tracker.knob_active = False
                self.tracker.three_finger_gesture = False
                self.tracker.gesture_stability_count = 0
                self.tracker.last_finger_angle = None
                self.tracker.wheel_sound_played = False
                self.tracker.is_touching = False
                self.tracker.touch_stability_count = 0
        
        # Update timers
        if self.tracker.whoosh_cooldown > 0:
            self.tracker.whoosh_cooldown -= 1
        
        if self.tracker.red_flash_timer > 0:
            self.tracker.red_flash_timer -= 1
        
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_display.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        
        if self.tracker.wheel_persistent and self.tracker.show_wheel and self.tracker.knob_center:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            cx = int(self.tracker.knob_center.x * w)
            cy = int(self.tracker.knob_center.y * h)
            
            self.wheel_widget.resize(w, h)
            self.wheel_widget.set_data(self.tracker.knob_angle, self.tracker.timeline_position, 
                                      True, cx, cy)
            self.wheel_widget.render(painter)
            painter.end()
        
        self.video_label.setPixmap(pixmap)
        
        self.tracker.fps_counter += 1
        if time.time() - self.tracker.fps_start_time >= 1.0:
            self.tracker.current_fps = self.tracker.fps_counter
            self.tracker.fps_counter = 0
            self.tracker.fps_start_time = time.time()
    
    def closeEvent(self, event):
        self.cap.release()
        pygame.mixer.quit()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
