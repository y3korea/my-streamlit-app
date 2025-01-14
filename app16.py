import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import os
import pyttsx3
import threading
import base64
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

st.set_page_config(
    page_title="KBU x HealthnAI PoseEstimation Pro",
    page_icon="🤺",
    layout="wide",
    initial_sidebar_state="expanded"
)

logo_path = "logo/healthnai_kyungbok.png"
with open(logo_path, "rb") as f:
    logo_bytes = f.read()
encoded_logo = base64.b64encode(logo_bytes).decode()

# CSS 수정: 로고 크기 2배(80px), 전체 테두리를 명확히.
st.markdown(f"""
<style>
.top-bar {{
    position: fixed;
    top: 40px; 
    left: 50%;
    transform: translateX(-50%);
    z-index: 9999;
    display: flex;
    align-items: center;
    background: #ffffff; /* 완전 흰색 배경 */
    padding: 10px 20px;
    border: 2px solid #aaa; /* 더 두껍고 명확한 회색 테두리 */
    border-radius: 20px; /* 전체 테두리를 부드럽게 둥글림 */
    box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* 부드러운 그림자 */
}}

.top-bar-logo {{
    height: 80px; /* 로고 크기 2배 */
    margin-right: 20px; /* 로고와 텍스트 사이 간격 확대 */
}}

.top-bar-text {{
    font-size: 24px;
    font-weight: bold;
    color: #333;
    margin: 0;
    padding: 0;
}}

.main {{
    padding-top: 200px; /* 상단바와 컨텐츠 사이 여백을 늘림 */
}}

.stButton>button {{
    width: 100%;
    border-radius: 20px;
    height: 3em;
}}
.metric-card {{
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}}
.guide-area {{
    border: 2px solid #4CAF50;
    border-radius: 10px;
    padding: 10px;
    margin: 10px 0;
}}
.feedback-text {{
    font-size: 24px;
    font-weight: bold;
    text-align: center;
}}
.success-feedback {{
    color: #4CAF50;
}}
.warning-feedback {{
    color: #ff9800;
}}
.references {{
    color: #0000FF;
    font-size: 14px;
    margin-top: 100px;
}}
</style>

<div class='top-bar'>
    <img src='data:image/png;base64,{encoded_logo}' class='top-bar-logo'/>
    <span class='top-bar-text'>PoseEstimation Pro</span>
</div>
""", unsafe_allow_html=True)

# 이하 로직은 기존과 동일
import json
from datetime import datetime

@dataclass
class SquatStandards:
    STANDING = {'hip_angle': 172, 'knee_angle': 175, 'ankle_angle': 80, 'tolerance': 5}
    PARALLEL = {'hip_angle': 95, 'knee_angle': 90, 'ankle_angle': 70, 'tolerance': 8}
    DEPTH_LIMITS = {'min_hip': 50, 'max_knee': 135, 'ankle_range': (35,45)}
    FORM_CHECKS = {
        'knee_tracking': {'description': '(knee-toe alignment)', 'tolerance': 12},
        'back_angle': {'min': 50, 'max': 85},
        'weight_distribution': {'front': 0.45, 'back':0.55}
    }

class VoiceFeedback:
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            if voices:
                self.engine.setProperty('voice', voices[0].id)
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"Failed to initialize voice engine: {str(e)}")
            self.engine = None
        self.speaking_thread = None
        self.last_feedback = ""

    def speak(self, text: str):
        if not self.engine:
            return
        if text == self.last_feedback:
            return
        self.last_feedback = text
        def speak_worker():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Voice feedback error: {str(e)}")
        try:
            if self.speaking_thread and self.speaking_thread.is_alive():
                if hasattr(self.engine, 'stop'):
                    self.engine.stop()
                self.speaking_thread.join(timeout=1.0)
            self.speaking_thread = threading.Thread(target=speak_worker)
            self.speaking_thread.start()
        except Exception as e:
            print(f"Thread error: {str(e)}")

    def cleanup(self):
        try:
            if self.speaking_thread and self.speaking_thread.is_alive():
                if hasattr(self.engine, 'stop'):
                    self.engine.stop()
                self.speaking_thread.join(timeout=1.0)
            if self.engine:
                self.engine.stop()
        except Exception as e:
            print(f"Cleanup error: {str(e)}")

class SquatGuide:
    def __init__(self):
        self.guide_points = []
        self.target_area = None
        self.calibration_complete = False
    def draw_guide_area(self, frame: np.ndarray, landmarks, target_area: dict = None) -> np.ndarray:
        height, width = frame.shape[:2]
        overlay = frame.copy()
        if target_area:
            hip = target_area['hip']
            knee = target_area['knee']
            ankle = target_area['ankle']
            hip_point = (int(hip.x * width), int(hip.y * height))
            knee_point = (int(knee.x * width), int(knee.y * height))
            seat_width = abs(hip_point[0] - knee_point[0]) * 1.2
            seat_height = abs(hip_point[1] - knee_point[1]) * 0.5
            seat_top_left = (int(hip_point[0] - seat_width / 2), int(hip_point[1]))
            seat_top_right = (int(hip_point[0] + seat_width / 2), int(hip_point[1]))
            seat_bottom_left = (int(hip_point[0] - seat_width / 2), int(hip_point[1] + seat_height))
            seat_bottom_right = (int(hip_point[0] + seat_width / 2), int(hip_point[1] + seat_height))
            backrest_height = seat_height * 1.5
            backrest_top_left = (seat_top_left[0], int(seat_top_left[1] - backrest_height))
            backrest_top_right = (seat_top_right[0], int(seat_top_right[1] - backrest_height))
            chair_points = np.array([
                backrest_top_left,
                backrest_top_right,
                seat_top_right,
                seat_bottom_right,
                seat_bottom_left,
                seat_top_left
            ], np.int32)
            cv2.polylines(overlay, [chair_points], True, (0,255,0), 2, cv2.LINE_AA)
            cv2.fillPoly(overlay, [chair_points], (0,255,0,64))
            cv2.addWeighted(overlay,0.3,frame,0.7,0,frame)
        return frame

class EnhancedSquatAnalysisEngine:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
        self.guide = SquatGuide()
        self.standards = SquatStandards()
        self.calibration_mode = True
        self.calibration_count = 0
        self.target_squat_area = None
        self.rep_count = 0
        self.current_phase = "STANDING"
        self.voice = VoiceFeedback()
        self.data_path = os.path.join(os.path.expanduser('~'), 'squat_data')
        for folder in ['daily','weekly','monthly']:
            path = os.path.join(self.data_path,folder)
            os.makedirs(path, exist_ok=True)
    def _calculate_joint_angles(self, landmarks):
        def calculate_angle(a, b, c):
            a = np.array([a.x,a.y])
            b = np.array([b.x,b.y])
            c = np.array([c.x,c.y])
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[1])
            angle = np.abs(radians * 180.0 / np.pi)
            if angle >180.0:
                angle =360-angle
            return angle

        hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip_angle = calculate_angle(shoulder, hip, knee)
        knee_angle = calculate_angle(hip, knee, ankle)
        return {'hip':hip_angle,'knee':knee_angle}
    def _is_valid_squat_position(self, landmarks):
        angles = self._calculate_joint_angles(landmarks)
        hip_in_range = abs(angles['hip']-self.standards.PARALLEL['hip_angle'])<=self.standards.PARALLEL['tolerance']
        knee_in_range = abs(angles['knee']-self.standards.PARALLEL['knee_angle'])<=self.standards.PARALLEL['tolerance']
        return hip_in_range and knee_in_range
    def _draw_landmarks(self, image, landmarks):
        height,width = image.shape[:2]
        connections = self.mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            start_point = (int(landmarks[start_idx].x*width), int(landmarks[start_idx].y*height))
            end_point = (int(landmarks[end_idx].x*width), int(landmarks[end_idx].y*height))
            cv2.line(image, start_point, end_point, (0,255,0), 2)
        angles = self._calculate_joint_angles(landmarks)
        for idx, landmark in enumerate(landmarks):
            pos = (int(landmark.x*width), int(landmark.y*height))
            cv2.circle(image, pos, 5, (0,0,255), -1)
            if idx==self.mp_pose.PoseLandmark.LEFT_HIP.value:
                angle_text = f"Hip: {angles['hip']:.1f}° / Std: {self.standards.PARALLEL['hip_angle']}°"
                cv2.putText(image, angle_text,(pos[0]+10,pos[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            elif idx==self.mp_pose.PoseLandmark.LEFT_KNEE.value:
                angle_text = f"Knee: {angles['knee']:.1f}° / Std: {self.standards.PARALLEL['knee_angle']}°"
                cv2.putText(image, angle_text,(pos[0]+10,pos[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    def _analyze_squat_phase(self, angles, landmarks):
        result = {'is_correct':False, 'feedback':'','score':0}
        if angles['hip']>160:
            self.current_phase ="STANDING"
            if not hasattr(self,'prev_hip_height'):
                self.prev_hip_height = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
        elif angles['hip']<100:
            self.current_phase ="SQUAT"
        result['phase']=self.current_phase
        if self.current_phase=="SQUAT":
            hip_deviation=abs(angles['hip']-self.standards.PARALLEL['hip_angle'])
            knee_deviation=abs(angles['knee']-self.standards.PARALLEL['knee_angle'])
            angle_score = max(0,100-(hip_deviation+knee_deviation))
            result['score']=angle_score
            if angle_score>=90:
                result['is_correct']=True
                result['feedback']="Good!"
            else:
                result['feedback']="Try again!"
            if not hasattr(self,'rep_counted'):
                self.rep_counted=False
            if angle_score>=90 and not self.rep_counted:
                self.rep_count+=1
                self.rep_counted=True
        elif self.current_phase=="STANDING":
            self.rep_counted=False
            result['feedback']="Ready"
            result['is_correct']=True
        return result
    def analyze_frame(self, frame):
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        results = self.pose.process(image)
        image.flags.writeable=True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        if not results.pose_landmarks:
            return image,{'is_correct':False,'feedback':"No pose detected. Please stand in frame.",'phase':'UNKNOWN'}
        self._draw_landmarks(image, results.pose_landmarks.landmark)
        if self.calibration_mode:
            return self._handle_calibration(image, results.pose_landmarks.landmark)
        return self._analyze_regular_squat(image, results.pose_landmarks.landmark)
    def _handle_calibration(self,image,landmarks):
        if self.calibration_count>=5:
            self.calibration_mode=False
            self.target_squat_area = self._calculate_target_area(landmarks)
            self.voice.speak("Calibration complete")
            return image,{'is_correct':True,'feedback':"Calibration complete! Ready.",'phase':'CALIBRATED'}
        if self._is_valid_squat_position(landmarks):
            self.calibration_count+=1
            self.voice.speak(f"Good form {self.calibration_count}")
            feedback=f"Calibration rep {self.calibration_count}/5 recorded"
        else:
            feedback="Please use proper squat form for calibration"
        image = self.guide.draw_guide_area(image,landmarks,self.target_squat_area)
        return image,{'is_correct':False,'feedback':feedback,'phase':'CALIBRATING'}
    def _analyze_regular_squat(self,image,landmarks):
        angles=self._calculate_joint_angles(landmarks)
        in_target_area=self._is_in_target_area(landmarks)
        phase_feedback=self._analyze_squat_phase(angles,landmarks)
        image=self.guide.draw_guide_area(image,landmarks,self.target_squat_area)
        if in_target_area and phase_feedback['is_correct']:
            self.voice.speak("Good!")
            feedback="Good!"
        else:
            self.voice.speak("Try again!")
            feedback="Try again!"
        return image,{'is_correct': in_target_area and phase_feedback['is_correct'],'feedback':feedback,'phase':self.current_phase,'angles':angles,'score':phase_feedback.get('score',0)}
    def _calculate_target_area(self,landmarks):
        self.guide.target_area={
            'hip':landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            'knee':landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            'ankle':landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        }
        return self.guide.target_area
    def _is_in_target_area(self,landmarks):
        if not self.guide.target_area:
            return False
        current_hip=landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        target_hip=self.guide.target_area['hip']
        tolerance=0.05
        hip_diff=np.hypot(current_hip.x-target_hip.x,current_hip.y-target_hip.y)
        return hip_diff<tolerance
    def cleanup(self):
        self.voice.cleanup()

def main():
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer=EnhancedSquatAnalysisEngine()
    with st.sidebar:
        st.header("Settings ⚙️")
        show_skeleton=st.checkbox("Show Skeleton 🦴",True)
        show_guide=st.checkbox("Show Guide Area 🎯",True)
        show_angles=st.checkbox("Show Joint Angles 📐",True)

    cols=st.columns(4)
    if cols[0].button("Start Calibration",use_container_width=True):
        st.session_state.analyzer=EnhancedSquatAnalysisEngine()
        st.info("Perform 5 squats for baseline.")
    if cols[1].button("Camera Check",use_container_width=True):
        st.session_state.camera_check=True
    if cols[2].button("Start Workout",use_container_width=True):
        st.session_state.workout_active=True
    if cols[3].button("Stop",use_container_width=True):
        if hasattr(st.session_state.analyzer,'cleanup'):
            st.session_state.analyzer.cleanup()
        st.session_state.workout_active=False

    FRAME_WINDOW=st.empty()
    progress_cols=st.columns(2)
    rep_count=progress_cols[0].empty()
    phase_indicator=progress_cols[1].empty()

    cap=cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            ret,frame=cap.read()
            if not ret:
                st.error("Failed to capture video frame")
                break
            processed_frame,feedback=st.session_state.analyzer.analyze_frame(frame)
            feedback_color=(0,255,0) if feedback['is_correct'] else (0,0,255)
            cv2.putText(processed_frame,feedback['feedback'],(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,feedback_color,2)
            rep_count.metric("Reps Completed 🔄",st.session_state.analyzer.rep_count)
            phase_indicator.info(f"Current Phase: {feedback['phase']}")
            FRAME_WINDOW.image(processed_frame,channels="BGR")
            if cv2.waitKey(1)&0xFF==27:
                break
    finally:
        cap.release()
        if hasattr(st.session_state.analyzer,'cleanup'):
            st.session_state.analyzer.cleanup()

    st.markdown("<div class='references'><b>References:</b><br>"
                "1. Escamilla RF. ...<br>"
                "2. Caterisano A, et al. ...<br>"
                "3. Behm DG, et al. ...<br>"
                "4. Schoenfeld BJ. ...<br>"
                "</div>",unsafe_allow_html=True)

if __name__=="__main__":
    main()