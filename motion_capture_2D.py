"""
MediaPipe 2D to 3D Pose Estimation Script
==========================================
Gets 2D coordinates using MediaPipe (like mp_webcam.py) and converts to 3D 
using the pretrained LinearModel. Saves both 2D and 3D coordinates to Excel.

Controls:
  - Press 's' to start/pause recording
  - Press 'q' to quit and save data
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
import os
from datetime import datetime
from scipy.signal import savgol_filter

# Import bone transformation
from CTransform import change_skele_length

# ============ CONFIGURATION VARIABLES ============
CAMERA_INDEX = 2
# Use Intel RealSense depth stream for Z values when True. Requires `pyrealsense2`.
USE_REALSENSE_DEPTH = True

# Output folder for saving files
OUTPUT_FOLDER = "output_excel"

# Output Excel file prefix (timestamp will be added automatically)
# OUTPUT_EXCEL_PREFIX = "pose_2d_3d_coordinates"
OUTPUT_EXCEL_PREFIX = "pose_2D"

# Recording duration in seconds (set to None for manual stop with 'q' key)
RECORDING_DURATION = 8

# Frame rate for processing (lower = less data points)
PROCESS_EVERY_N_FRAMES = 1

# Show visualization window
SHOW_VISUALIZATION = True

# MediaPipe detection confidence thresholds
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Path to the pretrained 2D-to-3D model
MODEL_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "ckpt_best.pth.tar")

# Apply bone length normalization from CTransform
# When True, skeleton bone lengths are normalized to standard lengths before 3D conversion
APPLY_BONE_TRANSFORM = True


#   'uniform': Simple uniform scaling - all joints scaled equally relative to hip
#   'fk': Forward kinematics approach - preserves joint angles while adjusting bone lengths
BONE_NORM_METHOD = 'fk'  # Change to 'fk' to use forward kinematics approach
#   'fk_chain': CTransform-like 2D chain normalization lifted to 3D
#                (runs `change_skele_length` on XY then scales Z accordingly)

# ============ NOISE FILTERING ============
# Enable Savitzky-Golay filtering for smoothing 3D motion capture data
ENABLE_SMOOTHING = False
# Savitzky-Golay filter parameters
# window_length: must be odd and <= data length; use ~11 for typical data
# polyorder: polynomial order (1-3); 2 is good balance of smoothing vs preservation
SAVGOL_WINDOW = 11
SAVGOL_POLYORDER = 2
# ================================================


# ==================== Model classes ====================
class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)
        return x + y


class LinearModel(nn.Module):
    def __init__(self, linear_size=1024, num_stage=2, p_dropout=0.5):
        super(LinearModel, self).__init__()
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.input_size = 17 * 2  # 17 joints x 2D coords
        self.output_size = 17 * 3  # 17 joints x 3D coords

        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        self.linear_stages = nn.ModuleList([
            Linear(self.linear_size, self.p_dropout) for _ in range(num_stage)
        ])
        self.w2 = nn.Linear(self.linear_size, self.output_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        for stage in self.linear_stages:
            y = stage(y)
        y = self.w2(y)
        return y


class Pose2Dto3D:
    """Converts MediaPipe 2D landmarks to 3D pose using the pretrained model."""
    
    def __init__(self, checkpoint_path):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.model = LinearModel().to(self.device)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        print("3D model loaded successfully")

    def mediapipe_to_h36m_17(self, landmarks, frame_width, frame_height):
        """
        Convert 33 MediaPipe landmarks to 17 Human3.6M joints.
        Returns a numpy array of shape (17, 2) with pixel coordinates.
        """
        # Compute virtual joints
        hip_center = np.array([
            (landmarks[23].x + landmarks[24].x) / 2 * frame_width,
            (landmarks[23].y + landmarks[24].y) / 2 * frame_height
        ])
        chest = np.array([
            (landmarks[11].x + landmarks[12].x) / 2 * frame_width,
            (landmarks[11].y + landmarks[12].y) / 2 * frame_height
        ])
        spine = (hip_center + chest) / 2
        mouth = np.array([
            (landmarks[9].x + landmarks[10].x) / 2 * frame_width,
            (landmarks[9].y + landmarks[10].y) / 2 * frame_height
        ])
        neck = (chest + mouth) / 2
        
        # Build 17-joint array in Human3.6M order
        # 0:Hip, 1:RHip, 2:RKnee, 3:RAnkle, 4:LHip, 5:LKnee, 6:LAnkle,
        # 7:Spine, 8:Chest, 9:Neck, 10:Head,
        # 11:LShoulder, 12:LElbow, 13:LWrist, 14:RShoulder, 15:RElbow, 16:RWrist
        joints_17 = np.array([
            hip_center,                                                          # 0: Hip (center)
            [landmarks[24].x * frame_width, landmarks[24].y * frame_height],    # 1: RHip
            [landmarks[26].x * frame_width, landmarks[26].y * frame_height],    # 2: RKnee
            [landmarks[28].x * frame_width, landmarks[28].y * frame_height],    # 3: RAnkle
            [landmarks[23].x * frame_width, landmarks[23].y * frame_height],    # 4: LHip
            [landmarks[25].x * frame_width, landmarks[25].y * frame_height],    # 5: LKnee
            [landmarks[27].x * frame_width, landmarks[27].y * frame_height],    # 6: LAnkle
            spine,                                                               # 7: Spine
            chest,                                                               # 8: Chest
            neck,                                                                # 9: Neck
            [landmarks[0].x * frame_width, landmarks[0].y * frame_height],      # 10: Head (nose)
            [landmarks[11].x * frame_width, landmarks[11].y * frame_height],    # 11: LShoulder
            [landmarks[13].x * frame_width, landmarks[13].y * frame_height],    # 12: LElbow
            [landmarks[15].x * frame_width, landmarks[15].y * frame_height],    # 13: LWrist
            [landmarks[12].x * frame_width, landmarks[12].y * frame_height],    # 14: RShoulder
            [landmarks[14].x * frame_width, landmarks[14].y * frame_height],    # 15: RElbow
            [landmarks[16].x * frame_width, landmarks[16].y * frame_height],    # 16: RWrist
        ])
        
        return joints_17

    def predict_3d(self, landmarks, frame_width, frame_height, apply_bone_transform=False):
        """
        Convert MediaPipe landmarks to 3D pose.
        
        Args:
            landmarks: MediaPipe pose landmarks
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            apply_bone_transform: If True, apply bone length normalization from CTransform
        
        Returns the full 17-joint 3D pose as numpy array (17, 3).
        """
        # Get 17 2D joints in pixel coordinates
        joints_2d = self.mediapipe_to_h36m_17(landmarks, frame_width, frame_height)
        
        # Center on hip
        hip = joints_2d[0].copy()
        joints_2d_centered = joints_2d - hip
        
        # Apply bone length normalization if enabled
        if apply_bone_transform:
            x_coords = joints_2d_centered[:, 0].tolist()
            y_coords = joints_2d_centered[:, 1].tolist()
            try:
                x_normalized, y_normalized = change_skele_length(x_coords, y_coords)
                joints_2d_centered = np.array(list(zip(x_normalized, y_normalized)))
            except Exception as e:
                # If transformation fails, use original coordinates
                print("Transform failed")
                pass
        
        # Flatten and convert to tensor
        input_2d = joints_2d_centered.flatten().reshape(1, -1).astype(np.float32)
        input_tensor = torch.tensor(input_2d).to(self.device)
        
        with torch.no_grad():
            output_3d = self.model(input_tensor)
        
        pose_3d = output_3d.cpu().numpy().reshape(17, 3)
        return pose_3d


def draw_body_landmarks(frame, pose_landmarks, mp_pose):
    """
    Draw only body landmarks, excluding face and hand finger points.
    """
    if pose_landmarks is None:
        return
    
    h, w, _ = frame.shape
    landmarks = pose_landmarks.landmark
    
    # Body landmark indices
    body_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    
    # Body connections
    body_connections = [
        (11, 12),  # Shoulders
        (11, 13),  # Left shoulder to left elbow
        (13, 15),  # Left elbow to left wrist
        (12, 14),  # Right shoulder to right elbow
        (14, 16),  # Right elbow to right wrist
        (11, 23),  # Left shoulder to left hip
        (12, 24)   # Right shoulder to right hip
    ]
    
    # Draw connections (lines)
    for start_idx, end_idx in body_connections:
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        
        if start.visibility > 0.5 and end.visibility > 0.5:
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
    
    # Draw landmarks (circles)
    for idx in body_indices:
        landmark = landmarks[idx]
        if landmark.visibility > 0.5:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)


def apply_savgol_smoothing(data_dict, window_length, polyorder):

    """
    Apply Savitzky-Goyal filter to smooth motion capture data.
    Operates on all 3D coordinate columns in the data dictionary.

    Args:
        data_dict: Dict with lists of coordinates {col_name: [values, ...], ...}
        window_length: Window size for filter (must be odd, <= data length)
        polyorder: Polynomial order (1-3 recommended)

    Returns:
        Smoothed dictionary with same structure as input
    """
    smoothed = {}
    n_samples = len(next(iter(data_dict.values())))

    # Only apply filter if we have enough data points
    if n_samples < window_length:
        print(f"Warning: Only {n_samples} samples, need at least {window_length} for Savitzky-Golay filter.")
        print("Returning unsmoothed data.")
        return data_dict

    for col_name, values in data_dict.items():
        if col_name == 'Timestamp':
            smoothed[col_name] = values
        else:
            # Apply Savitzky-Golay filter to numeric columns
            try:
                smoothed[col_name] = savgol_filter(values, window_length, polyorder).tolist()
            except Exception as e:
                print(f"Warning: Could not smooth {col_name}, keeping original. Error: {e}")
                smoothed[col_name] = values

    return smoothed


def normalize_skeleton_3d(pose_3d):
    """
    Normalize 3D skeleton to standard anatomical proportions.
    Scales all joints uniformly so bone lengths match standard skeleton.
    
    Args:
        pose_3d: 3D pose as (17, 3) numpy array
    
    Returns:
        Normalized pose_3d as (17, 3) numpy array
    """
    # Standard skeleton bone lengths from Human3.6M dataset (in mm for reference)
    standard_lengths = np.array([0.0, 132.95, 442.89, 454.21, 132.95, 442.89, 454.21, 233.38, 257.08, 121.13, 115,
                                 151.03, 278.88, 251.73, 151.03, 278.88, 251.73])
    
    # Key bone indices to measure (parent -> child)
    bone_pairs = [
        (0, 1),   # Hip -> RHip
        (1, 2),   # RHip -> RKnee
        (2, 3),   # RKnee -> RAnkle
        (0, 4),   # Hip -> LHip
        (4, 5),   # LHip -> LKnee
        (5, 6),   # LKnee -> LAnkle
        (0, 7),   # Hip -> Spine
        (7, 8),   # Spine -> Chest
        (8, 9),   # Chest -> Neck
        (9, 10),  # Neck -> Head
        (8, 11),  # Chest -> LShoulder
        (11, 12), # LShoulder -> LElbow
        (12, 13), # LElbow -> LWrist
        (8, 14),  # Chest -> RShoulder
        (14, 15), # RShoulder -> RElbow
        (15, 16), # RElbow -> RWrist
    ]
    
    # Calculate actual bone lengths
    actual_lengths = []
    for p1, p2 in bone_pairs:
        if p1 < len(pose_3d) and p2 < len(pose_3d):
            bone_vec = pose_3d[p2] - pose_3d[p1]
            actual_length = np.linalg.norm(bone_vec)
            actual_lengths.append(actual_length)
        else:
            actual_lengths.append(0)
    
    actual_lengths = np.array(actual_lengths)
    
    # Compute scale factor (avoid division by zero)
    valid_lengths = actual_lengths[actual_lengths > 0]
    standard_valid = standard_lengths[1:len(actual_lengths)+1][actual_lengths > 0]  # Skip first zero
    
    if len(valid_lengths) > 0 and len(standard_valid) > 0:
        # Average scale factor across all valid bones
        scale_factors = standard_valid / valid_lengths
        scale_factor = np.nanmean(scale_factors[np.isfinite(scale_factors)])
    else:
        scale_factor = 1.0
    
    # Apply scaling relative to hip (joint 0)
    normalized_pose = pose_3d.copy()
    hip_pos = pose_3d[0]
    
    for i in range(len(pose_3d)):
        offset = pose_3d[i] - hip_pos
        normalized_pose[i] = hip_pos + offset * scale_factor
    
    return normalized_pose

def normalize_skeleton_3d_fk(pose_3d):
    """
    Normalize 3D skeleton using forward kinematics approach (like CTransform.py).
    Preserves joint angles/directions while adjusting bone lengths to standard proportions.
    
    This approach maintains the pose's shape by preserving the direction of each bone
    while scaling the bone length to match standard skeleton proportions.
    
    Args:
        pose_3d: 3D pose as (17, 3) numpy array
    
    Returns:
        Normalized pose_3d as (17, 3) numpy array
    """
    # Standard skeleton bone lengths (same as normalize_skeleton_3d)
    standard_lengths = np.array([0.0, 132.95, 442.89, 454.21, 132.95, 442.89, 454.21, 233.38, 257.08, 121.13, 115,
                                 151.03, 278.88, 251.73, 151.03, 278.88, 251.73])
    
    # Bone parent-child relationships (same as normalize_skeleton_3d)
    bone_pairs = [
        (0, 1),   # Hip -> RHip
        (1, 2),   # RHip -> RKnee
        (2, 3),   # RKnee -> RAnkle
        (0, 4),   # Hip -> LHip
        (4, 5),   # LHip -> LKnee
        (5, 6),   # LKnee -> LAnkle
        (0, 7),   # Hip -> Spine
        (7, 8),   # Spine -> Chest
        (8, 9),   # Chest -> Neck
        (9, 10),  # Neck -> Head
        (8, 11),  # Chest -> LShoulder
        (11, 12), # LShoulder -> LElbow
        (12, 13), # LElbow -> LWrist
        (8, 14),  # Chest -> RShoulder
        (14, 15), # RShoulder -> RElbow
        (15, 16), # RElbow -> RWrist
    ]
    
    normalized_pose = np.zeros_like(pose_3d)
    normalized_pose[0] = pose_3d[0]  # Hip stays at origin
    
    # Process each bone sequentially (forward kinematics)
    for bone_idx, (parent_idx, child_idx) in enumerate(bone_pairs):
        # Get parent position (already normalized)
        parent_pos = normalized_pose[parent_idx]
        
        # Get original bone direction vector
        bone_vec = pose_3d[child_idx] - pose_3d[parent_idx]
        bone_length = np.linalg.norm(bone_vec)
        
        # Standard bone length for this bone
        std_length = standard_lengths[bone_idx + 1]
        
        # Compute new child position preserving direction
        if bone_length > 1e-6:  # Avoid division by zero
            # Normalize direction and apply standard length
            direction = bone_vec / bone_length
            normalized_pose[child_idx] = parent_pos + direction * std_length
        else:
            # If bone length is near zero, place at parent position
            normalized_pose[child_idx] = parent_pos
    
    return normalized_pose

def normalize_skeleton_3d_ctransform(pose_3d):
    """
    CTransform-like normalization for 3D poses.
    - Projects XY to 2D centered at hip, calls `change_skele_length` to get
      CTransform's 2D chain-normalized XY positions, then lifts back to 3D
      by scaling Z offsets per-bone according to the XY scaling.

    This preserves the 2D chained orientations produced by `change_skele_length`
    while producing a consistent Z that follows the same local scaling.
    """
    pose = np.asarray(pose_3d)
    if pose.shape != (17, 3):
        return pose_3d.copy()

    hip = pose[0].copy()

    # Prepare 2D lists expected by change_skele_length (centered on hip)
    x = (pose[:, 0] - hip[0]).tolist()
    y = (pose[:, 1] - hip[1]).tolist()

    try:
        x_new, y_new = change_skele_length(x, y)
    except Exception as e:
        print("CTransform 2D normalization failed:", e)
        return pose.copy()

    normalized = np.zeros_like(pose)
    normalized[0] = hip

    # Place new XY positions (add hip offset back)
    for i in range(len(x_new)):
        normalized[i, 0] = hip[0] + float(x_new[i])
        normalized[i, 1] = hip[1] + float(y_new[i])

    # Define bone parent-child pairs consistent with other normalizers
    bone_pairs = [
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8),
        (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)
    ]

    # Compute Z by scaling original per-bone Z offsets according to XY scaling
    orig_xy = pose[:, :2]
    for parent_idx, child_idx in bone_pairs:
        parent_orig_xy = orig_xy[parent_idx]
        child_orig_xy = orig_xy[child_idx]
        orig_dist = np.linalg.norm(child_orig_xy - parent_orig_xy)

        parent_new_xy = normalized[parent_idx, :2]
        child_new_xy = normalized[child_idx, :2]
        new_dist = np.linalg.norm(child_new_xy - parent_new_xy)

        scale = new_dist / (orig_dist if orig_dist > 1e-6 else 1.0)

        z_offset = pose[child_idx, 2] - pose[parent_idx, 2]
        normalized[child_idx, 2] = normalized[parent_idx, 2] + z_offset * scale

    # For any joints not covered above (shouldn't happen), copy Z from original
    for i in range(len(normalized)):
        if normalized[i, 2] == 0 and pose[i, 2] != 0:
            normalized[i, 2] = pose[i, 2]

    return normalized


def main():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    
    # Data storage lists
    timestamps = []
    
    # Data storage lists for 3D coordinates only (right hand only)
    right_shoulder_3d_x, right_shoulder_3d_y, right_shoulder_3d_z = [], [], []
    right_elbow_3d_x, right_elbow_3d_y, right_elbow_3d_z = [], [], []
    right_wrist_3d_x, right_wrist_3d_y, right_wrist_3d_z = [], [], []

    # Data storage lists for normalized 3D coordinates (scale-invariant, right hand only)
    right_shoulder_3d_norm_x, right_shoulder_3d_norm_y, right_shoulder_3d_norm_z = [], [], []
    right_elbow_3d_norm_x, right_elbow_3d_norm_y, right_elbow_3d_norm_z = [], [], []
    right_wrist_3d_norm_x, right_wrist_3d_norm_y, right_wrist_3d_norm_z = [], [], []

    # #NORM 2
    # right_shoulder_3d_norm2_x, right_shoulder_3d_norm2_y, right_shoulder_3d_norm2_z = [], [], []
    # right_elbow_3d_norm2_x, right_elbow_3d_norm2_y, right_elbow_3d_norm2_z = [], [], []
    # right_wrist_3d_norm2_x, right_wrist_3d_norm2_y, right_wrist_3d_norm2_z = [], [], []

    # Initialize 2D-to-3D converter
    print("Loading 2D-to-3D model...")
    converter = Pose2Dto3D(MODEL_CHECKPOINT_PATH)

    # Open camera (either OpenCV capture or RealSense pipeline)
    realsense_pipeline = None
    realsense_align = None
    if USE_REALSENSE_DEPTH:
        try:
            import pyrealsense2 as rs
        except Exception as e:
            print("Error: pyrealsense2 not installed or failed to import.")
            print("Install librealsense / pyrealsense2 or set USE_REALSENSE_DEPTH=False.")
            return

        print("Starting Intel RealSense pipeline (color+depth)...")
        realsense_pipeline = rs.pipeline()
        cfg = rs.config()
        # Let the device choose resolution; common defaults below
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = realsense_pipeline.start(cfg)
        realsense_align = rs.align(rs.stream.color)
        # Get intrinsics from depth stream later inside loop when frames are available
        # Use placeholder resolution until first frames arrive
        frame_width, frame_height = 640, 480
        print(f"RealSense started. Approx resolution: {frame_width}x{frame_height}")
    else:
        print(f"Opening camera at index {CAMERA_INDEX}...")
        cap = cv2.VideoCapture(CAMERA_INDEX)

        if not cap.isOpened():
            print(f"Error: Could not open camera at index {CAMERA_INDEX}")
            print("Try changing CAMERA_INDEX to a different value (0, 1, 2, etc.)")
            return

        # Get frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened successfully. Resolution: {frame_width}x{frame_height}")
    print("Press 'q' to stop recording and save data.")
    print("Press 's' to start/pause recording.")

    # Initialize pose model
    pose = mp_pose.Pose(
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        model_complexity=1,
        smooth_landmarks= True
    )

    start_time = time.time()
    frame_count = 0
    is_recording = True

    try:
        while True:
            if USE_REALSENSE_DEPTH:
                # Get frames from RealSense
                frames = realsense_pipeline.wait_for_frames()
                aligned_frames = realsense_align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    print("Warning: Incomplete RealSense frames, skipping")
                    continue
                frame = np.asanyarray(color_frame.get_data())
                # update frame dims from actual frames
                frame_width = color_frame.get_width()
                frame_height = color_frame.get_height()
                depth_intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break

            frame_count += 1

            # Skip frames based on PROCESS_EVERY_N_FRAMES

            # Check recording duration
            elapsed_time = time.time() - start_time
            if RECORDING_DURATION is not None and elapsed_time > RECORDING_DURATION:
                print(f"Recording duration of {RECORDING_DURATION}s reached.")
                break

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            # Process frame with MediaPipe Pose
            results = pose.process(rgb_frame)

            # Convert back to BGR for visualization
            rgb_frame.flags.writeable = True
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks and is_recording:
                landmarks = results.pose_landmarks.landmark

                current_time = time.time() - start_time
                timestamps.append(current_time)

                # ===== 3D Coordinates =====
                # By default use model prediction for full 3D pose
                pose_3d = converter.predict_3d(landmarks, frame_width, frame_height, APPLY_BONE_TRANSFORM)

                # If RealSense depth is enabled, replace 3D coordinates for body joints
                # by computing X,Y,Z from depth stream + camera intrinsics (camera coordinate system in meters).
                # On fallback, convert model output to camera coordinates.
                if USE_REALSENSE_DEPTH:
                    pose_3d_from_depth = np.copy(pose_3d)  # Start with model as fallback
                    # Joints of interest (Human3.6M indices)
                    depth_joints = [11, 12, 13, 14, 15, 16]
                    intr = depth_intrinsics
                    
                    for j in depth_joints:
                        # Map back to corresponding MediaPipe landmark indices
                        # Human3.6M indices mapping in mediapipe_to_h36m_17:
                        # 11->11(LShoulder), 12->13(LElbow), 13->15(LWrist), 14->12(RShoulder), 15->14(RElbow), 16->16(RWrist)
                        mp_index_map = {11:11, 12:13, 13:15, 14:12, 15:14, 16:16}
                        mp_idx = mp_index_map.get(j, None)
                        if mp_idx is None:
                            continue
                        
                        lm = landmarks[mp_idx]
                        u = int(lm.x * frame_width)
                        v = int(lm.y * frame_height)
                        
                        # Check bounds
                        if not (0 <= u < frame_width and 0 <= v < frame_height):
                            continue
                        
                        depth_m = depth_frame.get_distance(u, v)
                        if depth_m > 0 and not np.isnan(depth_m):
                            # Valid depth from RealSense (converted everything to mm)
                            x_cam = ((u - intr.ppx) / intr.fx * depth_m)*1000 
                            y_cam = ((v - intr.ppy) / intr.fy * depth_m)*1000
                            z_cam = depth_m*1000
                        else:
                            # Fallback: convert model output to camera coordinates
                            # Use 2D pixel position and model's Z, compute X,Y using intrinsics
                            model_z = pose_3d[j, 2]  # Use model's Z value
                            x_cam = (u - intr.ppx) / intr.fx * model_z
                            y_cam = (v - intr.ppy) / intr.fy * model_z
                            z_cam = model_z
                        
                        pose_3d_from_depth[j, 0] = x_cam
                        pose_3d_from_depth[j, 1] = y_cam
                        pose_3d_from_depth[j, 2] = z_cam

                    pose_3d = pose_3d_from_depth

                # 3D indices: 14:RShoulder, 15:RElbow, 16:RWrist (right hand only)
                right_shoulder_3d_x.append(pose_3d[14][0])
                right_shoulder_3d_y.append(pose_3d[14][1])
                right_shoulder_3d_z.append(pose_3d[14][2])

                right_elbow_3d_x.append(pose_3d[15][0])
                right_elbow_3d_y.append(pose_3d[15][1])
                right_elbow_3d_z.append(pose_3d[15][2])

                right_wrist_3d_x.append(pose_3d[16][0])
                right_wrist_3d_y.append(pose_3d[16][1])
                right_wrist_3d_z.append(pose_3d[16][2])

                # ===== Normalized 3D Coordinates (scale-invariant, right hand only) =====
                # Apply bone length normalization for cross-person comparison
                if APPLY_BONE_TRANSFORM:
                    # Use selected normalization method
                    if BONE_NORM_METHOD == 'fk':
                        pose_3d_normalized = normalize_skeleton_3d_fk(pose_3d)
                    elif BONE_NORM_METHOD == 'fk_chain':
                        pose_3d_normalized = normalize_skeleton_3d_ctransform(pose_3d)
                    else:  # Default to 'uniform'
                        pose_3d_normalized = normalize_skeleton_3d(pose_3d)
                    
                    
                    right_shoulder_3d_norm_x.append(pose_3d_normalized[14][0])
                    right_shoulder_3d_norm_y.append(pose_3d_normalized[14][1])
                    right_shoulder_3d_norm_z.append(pose_3d_normalized[14][2])

                    right_elbow_3d_norm_x.append(pose_3d_normalized[15][0])
                    right_elbow_3d_norm_y.append(pose_3d_normalized[15][1])
                    right_elbow_3d_norm_z.append(pose_3d_normalized[15][2])

                    right_wrist_3d_norm_x.append(pose_3d_normalized[16][0])
                    right_wrist_3d_norm_y.append(pose_3d_normalized[16][1])
                    right_wrist_3d_norm_z.append(pose_3d_normalized[16][2])

                    #NORM 2---------------------------------------------
                    # pose3d_norm2 = normalize_skeleton_3d_fk(pose_3d)
                    # right_shoulder_3d_norm2_x.append(pose3d_norm2[14][0])
                    # right_shoulder_3d_norm2_y.append(pose3d_norm2[14][1])
                    # right_shoulder_3d_norm2_z.append(pose3d_norm2[14][2])

                    # right_elbow_3d_norm2_x.append(pose3d_norm2[15][0])
                    # right_elbow_3d_norm2_y.append(pose3d_norm2[15][1])
                    # right_elbow_3d_norm2_z.append(pose3d_norm2[15][2])

                    # right_wrist_3d_norm2_x.append(pose3d_norm2[16][0])
                    # right_wrist_3d_norm2_y.append(pose3d_norm2[16][1])
                    # right_wrist_3d_norm2_z.append(pose3d_norm2[16][2])

                # Draw only body landmarks
                draw_body_landmarks(frame, results.pose_landmarks, mp_pose)

            # Display information on frame
            status_text = "RECORDING" if is_recording else "PAUSED"
            status_color = (0, 255, 0) if is_recording else (0, 0, 255)
            cv2.putText(frame, f"Status: {status_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"Time: {elapsed_time:.2f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Data points: {len(timestamps)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "2D + 3D Recording Active", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, "Press 'q' to quit, 's' to pause/resume", (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show frame
            if SHOW_VISUALIZATION:
                cv2.imshow('MediaPipe 2D to 3D Pose Estimation', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Stopping recording...")
                break
            elif key == ord('s'):
                is_recording = not is_recording
                state = "resumed" if is_recording else "paused"
                print(f"Recording {state}")

    except KeyboardInterrupt:
        print("Recording interrupted by user")
    finally:
        # Cleanup
        if USE_REALSENSE_DEPTH:
            try:
                realsense_pipeline.stop()
            except Exception:
                pass
        else:
            try:
                cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        pose.close()

    # Save data to Excel
    if len(timestamps) > 0:
        # Create output folder if it doesn't exist
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
            print(f"Created output folder: {OUTPUT_FOLDER}")
        
        # Generate filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_EXCEL_PREFIX}_{timestamp_str}.xlsx")
        
        print(f"Saving {len(timestamps)} data points to {output_file}...")
        
        # Prepare data dictionary (right hand only)
        data_dict = {
            'Timestamp': timestamps,
            # Unnormalized 3D coordinates (raw camera coordinates in meters)
            'Right_Shoulder_3D_X': right_shoulder_3d_x,
            'Right_Shoulder_3D_Y': right_shoulder_3d_y,
            'Right_Shoulder_3D_Z': right_shoulder_3d_z,
            'Right_Elbow_3D_X': right_elbow_3d_x,
            'Right_Elbow_3D_Y': right_elbow_3d_y,
            'Right_Elbow_3D_Z': right_elbow_3d_z,
            'Right_Wrist_3D_X': right_wrist_3d_x,
            'Right_Wrist_3D_Y': right_wrist_3d_y,
            'Right_Wrist_3D_Z': right_wrist_3d_z,
        }
        
        # Add normalized 3D coordinates only if bone transform is enabled
        if APPLY_BONE_TRANSFORM:
            data_dict.update({
                'Right_Shoulder_3D_Norm_X': right_shoulder_3d_norm_x,
                'Right_Shoulder_3D_Norm_Y': right_shoulder_3d_norm_y,
                'Right_Shoulder_3D_Norm_Z': right_shoulder_3d_norm_z,
                'Right_Elbow_3D_Norm_X': right_elbow_3d_norm_x,
                'Right_Elbow_3D_Norm_Y': right_elbow_3d_norm_y,
                'Right_Elbow_3D_Norm_Z': right_elbow_3d_norm_z,
                'Right_Wrist_3D_Norm_X': right_wrist_3d_norm_x,
                'Right_Wrist_3D_Norm_Y': right_wrist_3d_norm_y,
                'Right_Wrist_3D_Norm_Z': right_wrist_3d_norm_z,
            })
            # data_dict.update({
            #     'Right_Shoulder_3D_Norm2_X': right_shoulder_3d_norm2_x,
            #     'Right_Shoulder_3D_Norm2_Y': right_shoulder_3d_norm2_y,
            #     'Right_Shoulder_3D_Norm2_Z': right_shoulder_3d_norm2_z,
            #     'Right_Elbow_3D_Norm2_X': right_elbow_3d_norm2_x,
            #     'Right_Elbow_3D_Norm2_Y': right_elbow_3d_norm2_y,
            #     'Right_Elbow_3D_Norm2_Z': right_elbow_3d_norm2_z,
            #     'Right_Wrist_3D_Norm2_X': right_wrist_3d_norm2_x,
            #     'Right_Wrist_3D_Norm2_Y': right_wrist_3d_norm2_y,
            #     'Right_Wrist_3D_Norm2_Z': right_wrist_3d_norm2_z,
            # })
        
        # Apply Savitzky-Golay smoothing if enabled
        if ENABLE_SMOOTHING and len(timestamps) >= SAVGOL_WINDOW:
            print(f"Applying Savitzky-Golay smoothing (window={SAVGOL_WINDOW}, polyorder={SAVGOL_POLYORDER})...")
            data_dict = apply_savgol_smoothing(data_dict, SAVGOL_WINDOW, SAVGOL_POLYORDER)
        
        # Create DataFrame with all data
        df = pd.DataFrame(data_dict)
        df['Frame_Width'] = [frame_width] * len(timestamps)
        df['Frame_Height'] = [frame_height] * len(timestamps)

        # Save single sheet with all data
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='All_Data', index=False)

        print(f"Data saved successfully to {output_file} (sheet: All_Data)")
        print(f"Total recording time: {timestamps[-1]:.2f} seconds")
    else:
        print("No data recorded. Excel file not created.")


if __name__ == "__main__":
    main()
