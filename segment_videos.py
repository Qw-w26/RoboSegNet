import numpy as np
import cv2
from pathlib import Path
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from trajectory_segmentation import TrajectorySegmenter
from plot_fast_droid_embeddings import (
    load_fast_droid_tokenizer,
    extract_data_from_h5,
    create_windows,
    create_embeddings_from_windows_with_dino,
    extract_frame_from_video,
    extract_dino_features
)
import glob
import os
import shutil
from collections import defaultdict

def segment_video(video_path, segments, window_size, stride, output_dir):
    """
    Split a video into segments based on segmentation labels.
    
    Args:
        video_path: Path to the input video
        segments: Array of segment labels for each window
        window_size: Size of the sliding window
        stride: Stride used for window creation
        output_dir: Directory to save segmented videos
    """
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Group consecutive windows by segment
    segment_groups = []
    current_segment = segments[0]
    current_group = [0]
    
    for i in range(1, len(segments)):
        if segments[i] == current_segment:
            current_group.append(i)
        else:
            segment_groups.append((current_segment, current_group))
            current_segment = segments[i]
            current_group = [i]
    segment_groups.append((current_segment, current_group))
    
    # Convert window indices to frame ranges
    segment_frame_ranges = []
    for seg_id, window_indices in segment_groups:
        # Get start frame of first window
        start_window_idx = window_indices[0]
        start_frame = start_window_idx * stride
        
        # Get end frame of last window
        end_window_idx = window_indices[-1]
        end_frame = end_window_idx * stride + window_size
        
        # Clip to video length
        end_frame = min(end_frame, total_frames)
        
        segment_frame_ranges.append((seg_id, start_frame, end_frame))
    
    # Create video writers for each segment
    for seg_idx, (seg_id, start_frame, end_frame) in enumerate(segment_frame_ranges):
        output_path = output_dir / f"segment_{seg_idx:02d}_class_{seg_id}.mp4"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Write frames for this segment
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add segment info overlay
            cv2.putText(frame, f"Segment {seg_idx} (Class {seg_id})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame {frame_idx}/{total_frames}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1)
            
            out.write(frame)
        
        out.release()
        print(f"  Saved segment {seg_idx} (frames {start_frame}-{end_frame}) to {output_path.name}")
    
    cap.release()
    return segment_frame_ranges

def create_segment_visualization(trajectory_path, segments, window_size, stride, output_path):
    """
    Create a visualization showing the segmentation overlay on trajectory.
    """
    # Load trajectory data
    trajectory = extract_data_from_h5(trajectory_path)
    if trajectory is None:
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Convert segments to frame-level labels
    n_frames = len(trajectory)
    frame_labels = np.zeros(n_frames)
    
    for window_idx, seg_label in enumerate(segments):
        start_frame = window_idx * stride
        end_frame = min(start_frame + window_size, n_frames)
        frame_labels[start_frame:end_frame] = seg_label
    
    # Plot joint trajectories with segment coloring
    time_steps = np.arange(n_frames)
    
    # Plot first 3 joints
    for i in range(min(3, trajectory.shape[1])):
        ax = axes[i]
        
        # Plot trajectory colored by segment
        for seg_id in np.unique(frame_labels):
            mask = frame_labels == seg_id
            ax.plot(time_steps[mask], trajectory[mask, i], 
                   label=f'Segment {int(seg_id)}', linewidth=2)
        
        ax.set_ylabel(f'Joint {i+1} (rad)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        if i == 0:
            ax.set_title('Trajectory Segmentation')
    
    axes[-1].set_xlabel('Time Steps')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to segment videos based on trajectory segmentation results.
    """
    # Configuration
    window_size = 15
    stride = 1
    dataset_path = Path("/home/kushtimusprime/ur5_policy_training/data_tiger_picking_90_good_and_bad_cleaned")
    output_base_dir = Path("/home/syx/ucsf/task_seg/segmented_videos")
    
    print("Loading trajectory data and creating embeddings...")
    
    # Load tokenizer
    tokenizer = load_fast_droid_tokenizer()
    
    # Find HDF5 files
    h5_files = sorted(glob.glob(str(dataset_path / "trajectory_*/joints.h5")))[86:]
    print(f"Processing {len(h5_files)} trajectories")
    
    # Process each trajectory
    all_windows = []
    all_dino_features = []
    window_info = []
    trajectory_map = defaultdict(list)  # Map trajectory ID to window indices
    
    for h5_path in tqdm(h5_files, desc="Loading trajectories"):
        trajectory = extract_data_from_h5(h5_path)
        if trajectory is None or len(trajectory) < window_size:
            continue
        
        # Extract trajectory ID
        traj_dir = Path(h5_path).parent
        traj_id = int(traj_dir.name.split('_')[-1])
        
        # Get video path
        video_path = traj_dir / "wrist.mp4"
        if not video_path.exists():
            video_path = traj_dir / "exo.mp4"
        
        # Create windows
        windows = create_windows(trajectory, window_size, stride, use_joints=False)
        
        # Track window indices for this trajectory
        start_idx = len(all_windows)
        
        # Process windows
        for window, center_frame_idx in windows:
            all_windows.append(window)
            window_info.append((traj_id, len(all_windows) - 1))
            
            # Extract DINO features
            if video_path.exists():
                image = extract_frame_from_video(video_path, center_frame_idx)
                if image:
                    dino_features = extract_dino_features(image)
                else:
                    dino_features = np.zeros(1024)
            else:
                dino_features = np.zeros(1024)
            all_dino_features.append(dino_features)
        
        # Store mapping
        trajectory_map[traj_id] = {
            'h5_path': h5_path,
            'video_path': video_path if video_path.exists() else None,
            'window_indices': list(range(start_idx, len(all_windows)))
        }
    
    print(f"Created {len(all_windows)} windows")
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = create_embeddings_from_windows_with_dino(
        all_windows, tokenizer, dino_features_list=all_dino_features
    )
    
    # Initialize segmenter
    segmenter = TrajectorySegmenter(
        embeddings,
        window_info,
        window_size=window_size,
        stride=stride
    )
    
    # Run segmentation with different methods
    methods = {
        'spectral': lambda: segmenter.segment_with_spectral_clustering(n_clusters=5, temporal_weight=0.3),
        'change_point': lambda: segmenter.segment_with_change_point_detection(penalty=3),
        'gmm': lambda: segmenter.segment_with_gmm(n_components=5)[0],
        'hierarchical': lambda: segmenter.hierarchical_segmentation(n_clusters=5)[0]
    }
    
    print("\nSegmenting trajectories with different methods...")
    
    for method_name, method_func in methods.items():
        print(f"\n{'='*50}")
        print(f"Method: {method_name.upper()}")
        print('='*50)
        
        # Get segmentation labels
        if method_name == 'change_point':
            labels, _ = method_func()
        else:
            labels = method_func()
        
        # Process each trajectory
        for traj_id, traj_data in trajectory_map.items():
            if traj_data['video_path'] is None:
                continue
            
            # Get labels for this trajectory's windows
            window_indices = traj_data['window_indices']
            traj_segments = labels[window_indices]
            
            # Create output directory structure
            output_dir = output_base_dir / method_name / f"trajectory_{traj_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nTrajectory {traj_id}:")
            
            # Segment video
            segment_frame_ranges = segment_video(
                traj_data['video_path'],
                traj_segments,
                window_size,
                stride,
                output_dir
            )
            
            # Create visualization
            viz_path = output_dir / "segmentation_plot.png"
            create_segment_visualization(
                traj_data['h5_path'],
                traj_segments,
                window_size,
                stride,
                viz_path
            )
            print(f"  Saved visualization to {viz_path.name}")
            
            # Save segment info
            info_path = output_dir / "segment_info.txt"
            with open(info_path, 'w') as f:
                f.write(f"Trajectory {traj_id} Segmentation\n")
                f.write(f"Method: {method_name}\n")
                f.write(f"Number of segments: {len(segment_frame_ranges)}\n")
                f.write(f"Window size: {window_size}\n")
                f.write(f"Stride: {stride}\n\n")
                
                f.write("Segment Details:\n")
                for seg_idx, (seg_id, start_frame, end_frame) in enumerate(segment_frame_ranges):
                    duration = (end_frame - start_frame) / 30.0  # Assuming 30 FPS
                    f.write(f"  Segment {seg_idx}: Class {seg_id}, "
                           f"Frames {start_frame}-{end_frame} "
                           f"({duration:.2f} seconds)\n")
    
    print(f"\n{'='*50}")
    print(f"Segmentation complete! Videos saved to: {output_base_dir}")
    print('='*50)

if __name__ == "__main__":
    main()