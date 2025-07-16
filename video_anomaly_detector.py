import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime, timedelta
import json
import argparse
from pathlib import Path

class VideoAnomalyDetector:
    def __init__(self, model_path, video_path, gps_log_path, output_dir="output"):
        """
        Initialize the anomaly detector
        
        Args:
            model_path (str): Path to custom YOLOv8 model
            video_path (str): Path to video file
            gps_log_path (str): Path to GPS log file (CSV format)
            output_dir (str): Output directory for results
        """
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.gps_log_path = gps_log_path
        self.output_dir = Path(output_dir)
        
        # Class-specific confidence thresholds
        self.class_confidence_thresholds = {
            0: 0.60,  # crack
            1: 0.30,  # crosswalk_blur
            2: 0.60,  # facility
            3: 0.30,  # lane_blur
            4: 1.0,   # manhole
            5: 0.75,  # pole
            6: 0.25   # pothole
        }
        
        # Create output directories
        self.create_output_directories()
        
        # Load GPS data
        self.gps_data = self.load_gps_data()
        
        # Results storage - organized by class
        self.anomalies_by_class = {}
        for class_id in self.class_confidence_thresholds.keys():
            self.anomalies_by_class[class_id] = []
            
        # Complete video metadata storage
        self.complete_video_metadata = []
        
    def create_output_directories(self):
        """Create output directories for each class"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create class-specific directories
        self.class_dirs = {}
        for class_id, threshold in self.class_confidence_thresholds.items():
            class_name = self.get_class_name(class_id)
            class_dir = self.output_dir / f"class_{class_id}_{class_name}"
            class_dir.mkdir(parents=True, exist_ok=True)
            self.class_dirs[class_id] = class_dir
            
    def get_class_name(self, class_id):
        """Get class name from class ID"""
        class_names = {
            0: "crack",
            1: "crosswalk_blur", 
            2: "facility",
            3: "lane_blur",
            4: "manhole",
            5: "pole",
            6: "pothole"
        }
        return class_names.get(class_id, f"class_{class_id}")
        
    def load_gps_data(self):
        """
        Load GPS data from CSV file
        Handles your specific GPS format with record_timestamp, latitude, longitude, altitude
        """
        try:
            df = pd.read_csv(self.gps_log_path)
            
            # Your GPS data uses 'record_timestamp' as the main timestamp
            if 'record_timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['record_timestamp'])
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                print("Warning: No timestamp column found. Expected 'record_timestamp' or 'timestamp'")
                return None
            
            # Verify required columns exist
            required_columns = ['latitude', 'longitude']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Error: Missing required columns: {missing_columns}")
                return None
            
            # Filter out rows without valid GPS fix
            if 'has_fix' in df.columns:
                df = df[df['has_fix'] == True]
                print(f"Filtered to {len(df)} records with valid GPS fix")
            
            if 'has_position' in df.columns:
                df = df[df['has_position'] == True]
                print(f"Filtered to {len(df)} records with valid position")
            
            # Sort by timestamp to ensure proper ordering
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"Loaded GPS data with {len(df)} valid records")
            print(f"GPS data time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"GPS data columns: {df.columns.tolist()}")
            
            # Display sample of GPS data
            print("\nSample GPS data:")
            print(df[['timestamp', 'latitude', 'longitude', 'altitude', 'speed']].head())
            
            return df
            
        except Exception as e:
            print(f"Error loading GPS data: {e}")
            return None
    
    def get_video_info(self):
        """Get video information"""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        
        return {
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration
        }
    
    def frame_to_timestamp(self, frame_number, fps, video_start_time=None):
        """
        Convert frame number to timestamp
        
        Args:
            frame_number (int): Frame number
            fps (float): Video FPS
            video_start_time (datetime): Start time of video recording
        
        Returns:
            datetime: Timestamp for the frame
        """
        if video_start_time is None:
            # If no start time provided, use GPS data start time
            if self.gps_data is not None and 'timestamp' in self.gps_data.columns:
                video_start_time = self.gps_data['timestamp'].min()
            else:
                video_start_time = datetime.now()
        
        seconds_offset = frame_number / fps
        return video_start_time + timedelta(seconds=seconds_offset)
    
    def find_closest_gps_point(self, timestamp):
        """
        Find the closest GPS point to a given timestamp
        
        Args:
            timestamp (datetime): Target timestamp
            
        Returns:
            dict: GPS data for the closest point including additional GPS metrics
        """
        if self.gps_data is None or 'timestamp' not in self.gps_data.columns:
            return None
        
        # Find the closest timestamp
        time_diff = abs(self.gps_data['timestamp'] - timestamp)
        closest_idx = time_diff.idxmin()
        
        gps_point = self.gps_data.iloc[closest_idx]
        
        return {
            'latitude': gps_point.get('latitude', None),
            'longitude': gps_point.get('longitude', None),
            'altitude': gps_point.get('altitude', None),
            'speed': gps_point.get('speed', None),
            'course': gps_point.get('course', None),
            'fix_quality': gps_point.get('fix_quality', None),
            'satellites_used': gps_point.get('satellites_used', None),
            'hdop': gps_point.get('hdop', None),
            'vdop': gps_point.get('vdop', None),
            'pdop': gps_point.get('pdop', None),
            'has_fix': gps_point.get('has_fix', None),
            'has_position': gps_point.get('has_position', None),
            'timestamp': gps_point.get('timestamp', None),
            'time_diff_seconds': time_diff.iloc[closest_idx].total_seconds()
        }
    
    def draw_bounding_boxes(self, image, results, valid_detections):
        """
        Draw bounding boxes on image for valid detections only (for class-specific images)
        
        Args:
            image (numpy.ndarray): Input image
            results: YOLOv8 results
            valid_detections (list): List of valid detection indices
            
        Returns:
            numpy.ndarray: Image with bounding boxes
        """
        annotated_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    if i in valid_detections:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.get_class_name(class_id)
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Add label
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(annotated_image, label, (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated_image
    
    def draw_all_bounding_boxes(self, image, results):
        """
        Draw bounding boxes for all valid detections on image
        
        Args:
            image (numpy.ndarray): Input image
            results: YOLOv8 results
            
        Returns:
            tuple: (annotated_image, detection_info_list)
        """
        annotated_image = image.copy()
        detection_info = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Check if this detection meets the class-specific confidence threshold
                    if class_id in self.class_confidence_thresholds:
                        class_threshold = self.class_confidence_thresholds[class_id]
                        if confidence >= class_threshold:
                            # Get class name
                            class_name = self.get_class_name(class_id)
                            
                            # Choose color based on class
                            colors = {
                                0: (255, 0, 0),    # crack - blue
                                1: (0, 255, 0),    # crosswalk_blur - green
                                2: (0, 0, 255),    # facility - red
                                3: (255, 255, 0),  # lane_blur - cyan
                                4: (255, 0, 255),  # manhole - magenta
                                5: (0, 255, 255),  # pole - yellow
                                6: (128, 0, 128)   # pothole - purple
                            }
                            color = colors.get(class_id, (0, 255, 0))
                            
                            # Draw bounding box
                            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            
                            # Add label with background
                            label = f"{class_name}: {confidence:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(annotated_image, (int(x1), int(y1) - label_size[1] - 10), 
                                        (int(x1) + label_size[0], int(y1)), color, -1)
                            cv2.putText(annotated_image, label, (int(x1), int(y1) - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            # Store detection info
                            detection_info.append({
                                'class_name': class_name,
                                'confidence': float(confidence),
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'class_id': class_id
                            })
        
        return annotated_image, detection_info

    def process_video(self, video_start_time=None):
        """
        Process video and detect anomalies using class-specific confidence thresholds
        Also creates a complete inferenced video with all detections
        
        Args:
            video_start_time (datetime): Start time of video recording
        """
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return
        
        video_info = self.get_video_info()
        fps = video_info['fps']
        total_frames = video_info['total_frames']
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer for complete inferenced video
        output_video_path = self.output_dir / "complete_inferenced_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        print(f"Processing video: {self.video_path}")
        print(f"Video info: {video_info}")
        print(f"Output video: {output_video_path}")
        print(f"Class confidence thresholds: {self.class_confidence_thresholds}")
        
        frame_count = 0
        total_anomaly_count = 0
        class_anomaly_counts = {class_id: 0 for class_id in self.class_confidence_thresholds.keys()}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLOv8 inference with minimum confidence
            min_confidence = min(self.class_confidence_thresholds.values())
            results = self.model(frame, conf=min_confidence)
            
            # Get timestamp for this frame
            frame_timestamp = self.frame_to_timestamp(frame_count, fps, video_start_time)
            
            # Find closest GPS point
            gps_data = self.find_closest_gps_point(frame_timestamp)
            
            # Draw all valid detections for the complete video
            annotated_frame_complete, all_detections = self.draw_all_bounding_boxes(frame, results)
            
            # Add frame info overlay
            info_text = f"Frame: {frame_count} | Time: {frame_timestamp.strftime('%H:%M:%S')}"
            if gps_data and gps_data['latitude'] and gps_data['longitude']:
                info_text += f" | GPS: {gps_data['latitude']:.6f}, {gps_data['longitude']:.6f}"
            
            cv2.putText(annotated_frame_complete, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame_complete, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Write frame to complete video
            out.write(annotated_frame_complete)
            
            # Store complete frame metadata
            frame_metadata = {
                'frame_number': frame_count,
                'timestamp': frame_timestamp,
                'detections': all_detections,
                'detection_count': len(all_detections),
                'gps_latitude': gps_data['latitude'] if gps_data else None,
                'gps_longitude': gps_data['longitude'] if gps_data else None,
                'gps_altitude': gps_data['altitude'] if gps_data else None,
                'gps_speed': gps_data['speed'] if gps_data else None,
                'gps_course': gps_data['course'] if gps_data else None,
                'gps_timestamp': gps_data['timestamp'] if gps_data else None,
                'gps_time_diff_seconds': gps_data['time_diff_seconds'] if gps_data else None
            }
            self.complete_video_metadata.append(frame_metadata)
            
            # Process detections by class for class-specific outputs
            valid_detections_by_class = {}
            frame_has_anomalies = False
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Check if this detection meets the class-specific confidence threshold
                        if class_id in self.class_confidence_thresholds:
                            class_threshold = self.class_confidence_thresholds[class_id]
                            if confidence >= class_threshold:
                                if class_id not in valid_detections_by_class:
                                    valid_detections_by_class[class_id] = []
                                
                                class_name = self.get_class_name(class_id)
                                
                                valid_detections_by_class[class_id].append({
                                    'detection_index': i,
                                    'class_name': class_name,
                                    'confidence': float(confidence),
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'class_id': class_id
                                })
                                frame_has_anomalies = True
            
            # Process each class with valid detections for class-specific images
            if frame_has_anomalies:
                for class_id, detections in valid_detections_by_class.items():
                    # Draw bounding boxes for this class only
                    valid_detection_indices = [det['detection_index'] for det in detections]
                    annotated_frame = self.draw_bounding_boxes(frame, results, valid_detection_indices)
                    
                    # Save image to class-specific folder
                    class_anomaly_count = class_anomaly_counts[class_id]
                    class_name = self.get_class_name(class_id)
                    image_filename = f"{class_name}_anomaly_{class_anomaly_count:06d}_frame_{frame_count:08d}.jpg"
                    image_path = self.class_dirs[class_id] / image_filename
                    cv2.imwrite(str(image_path), annotated_frame)
                    
                    # Store anomaly metadata for this class
                    anomaly_data = {
                        'anomaly_id': class_anomaly_count,
                        'frame_number': frame_count,
                        'timestamp': frame_timestamp,
                        'image_filename': image_filename,
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence_threshold': self.class_confidence_thresholds[class_id],
                        'detections': detections,
                        'detection_count': len(detections),
                        'gps_latitude': gps_data['latitude'] if gps_data else None,
                        'gps_longitude': gps_data['longitude'] if gps_data else None,
                        'gps_altitude': gps_data['altitude'] if gps_data else None,
                        'gps_timestamp': gps_data['timestamp'] if gps_data else None,
                        'gps_time_diff_seconds': gps_data['time_diff_seconds'] if gps_data else None
                    }
                    
                    self.anomalies_by_class[class_id].append(anomaly_data)
                    class_anomaly_counts[class_id] += 1
                    
                    print(f"Class {class_name} anomaly {class_anomaly_counts[class_id]} detected at frame {frame_count}")
                
                total_anomaly_count += 1
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"\nProcessing complete!")
        print(f"Complete inferenced video saved to: {output_video_path}")
        print(f"Total frames processed: {frame_count}")
        print(f"Frames with anomalies: {total_anomaly_count}")
        for class_id, count in class_anomaly_counts.items():
            class_name = self.get_class_name(class_id)
            print(f"Class {class_name}: {count} anomalies")
    
    def save_metadata(self):
        """Save metadata to separate CSV files for each class"""
        print("\n=== Saving Class-Specific Metadata ===")
        
        for class_id, anomalies in self.anomalies_by_class.items():
            class_name = self.get_class_name(class_id)
            print(f"Processing class {class_name} (ID: {class_id}) - {len(anomalies)} anomalies")
            
            if not anomalies:
                # Create empty metadata files for classes with no detections
                empty_csv_path = self.class_dirs[class_id] / f"{class_name}_metadata.csv"
                empty_df = pd.DataFrame(columns=[
                    'anomaly_id', 'frame_number', 'timestamp', 'image_filename',
                    'class_id', 'class_name', 'confidence_threshold', 'detection_count',
                    'gps_latitude', 'gps_longitude', 'gps_altitude', 'gps_timestamp',
                    'gps_time_diff_seconds', 'detection_id', 'detection_confidence',
                    'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'
                ])
                empty_df.to_csv(empty_csv_path, index=False)
                print(f"  → Empty metadata CSV created: {empty_csv_path}")
                
                # Create empty JSON
                empty_json_path = self.class_dirs[class_id] / f"{class_name}_detailed_metadata.json"
                with open(empty_json_path, 'w') as f:
                    json.dump([], f, indent=2)
                print(f"  → Empty detailed JSON created: {empty_json_path}")
                continue
            
            # Flatten the data for CSV
            csv_data = []
            for anomaly in anomalies:
                base_data = {
                    'anomaly_id': anomaly['anomaly_id'],
                    'frame_number': anomaly['frame_number'],
                    'timestamp': anomaly['timestamp'],
                    'image_filename': anomaly['image_filename'],
                    'class_id': anomaly['class_id'],
                    'class_name': anomaly['class_name'],
                    'confidence_threshold': anomaly['confidence_threshold'],
                    'detection_count': anomaly['detection_count'],
                    'gps_latitude': anomaly['gps_latitude'],
                    'gps_longitude': anomaly['gps_longitude'],
                    'gps_altitude': anomaly['gps_altitude'],
                    'gps_timestamp': anomaly['gps_timestamp'],
                    'gps_time_diff_seconds': anomaly['gps_time_diff_seconds']
                }
                
                # Add detection details - one row per detection
                for i, detection in enumerate(anomaly['detections']):
                    detection_data = base_data.copy()
                    detection_data.update({
                        'detection_id': i,
                        'detection_confidence': detection['confidence'],
                        'bbox_x1': detection['bbox'][0],
                        'bbox_y1': detection['bbox'][1],
                        'bbox_x2': detection['bbox'][2],
                        'bbox_y2': detection['bbox'][3]
                    })
                    csv_data.append(detection_data)
            
            # Save to class-specific CSV
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_path = self.class_dirs[class_id] / f"{class_name}_metadata.csv"
                df.to_csv(csv_path, index=False)
                print(f"  → Class metadata CSV saved: {csv_path} ({len(csv_data)} detection records)")
                
                # Also save detailed JSON for reference
                json_path = self.class_dirs[class_id] / f"{class_name}_detailed_metadata.json"
                with open(json_path, 'w') as f:
                    json.dump(anomalies, f, indent=2, default=str)
                print(f"  → Class detailed JSON saved: {json_path} ({len(anomalies)} anomaly records)")
                
                # Create a summary file for this class
                summary_path = self.class_dirs[class_id] / f"{class_name}_summary.txt"
                total_detections = sum(len(anomaly['detections']) for anomaly in anomalies)
                summary_text = f"""
{class_name.upper()} CLASS SUMMARY
{'=' * (len(class_name) + 14)}

Class ID: {class_id}
Class Name: {class_name}
Confidence Threshold: {self.class_confidence_thresholds[class_id]}

Detection Statistics:
- Total Anomaly Frames: {len(anomalies)}
- Total Detections: {total_detections}
- Average Detections per Frame: {total_detections / len(anomalies):.2f}

Files in this folder:
- {class_name}_metadata.csv - Detailed CSV with all detection data
- {class_name}_detailed_metadata.json - Complete JSON metadata
- {class_name}_summary.txt - This summary file
- *.jpg - Anomaly detection images

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                with open(summary_path, 'w') as f:
                    f.write(summary_text)
                print(f"  → Class summary saved: {summary_path}")
        
        print("=== Class-Specific Metadata Saving Complete ===\n")
        
        # Save combined metadata
        self.save_combined_metadata()
        
        # Save complete video metadata
        self.save_complete_video_metadata()
    
    def save_combined_metadata(self):
        """Save combined metadata for all classes"""
        all_anomalies = []
        combined_csv_data = []
        
        for class_id, anomalies in self.anomalies_by_class.items():
            all_anomalies.extend(anomalies)
            
            # Add to combined CSV data
            for anomaly in anomalies:
                base_data = {
                    'anomaly_id': f"{self.get_class_name(class_id)}_{anomaly['anomaly_id']}",
                    'frame_number': anomaly['frame_number'],
                    'timestamp': anomaly['timestamp'],
                    'image_filename': anomaly['image_filename'],
                    'class_id': anomaly['class_id'],
                    'class_name': anomaly['class_name'],
                    'confidence_threshold': anomaly['confidence_threshold'],
                    'detection_count': anomaly['detection_count'],
                    'gps_latitude': anomaly['gps_latitude'],
                    'gps_longitude': anomaly['gps_longitude'],
                    'gps_altitude': anomaly['gps_altitude'],
                    'gps_timestamp': anomaly['gps_timestamp'],
                    'gps_time_diff_seconds': anomaly['gps_time_diff_seconds']
                }
                
                # Add detection details
                for i, detection in enumerate(anomaly['detections']):
                    detection_data = base_data.copy()
                    detection_data.update({
                        'detection_id': i,
                        'detection_confidence': detection['confidence'],
                        'bbox_x1': detection['bbox'][0],
                        'bbox_y1': detection['bbox'][1],
                        'bbox_x2': detection['bbox'][2],
                        'bbox_y2': detection['bbox'][3]
                    })
                    combined_csv_data.append(detection_data)
        
        if combined_csv_data:
            # Save combined CSV
            df = pd.DataFrame(combined_csv_data)
            csv_path = self.output_dir / "combined_metadata.csv"
            df.to_csv(csv_path, index=False)
            print(f"Combined metadata saved to: {csv_path}")
            
            # Save combined JSON
            json_path = self.output_dir / "combined_detailed_metadata.json"
            with open(json_path, 'w') as f:
                json.dump(all_anomalies, f, indent=2, default=str)
            print(f"Combined detailed metadata saved to: {json_path}")
    
    def save_complete_video_metadata(self):
        """Save complete video metadata for every frame"""
        if not self.complete_video_metadata:
            print("No video metadata to save.")
            return
        
        # Flatten the data for CSV
        csv_data = []
        for frame_data in self.complete_video_metadata:
            base_data = {
                'frame_number': frame_data['frame_number'],
                'timestamp': frame_data['timestamp'],
                'detection_count': frame_data['detection_count'],
                'gps_latitude': frame_data['gps_latitude'],
                'gps_longitude': frame_data['gps_longitude'],
                'gps_altitude': frame_data['gps_altitude'],
                'gps_speed': frame_data['gps_speed'],
                'gps_course': frame_data['gps_course'],
                'gps_timestamp': frame_data['gps_timestamp'],
                'gps_time_diff_seconds': frame_data['gps_time_diff_seconds']
            }
            
            if frame_data['detections']:
                # Add detection details
                for i, detection in enumerate(frame_data['detections']):
                    detection_data = base_data.copy()
                    detection_data.update({
                        'detection_id': i,
                        'class_name': detection['class_name'],
                        'confidence': detection['confidence'],
                        'bbox_x1': detection['bbox'][0],
                        'bbox_y1': detection['bbox'][1],
                        'bbox_x2': detection['bbox'][2],
                        'bbox_y2': detection['bbox'][3],
                        'class_id': detection['class_id']
                    })
                    csv_data.append(detection_data)
            else:
                # Frame with no detections
                base_data.update({
                    'detection_id': None,
                    'class_name': None,
                    'confidence': None,
                    'bbox_x1': None,
                    'bbox_y1': None,
                    'bbox_x2': None,
                    'bbox_y2': None,
                    'class_id': None
                })
                csv_data.append(base_data)
        
        # Save complete video metadata CSV
        df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / "complete_video_metadata.csv"
        df.to_csv(csv_path, index=False)
        print(f"Complete video metadata saved to: {csv_path}")
        
        # Save complete video metadata JSON
        json_path = self.output_dir / "complete_video_detailed_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(self.complete_video_metadata, f, indent=2, default=str)
        print(f"Complete video detailed metadata saved to: {json_path}")
        
        # Generate video statistics
        self.generate_video_statistics()
    
    def generate_video_statistics(self):
        """Generate comprehensive video statistics"""
        if not self.complete_video_metadata:
            return
        
        total_frames = len(self.complete_video_metadata)
        frames_with_detections = sum(1 for frame in self.complete_video_metadata if frame['detection_count'] > 0)
        total_detections = sum(frame['detection_count'] for frame in self.complete_video_metadata)
        
        # Class-wise statistics
        class_frame_counts = {}
        class_detection_counts = {}
        
        for frame in self.complete_video_metadata:
            for detection in frame['detections']:
                class_name = detection['class_name']
                class_frame_counts[class_name] = class_frame_counts.get(class_name, 0) + 1
                class_detection_counts[class_name] = class_detection_counts.get(class_name, 0) + 1
        
        # GPS coverage statistics
        frames_with_gps = sum(1 for frame in self.complete_video_metadata 
                             if frame['gps_latitude'] is not None and frame['gps_longitude'] is not None)
        
        # Time-based statistics
        if self.complete_video_metadata:
            start_time = self.complete_video_metadata[0]['timestamp']
            end_time = self.complete_video_metadata[-1]['timestamp']
            duration = end_time - start_time
        
        # Generate statistics report
        stats = {
            'video_statistics': {
                'total_frames': total_frames,
                'frames_with_detections': frames_with_detections,
                'frames_without_detections': total_frames - frames_with_detections,
                'detection_rate_percentage': (frames_with_detections / total_frames * 100) if total_frames > 0 else 0,
                'total_detections': total_detections,
                'average_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0,
                'start_time': start_time,
                'end_time': end_time,
                'duration_seconds': duration.total_seconds() if 'duration' in locals() else 0
            },
            'gps_statistics': {
                'frames_with_gps': frames_with_gps,
                'frames_without_gps': total_frames - frames_with_gps,
                'gps_coverage_percentage': (frames_with_gps / total_frames * 100) if total_frames > 0 else 0
            },
            'class_statistics': {
                'frames_per_class': class_frame_counts,
                'detections_per_class': class_detection_counts,
                'confidence_thresholds': self.class_confidence_thresholds
            }
        }
        
        # Save statistics
        stats_path = self.output_dir / "video_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"Video statistics saved to: {stats_path}")
        
        return stats
    
    def verify_output_files(self):
        """Verify that all expected output files have been created"""
        print("\n=== VERIFYING OUTPUT FILES ===")
        
        # Check main output directory files
        main_files = [
            "complete_inferenced_video.mp4",
            "complete_video_metadata.csv", 
            "complete_video_detailed_metadata.json",
            "combined_metadata.csv",
            "combined_detailed_metadata.json",
            "video_statistics.json",
            "summary_report.txt"
        ]
        
        print("Main output directory files:")
        for filename in main_files:
            filepath = self.output_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                print(f"  ✓ {filename} ({size:,} bytes)")
            else:
                print(f"  ✗ {filename} (MISSING)")
        
        # Check class-specific directories
        print("\nClass-specific directories:")
        for class_id, class_dir in self.class_dirs.items():
            class_name = self.get_class_name(class_id)
            anomaly_count = len(self.anomalies_by_class[class_id])
            
            print(f"\n  {class_name} (class_{class_id}):")
            
            # Expected files in each class directory
            expected_files = [
                f"{class_name}_metadata.csv",
                f"{class_name}_detailed_metadata.json", 
                f"{class_name}_summary.txt"
            ]
            
            for filename in expected_files:
                filepath = class_dir / filename
                if filepath.exists():
                    size = filepath.stat().st_size
                    print(f"    ✓ {filename} ({size:,} bytes)")
                else:
                    print(f"    ✗ {filename} (MISSING)")
            
            # Count image files
            image_files = list(class_dir.glob("*.jpg"))
            expected_images = anomaly_count
            print(f"    Images: {len(image_files)} found (expected: {expected_images})")
            
            if len(image_files) != expected_images:
                print(f"    ⚠ Image count mismatch!")
        
        print("\n=== VERIFICATION COMPLETE ===\n")
    
    def generate_summary_report(self):
        """Generate a summary report"""
        total_anomalies = sum(len(anomalies) for anomalies in self.anomalies_by_class.values())
        
        if total_anomalies == 0:
            print("No anomalies detected.")
            return
        
        # Count detections by class
        class_stats = {}
        total_detections = 0
        
        for class_id, anomalies in self.anomalies_by_class.items():
            class_name = self.get_class_name(class_id)
            detection_count = sum(len(anomaly['detections']) for anomaly in anomalies)
            class_stats[class_name] = {
                'anomalies': len(anomalies),
                'detections': detection_count,
                'confidence_threshold': self.class_confidence_thresholds[class_id]
            }
            total_detections += detection_count
        
        # Generate report
        report = f"""
ANOMALY DETECTION SUMMARY REPORT
================================

Total anomalies detected: {total_anomalies}
Total detections: {total_detections}

Detection breakdown by class:
{'-' * 50}
"""
        
        for class_name, stats in sorted(class_stats.items()):
            if stats['anomalies'] > 0:
                percentage = (stats['detections'] / total_detections) * 100
                report += f"{class_name:15} | Anomalies: {stats['anomalies']:4d} | Detections: {stats['detections']:4d} ({percentage:5.1f}%) | Threshold: {stats['confidence_threshold']:.2f}\n"
        
        report += f"""
{'-' * 50}

Output structure:
- Complete inferenced video: {self.output_dir / 'complete_inferenced_video.mp4'}
- Complete video metadata: {self.output_dir / 'complete_video_metadata.csv'}
- Combined metadata: {self.output_dir / 'combined_metadata.csv'}
- Combined detailed JSON: {self.output_dir / 'combined_detailed_metadata.json'}
- Video statistics: {self.output_dir / 'video_statistics.json'}

Class-specific outputs:
"""
        
        for class_id, anomalies in self.anomalies_by_class.items():
            if anomalies:
                class_name = self.get_class_name(class_id)
                report += f"- {class_name}: {self.class_dirs[class_id]}\n"
        
        # Save report
        report_path = self.output_dir / "summary_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"Summary report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Video Anomaly Detection with GPS Integration and Class-Specific Organization")
    parser.add_argument("--model", required=True, help="Path to custom YOLOv8 model")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--gps", required=True, help="Path to GPS log CSV file")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--start-time", help="Video start time (YYYY-MM-DD HH:MM:SS)")
    
    args = parser.parse_args()
    
    # Parse start time if provided
    start_time = None
    if args.start_time:
        try:
            start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print("Invalid start time format. Use YYYY-MM-DD HH:MM:SS")
            return
    
    # Initialize detector
    detector = VideoAnomalyDetector(
        model_path=args.model,
        video_path=args.video,
        gps_log_path=args.gps,
        output_dir=args.output
    )
    
    # Process video
    detector.process_video(video_start_time=start_time)
    
    # Save results
    detector.save_metadata()
    detector.generate_summary_report()
    
    # Verify all output files were created
    detector.verify_output_files()
    
    print("🎉 Processing completed successfully!")
    print(f"📁 All results saved to: {args.output}")
    print(f"🎥 Watch the complete inferenced video: {args.output}/complete_inferenced_video.mp4")

if __name__ == "__main__":
    main()