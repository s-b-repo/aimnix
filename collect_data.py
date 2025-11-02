#!/usr/bin/env python3
"""
CS2 Data Collection Script
Automated screenshot capture for training data
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

try:
    import cv2
    import numpy as np
    import mss
    import keyboard
except ImportError as e:
    print(f"[!] Required packages not installed: {e}")
    print("[!] Run: pip install opencv-python numpy mss keyboard")
    sys.exit(1)

class DataCollector:
    def __init__(self):
        self.output_dir = "cs2_dataset/images/train"
        self.capture_active = False
        self.frame_count = 0
        self.capture_rate = 5  # frames per second
        self.last_capture = 0
        
    def setup_directories(self):
        """Create output directories"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"[+] Output directory: {self.output_dir}")
    
    def capture_screen(self):
        """Capture screen using MSS"""
        with mss.mss() as sct:
            # Get primary monitor
            monitor = sct.monitors[1]  # Primary monitor
            
            while self.capture_active:
                current_time = time.time()
                
                # Check capture rate
                if current_time - self.last_capture < 1.0 / self.capture_rate:
                    time.sleep(0.01)
                    continue
                
                # Capture screenshot
                screenshot = sct.grab(monitor)
                
                # Convert to numpy array
                img = np.array(screenshot)
                
                # Convert BGRA to BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"frame_{timestamp}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                
                cv2.imwrite(filepath, img)
                
                self.frame_count += 1
                self.last_capture = current_time
                
                if self.frame_count % 10 == 0:
                    print(f"[+] Captured {self.frame_count} frames", end='\r')
    
    def start_capture(self):
        """Start the capture process"""
        print("[+] Starting data collection...")
        print("[+] Press F9 to start/stop capture")
        print("[+] Press ESC to exit")
        
        self.setup_directories()
        
        try:
            while True:
                if keyboard.is_pressed('F9'):
                    if not self.capture_active:
                        self.capture_active = True
                        self.frame_count = 0
                        print("\n[+] Capture started!")
                        
                        # Start capture in a separate thread
                        import threading
                        capture_thread = threading.Thread(target=self.capture_screen)
                        capture_thread.daemon = True
                        capture_thread.start()
                    else:
                        self.capture_active = False
                        print(f"\n[+] Capture stopped! Total frames: {self.frame_count}")
                        time.sleep(0.5)  # Debounce
                
                elif keyboard.is_pressed('ESC'):
                    print("\n[+] Exiting...")
                    self.capture_active = False
                    break
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n[!] Interrupted by user")
            self.capture_active = False

def main():
    parser = argparse.ArgumentParser(description="CS2 Data Collection Tool")
    parser.add_argument("--output", "-o", default="cs2_dataset/images/train", 
                       help="Output directory for captured images")
    parser.add_argument("--rate", "-r", type=int, default=5,
                       help="Capture rate in frames per second")
    parser.add_argument("--duration", "-d", type=int, default=0,
                       help="Maximum capture duration in minutes (0 = unlimited)")
    
    args = parser.parse_args()
    
    collector = DataCollector()
    collector.output_dir = args.output
    collector.capture_rate = args.rate
    
    print("CS2 Data Collection Tool")
    print("=======================")
    print(f"Output: {args.output}")
    print(f"Rate: {args.rate} fps")
    if args.duration > 0:
        print(f"Duration: {args.duration} minutes")
    print("")
    
    collector.start_capture()

if __name__ == "__main__":
    main()