#!/usr/bin/env python3
"""
CS2 Data Collection Script
Automated screenshot capture for training data
"""

import os
import sys
import time
import argparse
import threading
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
    def __init__(self, output_dir="cs2_dataset/images/train", rate=5,
                 duration=0, quality=85, region=None, val_split=0.15):
        self.output_dir = output_dir
        self.capture_rate = max(1, rate)
        self.duration = duration
        self.quality = quality
        self.region = region
        self.val_split = val_split
        self._lock = threading.Lock()
        self._frame_count = 0
        self._capture_active = threading.Event()
        self._exit_event = threading.Event()
        self._capture_start_time = 0.0
        self._last_hash = None

    @property
    def frame_count(self):
        with self._lock:
            return self._frame_count

    def _increment_frames(self):
        with self._lock:
            self._frame_count += 1
            return self._frame_count

    def setup_directories(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        val_dir = str(Path(self.output_dir).parent / "val")
        Path(val_dir).mkdir(parents=True, exist_ok=True)
        print(f"[+] Output: {self.output_dir}")
        print(f"[+] Val split: {self.val_split:.0%} -> {val_dir}")

    def _frame_is_duplicate(self, img):
        small = cv2.resize(img, (32, 32))
        h = hash(small.tobytes())
        if h == self._last_hash:
            return True
        self._last_hash = h
        return False

    def _pick_output_dir(self):
        if self.val_split > 0:
            import random
            if random.random() < self.val_split:
                return str(Path(self.output_dir).parent / "val")
        return self.output_dir

    def _check_disk_space(self):
        import shutil
        usage = shutil.disk_usage(self.output_dir)
        mb_free = usage.free / (1024 * 1024)
        if mb_free < 500:
            print(f"\n[!] Low disk space: {mb_free:.0f} MB remaining")
            return False
        return True

    def capture_loop(self):
        interval = 1.0 / self.capture_rate

        with mss.mss() as sct:
            if self.region:
                monitor = self.region
            else:
                monitor = sct.monitors[1]

            while self._capture_active.is_set() and not self._exit_event.is_set():
                if self.duration > 0:
                    elapsed = (time.monotonic() - self._capture_start_time) / 60.0
                    if elapsed >= self.duration:
                        print(f"\n[+] Duration limit reached ({self.duration} min)")
                        self._capture_active.clear()
                        break

                frame_start = time.monotonic()

                try:
                    screenshot = sct.grab(monitor)
                except Exception as e:
                    print(f"\n[!] Screen grab failed: {e}")
                    time.sleep(0.1)
                    continue

                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                if self._frame_is_duplicate(img):
                    time.sleep(0.01)
                    continue

                count_val = self.frame_count
                if count_val > 0 and count_val % 100 == 0 and not self._check_disk_space():
                    self._capture_active.clear()
                    break

                out_dir = self._pick_output_dir()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"frame_{timestamp}.jpg"
                filepath = os.path.join(out_dir, filename)

                encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                if not cv2.imwrite(filepath, img, encode_params):
                    print(f"\n[!] Failed to write: {filepath}")
                    continue

                count = self._increment_frames()
                if count % 10 == 0:
                    elapsed = time.monotonic() - self._capture_start_time
                    fps = count / elapsed if elapsed > 0 else 0
                    print(f"\r[+] Captured {count} frames ({fps:.1f} fps)", end="", flush=True)

                sleep_time = interval - (time.monotonic() - frame_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def run(self):
        print("[+] Starting data collection...")
        print("[+] Press F9 to start/stop capture")
        print("[+] Press ESC to exit")

        self.setup_directories()
        capture_thread = None

        try:
            f9_was_pressed = False
            while not self._exit_event.is_set():
                f9_down = keyboard.is_pressed("F9")

                if f9_down and not f9_was_pressed:
                    if not self._capture_active.is_set():
                        with self._lock:
                            self._frame_count = 0
                        self._capture_start_time = time.monotonic()
                        self._capture_active.set()
                        self._last_hash = None
                        print("\n[+] Capture started!")

                        capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
                        capture_thread.start()
                    else:
                        self._capture_active.clear()
                        if capture_thread:
                            capture_thread.join(timeout=2)
                        print(f"\n[+] Capture stopped. Total frames: {self.frame_count}")

                f9_was_pressed = f9_down

                if keyboard.is_pressed("ESC"):
                    self._exit_event.set()
                    self._capture_active.clear()
                    if capture_thread and capture_thread.is_alive():
                        capture_thread.join(timeout=2)
                    print(f"\n[+] Exiting. Total frames: {self.frame_count}")
                    break

                time.sleep(0.02)

        except KeyboardInterrupt:
            self._exit_event.set()
            self._capture_active.clear()
            if capture_thread and capture_thread.is_alive():
                capture_thread.join(timeout=2)
            print(f"\n[!] Interrupted. Saved {self.frame_count} frames.")


def parse_region(s):
    parts = s.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Region must be x,y,w,h")
    x, y, w, h = [int(p) for p in parts]
    return {"left": x, "top": y, "width": w, "height": h}


def main():
    parser = argparse.ArgumentParser(description="CS2 Data Collection Tool")
    parser.add_argument("--output", "-o", default="cs2_dataset/images/train",
                        help="Output directory for captured images")
    parser.add_argument("--rate", "-r", type=int, default=5,
                        help="Capture rate in frames per second (default: 5)")
    parser.add_argument("--duration", "-d", type=int, default=0,
                        help="Max capture duration in minutes (0 = unlimited)")
    parser.add_argument("--quality", "-q", type=int, default=85,
                        help="JPEG quality 1-100 (default: 85)")
    parser.add_argument("--region", type=parse_region, default=None,
                        help="Capture region as x,y,w,h (default: full screen)")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of frames routed to val/ (default: 0.15)")

    args = parser.parse_args()

    print("CS2 Data Collection Tool")
    print("========================")
    print(f"Output:    {args.output}")
    print(f"Rate:      {args.rate} fps")
    print(f"Quality:   {args.quality}")
    print(f"Val split: {args.val_split:.0%}")
    if args.duration > 0:
        print(f"Duration:  {args.duration} minutes")
    if args.region:
        print(f"Region:    {args.region}")
    print()

    collector = DataCollector(
        output_dir=args.output,
        rate=args.rate,
        duration=args.duration,
        quality=args.quality,
        region=args.region,
        val_split=args.val_split,
    )
    collector.run()


if __name__ == "__main__":
    main()
