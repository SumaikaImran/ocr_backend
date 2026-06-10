import cv2
import numpy as np
import time
import os
import threading
import queue
from paddleocr import PaddleOCR

os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

class EfficientTextReader:
    def __init__(self, mobile_mode=True):
        self.mobile_mode = mobile_mode

        self.ocr = PaddleOCR(lang='en')

        self.roi_scale = 0.5
        self.text_cache = {}
        self.cache_timeout = 5

        self.frame_count = 0
        self.process_every_n = 3

        self.roi_width = 0.7
        self.roi_height = 0.7

        # Thread-safe queue: main thread puts frames, OCR thread reads them
        self.frame_queue = queue.Queue(maxsize=1)

        # OCR results come back here
        self.result_queue = queue.Queue(maxsize=5)

        # Latest overlay drawn by OCR thread
        self.latest_overlay = None
        self.overlay_lock = threading.Lock()

        # Start background OCR thread
        self.running = True
        self.ocr_thread = threading.Thread(target=self._ocr_worker, daemon=True)
        self.ocr_thread.start()

        print("PaddleOCR initialized — running OCR in background thread")

    def _ocr_worker(self):
        """Runs in background. Pulls frames from queue, runs OCR, stores overlay."""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                processed = self._preprocess(frame)
                result = self.ocr.ocr(processed, cls=False)
                items = result[0] if result and result[0] else []

                roi_offset = self._get_roi_offset(frame)
                overlay, texts = self._build_overlay(frame, items, roi_offset)

                with self.overlay_lock:
                    self.latest_overlay = (overlay, texts)

                # Put detected texts into result queue for printing
                for t in texts:
                    try:
                        self.result_queue.put_nowait(t)
                    except queue.Full:
                        pass

            except Exception as e:
                print(f"OCR Error: {e}")

    def _get_roi_offset(self, frame):
        h, w = frame.shape[:2]
        x1 = int(w * (1 - self.roi_width) / 2)
        y1 = int(h * (1 - self.roi_height) / 2)
        return (x1, y1)

    def _preprocess(self, frame):
        """Crop ROI, resize, enhance contrast, return BGR for PaddleOCR."""
        h, w = frame.shape[:2]
        x1 = int(w * (1 - self.roi_width) / 2)
        y1 = int(h * (1 - self.roi_height) / 2)
        x2 = int(w - x1)
        y2 = int(h - y1)
        roi = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if self.mobile_mode:
            new_w = int(gray.shape[1] * self.roi_scale)
            new_h = int(gray.shape[0] * self.roi_scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def _build_overlay(self, base_frame, items, roi_offset):
        """Draw bounding boxes and labels on a copy of the base frame."""
        annotated = base_frame.copy()
        detected_texts = []
        current_time = time.time()

        for item in items:
            if len(item) < 2:
                continue
            points = item[0]
            if not points:
                continue
            if not (isinstance(item[1], tuple) and len(item[1]) >= 2):
                continue

            text, confidence = item[1][0], item[1][1]
            if confidence < 0.5:
                continue

            adjusted_points = []
            for point in points:
                if len(point) >= 2:
                    x, y = point[0], point[1]
                    adj_x = int(x / self.roi_scale + roi_offset[0])
                    adj_y = int(y / self.roi_scale + roi_offset[1])
                    adjusted_points.append([adj_x, adj_y])

            if not adjusted_points:
                continue

            text_hash = hash(text.lower().strip())
            if text_hash not in self.text_cache:
                self.text_cache[text_hash] = {
                    'text': text,
                    'confidence': confidence,
                    'timestamp': current_time,
                }
                detected_texts.append(text)
            else:
                # Refresh timestamp so it stays visible
                self.text_cache[text_hash]['timestamp'] = current_time

            pts = np.array(adjusted_points, dtype=np.int32)
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
            text_position = (adjusted_points[0][0], adjusted_points[0][1] - 5)
            cv2.putText(
                annotated,
                f"{text[:25]} ({confidence:.2f})",
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        # Clean expired cache entries
        expired = [k for k, v in self.text_cache.items()
                   if current_time - v['timestamp'] > self.cache_timeout]
        for k in expired:
            del self.text_cache[k]

        return annotated, detected_texts

    def process_frame(self, frame):
        """
        Non-blocking: sends frame to OCR thread if slot is free,
        returns the latest available overlay immediately.
        """
        self.frame_count += 1

        # Only submit every N-th frame to keep OCR load manageable
        if self.frame_count % self.process_every_n == 0:
            try:
                self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass  # OCR thread still busy — skip this frame, no hang

        # Return last known overlay (camera stays smooth)
        with self.overlay_lock:
            if self.latest_overlay is not None:
                return self.latest_overlay[0], []
        return frame, []

    def stop(self):
        self.running = False
        self.ocr_thread.join(timeout=2)


def main():
    reader = EfficientTextReader(mobile_mode=True)
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Assistive Text Reader Started")
    print("Focus camera on text and hold steady for best results")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _ = reader.process_frame(frame)

        # Print any newly detected text (non-blocking)
        while True:
            try:
                text = reader.result_queue.get_nowait()
                print(f"Detected: {text}")
            except queue.Empty:
                break

        # Draw hint if nothing detected yet
        if reader.latest_overlay is None:
            cv2.putText(
                annotated,
                "Hold camera steady on text",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow("Assistive Text Reader", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    reader.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()