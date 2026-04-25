import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class PhoneDetection:
	class_id: int
	class_name: str
	confidence: float
	box: Tuple[int, int, int, int]
	is_phone: bool


@dataclass
class PhoneFrameResult:
	detections: List[PhoneDetection]
	phone_detections: List[PhoneDetection]
	alert_triggered: bool
	n_phones: int
	annotated_frame: np.ndarray


class PhoneDetector:
	"""Runtime phone detector with alert-focused drawing rules.

	Notes:
	- Phone classes trigger alerts and are drawn prominently.
	- Driver-like context classes are optional and drawn subtly.
	- Wheel-like classes are hidden to reduce dashboard noise.
	"""

	PHONE_CLASSES = {"phone", "cell phone", "cellphone", "mobile"}
	HIDDEN_CLASSES = {"wheel", "steering wheel", "wheel_driver"}

	def __init__(
		self,
		model_path: str,
		conf_threshold: float = 0.25,
		device: str = "cpu",
		show_driver_context: bool = True,
	):
		self.model = YOLO(model_path)
		self.class_names = self.model.names
		self.conf_threshold = float(conf_threshold)
		self.device = device
		self.show_driver_context = show_driver_context

	def _class_name(self, class_id: int) -> str:
		names = self.class_names
		if isinstance(names, dict):
			return str(names.get(class_id, class_id))
		if isinstance(names, list) and 0 <= class_id < len(names):
			return str(names[class_id])
		return str(class_id)

	@classmethod
	def _is_phone(cls, name: str) -> bool:
		return name.lower() in cls.PHONE_CLASSES

	@classmethod
	def _is_hidden(cls, name: str) -> bool:
		return name.lower() in cls.HIDDEN_CLASSES

	def detect(self, frame: np.ndarray) -> PhoneFrameResult:
		result = self.model.predict(
			source=frame,
			conf=self.conf_threshold,
			device=self.device,
			verbose=False,
		)[0]

		detections: List[PhoneDetection] = []
		phone_detections: List[PhoneDetection] = []

		if result.boxes is not None:
			for box in result.boxes:
				class_id = int(box.cls[0])
				class_name = self._class_name(class_id)

				if self._is_hidden(class_name):
					continue

				is_phone = self._is_phone(class_name)
				det = PhoneDetection(
					class_id=class_id,
					class_name=class_name,
					confidence=float(box.conf[0]),
					box=tuple(int(v) for v in box.xyxy[0].tolist()),
					is_phone=is_phone,
				)
				detections.append(det)
				if is_phone:
					phone_detections.append(det)

		annotated = self.draw(frame.copy(), detections)
		return PhoneFrameResult(
			detections=detections,
			phone_detections=phone_detections,
			alert_triggered=len(phone_detections) > 0,
			n_phones=len(phone_detections),
			annotated_frame=annotated,
		)

	def draw(self, frame: np.ndarray, detections: List[PhoneDetection]) -> np.ndarray:
		phone_color = (0, 0, 220)
		driver_color = (60, 60, 60)

		for det in detections:
			x1, y1, x2, y2 = det.box
			if det.is_phone:
				cv2.rectangle(frame, (x1, y1), (x2, y2), phone_color, 3)
				label = f"phone {det.confidence:.2f}"
				(tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
				cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), phone_color, -1)
				cv2.putText(
					frame,
					label,
					(x1 + 3, y1 - 5),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,
					(255, 255, 255),
					2,
				)
			elif self.show_driver_context:
				cv2.rectangle(frame, (x1, y1), (x2, y2), driver_color, 1)

		has_phone = any(d.is_phone for d in detections)
		banner_color = phone_color if has_phone else (0, 170, 0)
		status_text = (
			f"PHONE DETECTED ({sum(1 for d in detections if d.is_phone)})"
			if has_phone
			else "SAFE"
		)
		cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), banner_color, -1)
		cv2.putText(
			frame,
			status_text,
			(10, 28),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.85,
			(255, 255, 255),
			2,
		)
		return frame

	def run_on_video(
		self,
		source: str,
		output_path: Optional[str] = None,
		display: bool = False,
		max_frames: Optional[int] = None,
		window_width: Optional[int] = None,
		window_height: Optional[int] = None,
	) -> Dict[str, float]:
		cap = cv2.VideoCapture(source)
		if not cap.isOpened():
			raise RuntimeError(f"Could not open video source: {source}")

		fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		limit = max_frames or total

		writer = None
		if output_path:
			Path(output_path).parent.mkdir(parents=True, exist_ok=True)
			writer = cv2.VideoWriter(
				output_path,
				cv2.VideoWriter_fourcc(*"mp4v"),
				fps,
				(width, height),
			)

		frame_count = 0
		frames_with_phone = 0
		timings_ms: List[float] = []
		window_name = "Phone Detection"

		if display:
			# WINDOW_NORMAL allows manual resizing by dragging the preview window edges.
			cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
			if window_width is not None and window_height is not None:
				cv2.resizeWindow(window_name, int(window_width), int(window_height))

		try:
			while cap.isOpened() and frame_count < limit:
				ok, frame = cap.read()
				if not ok:
					break

				t0 = time.perf_counter()
				res = self.detect(frame)
				timings_ms.append((time.perf_counter() - t0) * 1000.0)

				if res.alert_triggered:
					frames_with_phone += 1

				if writer is not None:
					writer.write(res.annotated_frame)

				if display:
					cv2.imshow(window_name, res.annotated_frame)
					if cv2.waitKey(1) & 0xFF == ord("q"):
						break

				frame_count += 1
		finally:
			cap.release()
			if writer is not None:
				writer.release()
			if display:
				cv2.destroyAllWindows()

		mean_latency = float(np.mean(timings_ms)) if timings_ms else 0.0
		return {
			"total_frames": frame_count,
			"frames_with_phone": frames_with_phone,
			"phone_rate": (frames_with_phone / frame_count) if frame_count else 0.0,
			"mean_fps": (1000.0 / mean_latency) if mean_latency else 0.0,
			"mean_latency_ms": mean_latency,
		}


class VideoAnalyzer:
	"""Video-level analysis wrapper for timeline and event extraction."""

	def __init__(self, detector: PhoneDetector):
		self.detector = detector

	def process(
		self,
		video_path: str,
		output_path: Optional[str] = None,
		max_frames: Optional[int] = None,
		frame_skip: int = 1,
	) -> List[Dict[str, Any]]:
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			raise RuntimeError(f"Could not open video: {video_path}")

		fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		limit = min(max_frames or total, total)

		writer = None
		if output_path:
			Path(output_path).parent.mkdir(parents=True, exist_ok=True)
			writer = cv2.VideoWriter(
				output_path,
				cv2.VideoWriter_fourcc(*"mp4v"),
				fps,
				(width, height),
			)

		log: List[Dict[str, Any]] = []
		prev_annotated = None
		idx = 0

		try:
			while cap.isOpened() and idx < limit:
				ok, frame = cap.read()
				if not ok:
					break

				if idx % max(frame_skip, 1) == 0:
					t0 = time.perf_counter()
					res = self.detector.detect(frame)
					latency_ms = (time.perf_counter() - t0) * 1000.0

					max_conf = max(
						(d.confidence for d in res.phone_detections),
						default=0.0,
					)
					log.append(
						{
							"frame_idx": idx,
							"timestamp_s": idx / fps,
							"alert": res.alert_triggered,
							"n_phones": res.n_phones,
							"max_conf": max_conf,
							"boxes": [d.box for d in res.phone_detections],
							"latency_ms": latency_ms,
						}
					)
					prev_annotated = res.annotated_frame
					if writer is not None:
						writer.write(res.annotated_frame)
				elif writer is not None:
					writer.write(prev_annotated if prev_annotated is not None else frame)

				idx += 1
		finally:
			cap.release()
			if writer is not None:
				writer.release()

		return log

	@staticmethod
	def to_dataframe(log: List[Dict[str, Any]]):
		try:
			import pandas as pd
		except ImportError as exc:
			raise ImportError("pandas is required for dataframe export") from exc
		return pd.DataFrame(log)

	@staticmethod
	def get_segments(log: List[Dict[str, Any]], gap_tolerance: int = 2) -> List[Dict[str, Any]]:
		if not log:
			return []

		segments: List[Dict[str, Any]] = []
		in_segment = False
		seg_frames: List[Dict[str, Any]] = []
		gap_count = 0

		for row in log:
			alert = bool(row.get("alert", False))

			if alert:
				if not in_segment:
					in_segment = True
					seg_frames = []
				gap_count = 0
				seg_frames.append(row)
				continue

			if in_segment:
				gap_count += 1
				if gap_count > gap_tolerance:
					if seg_frames:
						start = seg_frames[0]
						end = seg_frames[-1]
						segments.append(
							{
								"start_s": float(start["timestamp_s"]),
								"end_s": float(end["timestamp_s"]),
								"duration_s": float(end["timestamp_s"] - start["timestamp_s"]),
								"n_frames": len(seg_frames),
								"mean_conf": float(np.mean([f["max_conf"] for f in seg_frames])),
								"start_frame": int(start["frame_idx"]),
								"end_frame": int(end["frame_idx"]),
							}
						)
					in_segment = False
					seg_frames = []
					gap_count = 0

		if in_segment and seg_frames:
			start = seg_frames[0]
			end = seg_frames[-1]
			segments.append(
				{
					"start_s": float(start["timestamp_s"]),
					"end_s": float(end["timestamp_s"]),
					"duration_s": float(end["timestamp_s"] - start["timestamp_s"]),
					"n_frames": len(seg_frames),
					"mean_conf": float(np.mean([f["max_conf"] for f in seg_frames])),
					"start_frame": int(start["frame_idx"]),
					"end_frame": int(end["frame_idx"]),
				}
			)

		return segments
