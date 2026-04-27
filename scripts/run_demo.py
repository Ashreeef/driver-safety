import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import threading
import time
import cv2
import yaml
import numpy as np
from src.face_mesh.mediapipe_pipeline import FaceMeshDetector
from src.fatigue.ear import EARCalibrator, EARTracker
from src.fatigue.mar import MARTracker
from src.fatigue.perclos import PERCLOSTracker
from src.gaze.gaze_estimator import GazeEstimator
from src.compliance.adapters import seatbelt_to_result, smoking_to_result, phone_to_result
from src.fusion.alert_engine import AlertEngine


# ── Compliance worker (runs in its own daemon thread) ─────────────────────────

class ComplianceWorker:
    """
    Runs seatbelt and/or smoking detection in a background daemon thread so
    that the main face-mesh loop is not blocked by the heavier YOLO/pose
    inference.  The worker always operates on the most recently pushed frame.

    Thread safety: all shared state is guarded by a single Lock.
    """

    _DEFAULT = {
        'seatbelt_detected':      False,
        'smoking_detected':       False,
        'smoking_proximity_sec':  0.0,
        'smoking_hand_visible':   False,
        'smoking_wrist_ok':       True,
        'phone_detected':         False,
    }

    def __init__(self, seatbelt_pipeline=None, smoking_detector=None, phone_detector=None):
        self._sb     = seatbelt_pipeline
        self._sm     = smoking_detector
        self._ph     = phone_detector
        self._lock   = threading.Lock()
        self._frame  = None
        self._face_lm = None
        self._latest = dict(self._DEFAULT)
        self._running = False
        self._thread  = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(self, frame: np.ndarray, face_landmarks):
        """Push the latest frame and pre-computed face landmarks."""
        with self._lock:
            self._frame   = frame
            self._face_lm = face_landmarks

    def merge(self, result_dict: dict) -> dict:
        """Copy the latest compliance results into result_dict."""
        with self._lock:
            result_dict.update(self._latest)
        return result_dict

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        while self._running:
            with self._lock:
                frame   = self._frame
                face_lm = self._face_lm

            if frame is None:
                time.sleep(0.005)
                continue

            update = {}

            if self._sb is not None:
                try:
                    _, scene = self._sb.process_frame(frame)
                    seatbelt_to_result(scene, update)
                except Exception as exc:
                    print(f"[ComplianceWorker] seatbelt error: {exc}")

            if self._sm is not None:
                try:
                    sm_res = self._sm.process(frame, face_landmarks=face_lm)
                    smoking_to_result(sm_res, update)
                except Exception as exc:
                    print(f"[ComplianceWorker] smoking error: {exc}")

            if self._ph is not None:
                try:
                    ph_res = self._ph.detect(frame)
                    phone_to_result(ph_res, update)
                except Exception as exc:
                    print(f"[ComplianceWorker] phone error: {exc}")

            if update:
                with self._lock:
                    self._latest.update(update)


# ── Config helpers ────────────────────────────────────────────────────────────

def load_configs():
    with open('configs/thresholds.yaml') as f:
        thresholds = yaml.safe_load(f)
    with open('configs/model_paths.yaml') as f:
        paths = yaml.safe_load(f)
    return thresholds, paths


def _try_load_compliance(args, paths, thresholds):
    """
    Attempt to load seatbelt, smoking, and phone modules.
    Returns (pipeline, smoking_det, phone_det) — any may be None.
    """
    mp_paths = paths.get('mediapipe', {})
    sb_paths = paths.get('seatbelt', {})
    sm_paths = paths.get('smoking', {})
    ph_paths = paths.get('phone', {})

    pipeline    = None
    smoking_det = None
    phone_det   = None

    # ── Seatbelt ─────────────────────────────────────────────────────────────
    if not args.no_compliance:
        sb_clf   = sb_paths.get('classifier', '')
        sb_yolo1 = sb_paths.get('yolo_p1', '')   # YOLOv5s — Pipeline 1 ROI
        sb_yolo2 = sb_paths.get('yolo_p2', '')   # YOLOv8n — Pipeline 2/3

        ptype = str(args.seatbelt_pipeline)
        sb_yolo = sb_yolo1 if ptype == '1' else sb_yolo2

        if not os.path.exists(sb_yolo) or not os.path.exists(sb_clf):
            print(f"[run_demo] Seatbelt weights not found — seatbelt disabled.")
            print(f"  yolo:       {sb_yolo}")
            print(f"  classifier: {sb_clf}")
        else:
            try:
                with open('configs/seatbelt.yaml') as _f:
                    sb_cfg = yaml.safe_load(_f)

                if ptype == '1':
                    from src.compliance.seatbelt.pipelines import Pipeline1
                    pipeline = Pipeline1(sb_yolo, sb_clf, config=sb_cfg)
                    print("[run_demo] Seatbelt Pipeline 1 loaded (YOLOv5s ROI + CNN).")
                elif ptype == '3':
                    from src.compliance.seatbelt.pipelines import Pipeline3
                    pipeline = Pipeline3(sb_yolo, config=sb_cfg)
                    print("[run_demo] Seatbelt Pipeline 3 loaded (YOLOv8n direct).")
                else:
                    from src.compliance.seatbelt.pipelines import Pipeline2
                    pipeline = Pipeline2(sb_yolo, sb_clf, config=sb_cfg,
                                        model_paths=paths)
                    print("[run_demo] Seatbelt Pipeline 2 loaded (Pose+YOLOv8n+CNN, default).")
            except Exception as exc:
                print(f"[run_demo] Seatbelt load failed: {exc}")

    # ── Smoking ───────────────────────────────────────────────────────────────
    if not args.no_compliance and not args.no_smoking:
        sm_yolo = sm_paths.get('yolo', '')
        hand_path = mp_paths.get('hand_landmarker', '')
        pose_path = mp_paths.get('pose_landmarker', '')

        if not os.path.exists(hand_path) or not os.path.exists(pose_path):
            print(f"[run_demo] MediaPipe hand/pose models not found — smoking disabled.")
            print(f"  hand: {hand_path}")
            print(f"  pose: {pose_path}")
        else:
            try:
                with open('configs/smoking.yaml') as _f:
                    sm_cfg = yaml.safe_load(_f)

                from src.compliance.smoking.landmarks import LandmarkExtractor
                from src.compliance.smoking.detector import SmokingDetector
                extractor   = LandmarkExtractor(
                    sm_cfg.get('landmarks', {}),
                    model_paths=mp_paths,
                    use_external_face_landmarks=True,
                )
                smoking_det = SmokingDetector(
                    sm_cfg, extractor,
                    yolo_weights=sm_yolo if os.path.exists(sm_yolo) else None,
                )
                mode = "YOLO+landmark" if os.path.exists(sm_yolo) else "landmark-only"
                print(f"[run_demo] Smoking detector loaded ({mode}).")
            except Exception as exc:
                print(f"[run_demo] Smoking load failed: {exc}")

    # ── Phone ─────────────────────────────────────────────────────────────────
    if not args.no_compliance and not args.no_phone:
        ph_yolo = ph_paths.get('yolo', '')
        if not os.path.exists(ph_yolo):
            print(f"[run_demo] Phone weights not found — phone detection disabled.")
            print(f"  yolo: {ph_yolo}")
        else:
            try:
                with open('configs/phone.yaml') as _f:
                    ph_cfg = yaml.safe_load(_f)
                from src.compliance.phone import PhoneDetector
                det_cfg = ph_cfg.get('detection', {})
                phone_det = PhoneDetector(
                    model_path=ph_yolo,
                    conf_threshold=float(det_cfg.get('conf_threshold', 0.25)),
                    device=str(det_cfg.get('device', 'cpu')),
                    show_driver_context=bool(
                        ph_cfg.get('visualization', {}).get('show_driver_context', True)
                    ),
                )
                print("[run_demo] Phone detector loaded.")
            except Exception as exc:
                print(f"[run_demo] Phone load failed: {exc}")

    return pipeline, smoking_det, phone_det


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Driver fatigue detection demo')
    parser.add_argument('--source', default=None,
                        help='Video source: camera index (0, 1, …) or path to video file')
    parser.add_argument('--output', default=None,
                        help='Path for annotated output video (e.g. out.mp4). Omit to disable.')
    parser.add_argument('--no-compliance', action='store_true',
                        help='Disable all compliance modules (seatbelt + smoking)')
    parser.add_argument('--no-smoking', action='store_true',
                        help='Disable smoking detection only')
    parser.add_argument('--no-phone', action='store_true',
                        help='Disable phone detection only')
    parser.add_argument('--seatbelt-pipeline', choices=['1', '2', '3'], default='2',
                        help='Seatbelt pipeline: 1=YOLO+CNN, 2=Pose+YOLO+CNN+EMA (default), 3=Direct YOLO')
    args = parser.parse_args()

    thresholds, paths = load_configs()
    fps = thresholds.get('camera_fps', 15)

    print("Initializing core modules...")
    face_mesh   = FaceMeshDetector(thresholds, paths)
    calibrator  = EARCalibrator(thresholds, fps=fps)
    ear_tracker = EARTracker(thresholds, calibrator)
    mar_tracker = MARTracker(thresholds)
    perclos     = PERCLOSTracker(thresholds, calibrator, fps=fps)
    gaze        = GazeEstimator(thresholds)

    # ── Compliance (optional) ─────────────────────────────────────────────────
    sb_pipeline, sm_detector, ph_detector = _try_load_compliance(args, paths, thresholds)
    compliance_worker = None
    if sb_pipeline is not None or sm_detector is not None or ph_detector is not None:
        compliance_worker = ComplianceWorker(sb_pipeline, sm_detector, ph_detector)
        compliance_worker.start()
        print("[run_demo] Compliance worker thread started.")

    # ── Camera / source ───────────────────────────────────────────────────────
    if args.source is not None:
        raw = args.source
        source = int(raw) if raw.isdigit() else raw
    else:
        source = thresholds.get('camera_source', 0)

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  thresholds.get('camera_width',  640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, thresholds.get('camera_height', 480))
    cap.set(cv2.CAP_PROP_FPS, fps)

    writer = None
    if args.output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'),
                                 real_fps, (w, h))
        print(f"Recording annotated output → {args.output}")

    alert_engine = AlertEngine()

    print("Press 'q' to quit  |  'r' to reset gaze calibration.")

    _fps_t0      = time.time()
    _fps_count   = 0
    _fps_display = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── FPS counter ───────────────────────────────────────────────────────
        _fps_count += 1
        _fps_elapsed = time.time() - _fps_t0
        if _fps_elapsed >= 1.0:
            _fps_display = _fps_count / _fps_elapsed
            _fps_count   = 0
            _fps_t0      = time.time()

        # ── Core modules (synchronous, main thread) ───────────────────────────
        result = face_mesh.process_frame(frame)
        result = ear_tracker.update(result)
        result = mar_tracker.update(result)
        result = perclos.update(result)
        result = gaze.update(result)

        # ── Compliance (async results from worker thread) ─────────────────────
        if compliance_worker is not None:
            compliance_worker.push(frame, result.get('landmarks'))
            result = compliance_worker.merge(result)

        # ── Alert engine — escalation, compliance alerts, priority sort ────────
        result = alert_engine.process(result)

        # ── Overlay ───────────────────────────────────────────────────────────
        h, w = frame.shape[:2]
        panel_w = 245
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        y = 28
        step = 24

        def put(text, color=(190, 210, 190), bold=False):
            nonlocal y
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color,
                        2 if bold else 1, cv2.LINE_AA)
            y += step

        fps_color = (80, 220, 80) if _fps_display >= 15 else (0, 80, 255)
        put("DRIVER MONITOR", color=(100, 200, 255), bold=True)
        put(f"FPS  {_fps_display:.1f}", color=fps_color)
        put("-" * 16, color=(70, 70, 70))

        if result['valid']:
            cal    = result.get('ear_calibrated', False)
            bl     = result.get('ear_baseline')
            ear_val = result.get('ear')
            mar_val = result.get('mar')
            perclos_val = result.get('perclos')

            cal_str = f"BL={bl:.3f}" if (cal and bl is not None) else \
                      f"CAL {calibrator.progress() * 100:.0f}%"

            if ear_val is None:
                put("EAR  --", color=(170, 170, 170))
            else:
                ear_color = (0, 80, 255) if ear_val < calibrator.alert_threshold \
                            else (100, 220, 100)
                put(f"EAR  {ear_val:.3f}  {cal_str}", color=ear_color)

            mar_thr = thresholds.get('mar_threshold', 0.5)
            mar_color = (0, 80, 255) if (mar_val is not None and mar_val > mar_thr) \
                        else (100, 220, 100)
            put(f"MAR  {mar_val:.3f}" if mar_val is not None else "MAR  --",
                color=mar_color)

            if perclos_val is None:
                put("PERCLOS  buffering...", color=(180, 180, 60))
            else:
                pct = perclos_val * 100
                if pct < 8:
                    pc_color, label = (100, 220, 100), "OK"
                elif pct < 15:
                    pc_color, label = (0, 165, 255), "MILD"
                elif pct < 25:
                    pc_color, label = (0, 80, 255), "MOD"
                else:
                    pc_color, label = (0, 0, 255), "HIGH"
                put(f"PERCLOS  {pct:.1f}%  {label}", color=pc_color)

            put(f"Yawns  {result.get('yawn_count', 0)}", color=(180, 180, 180))
            put("-" * 16, color=(70, 70, 70))

            gaze_dir  = result.get('gaze_direction', '--')
            gaze_cal  = result.get('gaze_calibrated', False)
            gaze_prog = result.get('gaze_calib_progress', 0.0)
            gaze_h    = result.get('gaze_h_ratio')
            gaze_v    = result.get('gaze_v_ratio')

            head_yaw_thresh = thresholds.get('gaze_head_yaw_threshold', 20.0)
            head_pd_thresh  = thresholds.get('gaze_head_pitch_down_threshold', 15.0)
            head_pu_thresh  = thresholds.get('gaze_head_pitch_up_threshold', 10.0)
            yaw_abs   = abs(result.get('yaw', 0.0))
            pitch_v   = result.get('pitch', 0.0)
            head_driven = (yaw_abs > head_yaw_thresh or
                           pitch_v > head_pd_thresh or
                           pitch_v < -head_pu_thresh)
            gaze_signal = "H" if head_driven else ("I" if gaze_cal else "?")

            if not gaze_cal and not head_driven:
                put(f"GAZE CAL  {gaze_prog * 100:.0f}%", color=(180, 180, 60))
                put("Look straight ahead", color=(150, 150, 150))
            else:
                gaze_color = (100, 220, 100) if gaze_dir == 'forward' else (0, 80, 255)
                put(f"Gaze  {gaze_dir}  [{gaze_signal}]", color=gaze_color)
                if gaze_h is not None and gaze_v is not None and not head_driven:
                    put(f"h={gaze_h:.3f}  v={gaze_v:.3f}", color=(150, 150, 150))

            put(f"Head  P:{result['pitch']:.0f} Y:{result['yaw']:.0f} "
                f"[{result['head_pose_method'][:3]}]",
                color=(160, 160, 160))

            # Compliance status
            if compliance_worker is not None:
                put("-" * 16, color=(70, 70, 70))
                sb_ok = result.get('seatbelt_detected', False)
                sm_on = result.get('smoking_detected', False)
                ph_on = result.get('phone_detected', False)
                put(f"Seatbelt  {'ON' if sb_ok else 'OFF'}",
                    color=(100, 220, 100) if sb_ok else (0, 80, 255))
                put(f"Smoking   {'YES' if sm_on else 'clear'}",
                    color=(0, 0, 255) if sm_on else (100, 220, 100))
                if ph_detector is not None:
                    put(f"Phone     {'YES' if ph_on else 'clear'}",
                        color=(0, 0, 255) if ph_on else (100, 220, 100))

            alerts = result.get('alerts', [])
            if alerts:
                put("-" * 16, color=(70, 70, 70))
                put("ALERTS", color=(0, 120, 255), bold=True)
                for alert in alerts[:3]:
                    put(f"! {alert}", color=(0, 0, 255))
        else:
            put("NO FACE / ANGLE EXCEEDED", color=(0, 0, 255), bold=True)
            put(f"Pitch:{result['pitch']:.0f}  Yaw:{result['yaw']:.0f}",
                color=(0, 140, 255))

        if writer is not None:
            writer.write(frame)

        cv2.imshow('Fatigue Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            gaze.reset_calibration()
            print("[run_demo] Gaze calibration reset.")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated video → {args.output}")
    if compliance_worker is not None:
        compliance_worker.stop()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == '__main__':
    main()
