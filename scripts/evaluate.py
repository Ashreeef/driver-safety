"""
Per-component latency benchmarking for the fatigue detection pipeline.

Usage:
    python scripts/evaluate.py                        # webcam, all modules
    python scripts/evaluate.py --source video.mp4     # video file
    python scripts/evaluate.py --frames 300           # stop after N frames
    python scripts/evaluate.py --no-compliance        # skip YOLO modules
    python scripts/evaluate.py --seatbelt-pipeline 3  # Pipeline 3
    python scripts/evaluate.py --output report.txt    # save report to file

Output: table of mean / p95 / p99 latency per component + total FPS.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import threading
import collections
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


# ── Timing accumulator ────────────────────────────────────────────────────────

class Stopwatch:
    def __init__(self, name: str):
        self.name = name
        self._samples: list[float] = []

    def record(self, elapsed_ms: float):
        self._samples.append(elapsed_ms)

    def stats(self):
        if not self._samples:
            return None
        a = np.array(self._samples)
        return {
            'n':    len(a),
            'mean': float(np.mean(a)),
            'p50':  float(np.percentile(a, 50)),
            'p95':  float(np.percentile(a, 95)),
            'p99':  float(np.percentile(a, 99)),
            'max':  float(np.max(a)),
        }


# ── Compliance worker (mirrors run_demo.py, with per-call timing) ─────────────

class ComplianceWorker:
    _DEFAULT = {
        'seatbelt_detected':      False,
        'smoking_detected':       False,
        'smoking_proximity_sec':  0.0,
        'smoking_hand_visible':   False,
        'smoking_wrist_ok':       True,
        'phone_detected':         False,
    }

    def __init__(self, seatbelt_pipeline=None, smoking_detector=None, phone_detector=None):
        self._sb      = seatbelt_pipeline
        self._sm      = smoking_detector
        self._ph      = phone_detector
        self._lock    = threading.Lock()
        self._frame   = None
        self._face_lm = None
        self._latest  = dict(self._DEFAULT)
        self._running = False
        self._thread  = None

        self.sw_seatbelt = Stopwatch('seatbelt')
        self.sw_smoking  = Stopwatch('smoking')
        self.sw_phone    = Stopwatch('phone')

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(self, frame: np.ndarray, face_landmarks):
        with self._lock:
            self._frame   = frame
            self._face_lm = face_landmarks

    def merge(self, result_dict: dict) -> dict:
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
                t0 = time.perf_counter()
                try:
                    _, scene = self._sb.process_frame(frame)
                    seatbelt_to_result(scene, update)
                except Exception as exc:
                    print(f"[ComplianceWorker] seatbelt error: {exc}")
                self.sw_seatbelt.record((time.perf_counter() - t0) * 1000)

            if self._sm is not None:
                t0 = time.perf_counter()
                try:
                    sm_res = self._sm.process(frame, face_landmarks=face_lm)
                    smoking_to_result(sm_res, update)
                except Exception as exc:
                    print(f"[ComplianceWorker] smoking error: {exc}")
                self.sw_smoking.record((time.perf_counter() - t0) * 1000)

            if self._ph is not None:
                t0 = time.perf_counter()
                try:
                    ph_res = self._ph.detect(frame)
                    phone_to_result(ph_res, update)
                except Exception as exc:
                    print(f"[ComplianceWorker] phone error: {exc}")
                self.sw_phone.record((time.perf_counter() - t0) * 1000)

            if update:
                with self._lock:
                    self._latest.update(update)


# ── Config / compliance loader (mirrors run_demo.py) ─────────────────────────

def load_configs():
    with open('configs/thresholds.yaml') as f:
        thresholds = yaml.safe_load(f)
    with open('configs/model_paths.yaml') as f:
        paths = yaml.safe_load(f)
    return thresholds, paths


def _try_load_compliance(args, paths, thresholds):
    mp_paths = paths.get('mediapipe', {})
    sb_paths = paths.get('seatbelt', {})
    sm_paths = paths.get('smoking', {})
    ph_paths = paths.get('phone', {})

    pipeline    = None
    smoking_det = None
    phone_det   = None

    if not args.no_compliance:
        sb_clf   = sb_paths.get('classifier', '')
        sb_yolo1 = sb_paths.get('yolo_p1', '')
        sb_yolo2 = sb_paths.get('yolo_p2', '')
        ptype    = str(args.seatbelt_pipeline)
        sb_yolo  = sb_yolo1 if ptype == '1' else sb_yolo2

        if os.path.exists(sb_yolo) and os.path.exists(sb_clf):
            try:
                with open('configs/seatbelt.yaml') as _f:
                    sb_cfg = yaml.safe_load(_f)
                if ptype == '1':
                    from src.compliance.seatbelt.pipelines import Pipeline1
                    pipeline = Pipeline1(sb_yolo, sb_clf, config=sb_cfg)
                    print("[evaluate] Seatbelt Pipeline 1 loaded.")
                elif ptype == '3':
                    from src.compliance.seatbelt.pipelines import Pipeline3
                    pipeline = Pipeline3(sb_yolo, config=sb_cfg)
                    print("[evaluate] Seatbelt Pipeline 3 loaded.")
                else:
                    from src.compliance.seatbelt.pipelines import Pipeline2
                    pipeline = Pipeline2(sb_yolo, sb_clf, config=sb_cfg, model_paths=paths)
                    print("[evaluate] Seatbelt Pipeline 2 loaded.")
            except Exception as exc:
                print(f"[evaluate] Seatbelt load failed: {exc}")
        else:
            print("[evaluate] Seatbelt weights not found — seatbelt disabled.")

    if not args.no_compliance and not args.no_smoking:
        sm_yolo   = sm_paths.get('yolo', '')
        hand_path = mp_paths.get('hand_landmarker', '')
        pose_path = mp_paths.get('pose_landmarker', '')
        if os.path.exists(hand_path) and os.path.exists(pose_path):
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
                print(f"[evaluate] Smoking detector loaded ({mode}).")
            except Exception as exc:
                print(f"[evaluate] Smoking load failed: {exc}")
        else:
            print("[evaluate] MediaPipe hand/pose models not found — smoking disabled.")

    if not args.no_compliance and not args.no_phone:
        ph_yolo = ph_paths.get('yolo', '')
        if os.path.exists(ph_yolo):
            try:
                with open('configs/phone.yaml') as _f:
                    ph_cfg = yaml.safe_load(_f)
                from src.compliance.phone import PhoneDetector
                det_cfg   = ph_cfg.get('detection', {})
                phone_det = PhoneDetector(
                    model_path=ph_yolo,
                    conf_threshold=float(det_cfg.get('conf_threshold', 0.25)),
                    device=str(det_cfg.get('device', 'cpu')),
                    show_driver_context=bool(
                        ph_cfg.get('visualization', {}).get('show_driver_context', True)
                    ),
                )
                print("[evaluate] Phone detector loaded.")
            except Exception as exc:
                print(f"[evaluate] Phone load failed: {exc}")
        else:
            print("[evaluate] Phone weights not found — phone disabled.")

    return pipeline, smoking_det, phone_det


# ── Report printer ────────────────────────────────────────────────────────────

def _fmt_row(name, stats, is_async=False):
    if stats is None:
        return f"  {name:<28}  {'(no data)':>8}"
    tag = ' [async]' if is_async else '        '
    return (
        f"  {name:<28}{tag}"
        f"  mean={stats['mean']:6.1f}ms"
        f"  p50={stats['p50']:6.1f}ms"
        f"  p95={stats['p95']:6.1f}ms"
        f"  p99={stats['p99']:6.1f}ms"
        f"  max={stats['max']:6.1f}ms"
        f"  n={stats['n']}"
    )


def print_report(stopwatches: dict, compliance_worker, total_frames: int,
                 elapsed_total: float, output_path: str | None = None):
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  FATIGUE DETECTION — LATENCY BENCHMARK REPORT")
    lines.append("=" * 80)

    fps = total_frames / elapsed_total if elapsed_total > 0 else 0
    lines.append(f"  Frames processed : {total_frames}")
    lines.append(f"  Wall time        : {elapsed_total:.1f}s")
    lines.append(f"  End-to-end FPS   : {fps:.1f}  (target ≥15)")
    lines.append("")
    lines.append("  ── Main-thread modules (synchronous) ──────────────────────────")

    main_keys = ['face_mesh', 'ear', 'mar', 'perclos', 'gaze', 'alert_engine']
    labels    = {
        'face_mesh':     'Face Mesh (MediaPipe)',
        'ear':           'EAR + EARTracker',
        'mar':           'MAR + MARTracker',
        'perclos':       'PERCLOS',
        'gaze':          'Gaze Estimator',
        'alert_engine':  'Alert Engine',
    }
    for key in main_keys:
        lines.append(_fmt_row(labels[key], stopwatches[key].stats()))

    main_total_ms = sum(
        stopwatches[k].stats()['mean']
        for k in main_keys
        if stopwatches[k].stats() is not None
    )
    lines.append(f"  {'─'*72}")
    lines.append(f"  {'Main-thread subtotal':<36}  mean={main_total_ms:6.1f}ms  →  {1000/main_total_ms:.1f} FPS budget")

    lines.append("")
    lines.append("  ── Compliance worker thread (asynchronous — does NOT block main) ──")
    if compliance_worker is not None:
        for sw_name, label in [
            ('sw_seatbelt', 'Seatbelt Pipeline'),
            ('sw_smoking',  'Smoking Detector'),
            ('sw_phone',    'Phone Detector'),
        ]:
            sw = getattr(compliance_worker, sw_name)
            lines.append(_fmt_row(label, sw.stats(), is_async=True))
    else:
        lines.append("  (compliance worker not started — run without --no-compliance)")

    lines.append("")
    lines.append("  ── Notes ──────────────────────────────────────────────────────────")
    lines.append("  [async] modules run in a daemon thread; they do NOT subtract from FPS.")
    lines.append("  Main-thread subtotal is the real per-frame budget.")
    lines.append(f"  FPS target: ≥15.  Headroom: {fps - 15:.1f} FPS {'OK' if fps >= 15 else '*** BELOW TARGET ***'}.")
    lines.append("=" * 80)
    lines.append("")

    report = "\n".join(lines)
    print(report)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"[evaluate] Report saved → {output_path}")


# ── Overlay drawing ──────────────────────────────────────────────────────────

def _draw_overlay(frame: np.ndarray, result: dict, sw: dict,
                  live_fps: float, measured: int, total: int) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_w = 260
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y    = 26
    step = 22

    def put(text, color=(185, 210, 185), scale=0.52, bold=False):
        nonlocal y
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 2 if bold else 1, cv2.LINE_AA)
        y += step

    # ── Header ────────────────────────────────────────────────────────────────
    fps_color = (80, 220, 80) if live_fps >= 15 else (60, 80, 220)
    put("EVALUATE MODE", color=(100, 200, 255), scale=0.58, bold=True)
    put(f"FPS {live_fps:5.1f}  frame {measured}" +
        (f"/{total}" if total > 0 else ""),
        color=fps_color)
    put("-" * 20, color=(60, 60, 60))

    # ── Face / fatigue metrics ─────────────────────────────────────────────────
    valid = result.get('valid', False)
    if valid:
        ear_val  = result.get('ear')
        mar_val  = result.get('mar')
        pcl_val  = result.get('perclos')
        gaze_dir = result.get('gaze_direction') or '?'
        cal      = result.get('ear_calibrated', False)
        bl       = result.get('ear_baseline')

        ear_str = f"EAR {ear_val:.3f}" if ear_val is not None else "EAR ---"
        mar_str = f"MAR {mar_val:.3f}" if mar_val is not None else "MAR ---"
        pcl_str = (f"PERCLOS {pcl_val:.2f}" if pcl_val is not None else "PERCLOS buf...")
        cal_str = (f"BL={bl:.3f}" if (cal and bl) else
                   f"CAL {result.get('ear_calibrated', 0)*100:.0f}%")

        put(ear_str + f"  {cal_str}", color=(170, 220, 170))
        put(mar_str, color=(170, 220, 170))
        put(pcl_str, color=(170, 220, 170))
        put(f"Gaze {gaze_dir}", color=(170, 220, 170))
    else:
        put("No face detected", color=(100, 100, 200))
        y += step * 3

    # ── Compliance ─────────────────────────────────────────────────────────────
    put("-" * 20, color=(60, 60, 60))
    sb  = result.get('seatbelt_detected', False)
    sm  = result.get('smoking_detected',  False)
    ph  = result.get('phone_detected',    False)
    put(f"Seatbelt {'OK' if sb else 'OFF'}",
        color=(80, 200, 80) if sb else (60, 60, 220))
    put(f"Smoking  {'YES' if sm else 'clear'}",
        color=(60, 60, 220) if sm else (130, 160, 130))
    put(f"Phone    {'YES' if ph else 'clear'}",
        color=(60, 60, 220) if ph else (130, 160, 130))

    # ── Alerts ────────────────────────────────────────────────────────────────
    alerts = result.get('alerts', [])
    if alerts:
        put("-" * 20, color=(60, 60, 60))
        for alert in alerts[:4]:
            is_crit = 'CRITICAL' in alert
            put(f"! {alert}", color=(40, 40, 230) if is_crit else (60, 130, 230),
                scale=0.50, bold=is_crit)

    # ── Per-module timing strip at bottom ────────────────────────────────────
    strip_y = h - 14
    labels  = [('FM', 'face_mesh'), ('EAR', 'ear'), ('MAR', 'mar'),
               ('PCL', 'perclos'), ('Gaze', 'gaze')]
    x_cur   = 6
    for label, key in labels:
        s = sw[key].stats()
        ms_str = f"{s['mean']:.0f}ms" if s else "--"
        txt = f"{label}:{ms_str}"
        cv2.putText(frame, txt, (x_cur, strip_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
        x_cur += len(txt) * 7 + 4

    return frame


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Per-component latency benchmark')
    parser.add_argument('--source', default=None,
                        help='Camera index or video file path (e.g. video.mp4)')
    parser.add_argument('--frames', type=int, default=0,
                        help='Stop after N frames (0 = run until q or end of file)')
    parser.add_argument('--no-compliance', action='store_true',
                        help='Skip all compliance modules')
    parser.add_argument('--no-smoking', action='store_true',
                        help='Skip smoking detection')
    parser.add_argument('--no-phone', action='store_true',
                        help='Skip phone detection')
    parser.add_argument('--seatbelt-pipeline', choices=['1', '2', '3'], default='2')
    parser.add_argument('--output', default=None,
                        help='Save text report to this file')
    parser.add_argument('--save-video', default=None, metavar='PATH',
                        help='Save annotated output video (e.g. out.mp4)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable the live preview window (useful for headless runs)')
    parser.add_argument('--warmup', type=int, default=30,
                        help='Frames to skip before recording (default 30)')
    args = parser.parse_args()

    thresholds, paths = load_configs()
    fps_cfg = thresholds.get('camera_fps', 15)

    print("[evaluate] Loading modules...")
    face_mesh   = FaceMeshDetector(thresholds, paths)
    calibrator  = EARCalibrator(thresholds, fps=fps_cfg)
    ear_tracker = EARTracker(thresholds, calibrator)
    mar_tracker = MARTracker(thresholds)
    perclos     = PERCLOSTracker(thresholds, calibrator, fps=fps_cfg)
    gaze        = GazeEstimator(thresholds)
    alert_engine = AlertEngine()

    sb_pipeline, sm_detector, ph_detector = _try_load_compliance(args, paths, thresholds)
    compliance_worker = None
    if sb_pipeline is not None or sm_detector is not None or ph_detector is not None:
        compliance_worker = ComplianceWorker(sb_pipeline, sm_detector, ph_detector)
        compliance_worker.start()
        print("[evaluate] Compliance worker thread started.")

    # ── Camera / video setup ──────────────────────────────────────────────────
    is_video_file = False
    if args.source is not None:
        raw = args.source
        if raw.isdigit():
            source = int(raw)
        else:
            source = raw
            is_video_file = os.path.isfile(raw)
    else:
        source = thresholds.get('camera_source', 0)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[evaluate] ERROR: cannot open source '{source}'")
        return

    # For video files use native resolution; for live camera force config values
    if is_video_file:
        vid_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or fps_cfg
        total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[evaluate] Video: {vid_w}x{vid_h} @ {vid_fps:.1f}fps  ({total_vid_frames} frames)")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  thresholds.get('camera_width',  640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, thresholds.get('camera_height', 480))
        cap.set(cv2.CAP_PROP_FPS, fps_cfg)
        vid_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps = fps_cfg
        total_vid_frames = 0

    # ── Optional video writer ─────────────────────────────────────────────────
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video, fourcc, vid_fps, (vid_w, vid_h))
        print(f"[evaluate] Saving annotated video → {args.save_video}")

    show = not args.no_display

    # ── Stopwatches ───────────────────────────────────────────────────────────
    sw = {k: Stopwatch(k) for k in ['face_mesh', 'ear', 'mar', 'perclos', 'gaze', 'alert_engine']}

    frame_count  = 0
    warmup_done  = False
    wall_start   = None

    print(f"[evaluate] Warmup: {args.warmup} frames. Press 'q' to stop early.")
    print()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count <= args.warmup:
                print(f"\r  Warming up... {frame_count}/{args.warmup}", end='', flush=True)
                result = face_mesh.process_frame(frame)
                result = ear_tracker.update(result)
                result = mar_tracker.update(result)
                result = perclos.update(result)
                result = gaze.update(result)
                if compliance_worker is not None:
                    compliance_worker.push(frame, result.get('landmarks'))
                    compliance_worker.merge(result)
                alert_engine.process(result)

                if show:
                    wu_frame = frame.copy()
                    cv2.putText(wu_frame, f"Warming up... {frame_count}/{args.warmup}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (100, 200, 255), 2, cv2.LINE_AA)
                    cv2.imshow('evaluate — driver monitor', wu_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if frame_count == args.warmup:
                    print(f"\r  Warmup done. Recording timing now...          ")
                    wall_start  = time.perf_counter()
                    warmup_done = True
                continue

            # ── Timed run ─────────────────────────────────────────────────────
            t0 = time.perf_counter()
            result = face_mesh.process_frame(frame)
            sw['face_mesh'].record((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            result = ear_tracker.update(result)
            sw['ear'].record((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            result = mar_tracker.update(result)
            sw['mar'].record((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            result = perclos.update(result)
            sw['perclos'].record((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            result = gaze.update(result)
            sw['gaze'].record((time.perf_counter() - t0) * 1000)

            if compliance_worker is not None:
                compliance_worker.push(frame, result.get('landmarks'))
                compliance_worker.merge(result)

            t0 = time.perf_counter()
            alert_engine.process(result)
            sw['alert_engine'].record((time.perf_counter() - t0) * 1000)

            # ── Timing stats ──────────────────────────────────────────────────
            measured = frame_count - args.warmup
            elapsed  = time.perf_counter() - wall_start
            live_fps = measured / elapsed if elapsed > 0 else 0

            # ── Overlay (for display and/or video save) ───────────────────────
            if show or writer is not None:
                frame = _draw_overlay(frame, result, sw, live_fps,
                                      measured, total_vid_frames - args.warmup)

            if writer is not None:
                writer.write(frame)

            if show:
                cv2.imshow('evaluate — driver monitor', frame)

            # ── Terminal status line ──────────────────────────────────────────
            fm_ms  = sw['face_mesh'].stats()
            fm_str = f"{fm_ms['mean']:.1f}ms" if fm_ms else "---"
            pct    = f" ({measured*100//(total_vid_frames-args.warmup)}%)" \
                     if total_vid_frames > args.warmup else ""
            status = (
                f"  Frame {measured:>5}{pct}  |  FPS {live_fps:5.1f}  |  "
                f"FaceMesh {fm_str}  |  q=quit"
            )
            print(f"\r{status}", end='', flush=True)

            if args.frames > 0 and measured >= args.frames:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[evaluate] Interrupted.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"\n[evaluate] Annotated video saved → {args.save_video}")
        cv2.destroyAllWindows()
        if compliance_worker is not None:
            compliance_worker.stop()

    print()

    measured_frames = frame_count - args.warmup
    if measured_frames <= 0 or wall_start is None:
        print("[evaluate] No frames measured — increase --frames or use a longer video.")
        return

    elapsed_total = time.perf_counter() - wall_start
    print_report(sw, compliance_worker, measured_frames, elapsed_total, args.output)


if __name__ == '__main__':
    main()
