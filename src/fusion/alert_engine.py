import time


class AlertEngine:
    """
    Post-processes result['alerts'] after all modules have run each frame.

    Responsibilities:
    - Adds compliance alerts ('Seatbelt OFF', 'Smoking detected') from booleans
    - Escalates to 'CRITICAL FATIGUE' when 2+ fatigue signals fire simultaneously
    - Suppresses individual fatigue sub-alerts when CRITICAL subsumes them
    - Deduplicates and sorts by priority (highest first)

    No audio, no cross-frame state beyond a startup timestamp.
    """

    # Two or more of these firing simultaneously → CRITICAL FATIGUE
    _FATIGUE_ESCALATION = frozenset({
        'Drowsiness (EAR)',
        'Drowsiness (PERCLOS)',
        'Fatigue (EAR Trend)',
        'Fatigue (Yawn Frequency)',
    })

    # Higher number = shown first; unlisted alerts default to 1
    _PRIORITY = {
        'CRITICAL FATIGUE':         10,
        'Drowsiness (EAR)':          5,
        'Drowsiness (PERCLOS)':      5,
        'Fatigue (EAR Trend)':       4,
        'Fatigue (Yawn Frequency)':  3,
        'Yawning (MAR)':             2,
        'Distraction (Gaze)':        2,
        'Seatbelt OFF':              2,
        'Smoking detected':          2,
        'Phone detected':            2,
    }

    # Don't alert seatbelt-off during pipeline warmup
    _SEATBELT_WARMUP_SEC = 5.0

    def __init__(self):
        self._start = time.time()

    def process(self, result_dict: dict) -> dict:
        raw = list(result_dict.get('alerts', []))

        # ── Compliance alerts ─────────────────────────────────────────────────
        elapsed = time.time() - self._start
        if elapsed > self._SEATBELT_WARMUP_SEC and not result_dict.get('seatbelt_detected', True):
            raw.append('Seatbelt OFF')
        if result_dict.get('smoking_detected', False):
            raw.append('Smoking detected')
        if result_dict.get('phone_detected', False):
            raw.append('Phone detected')

        # ── Cross-module escalation ───────────────────────────────────────────
        active_fatigue = set(raw) & self._FATIGUE_ESCALATION
        if len(active_fatigue) >= 2:
            for s in active_fatigue:
                raw.remove(s)
            raw.insert(0, 'CRITICAL FATIGUE')

        # ── Deduplicate (preserve first occurrence order) ─────────────────────
        seen: set = set()
        deduped = []
        for alert in raw:
            if alert not in seen:
                seen.add(alert)
                deduped.append(alert)

        # ── Sort by priority descending ───────────────────────────────────────
        deduped.sort(key=lambda a: -self._PRIORITY.get(a, 1))

        # ── Suppress individual fatigue alerts subsumed by CRITICAL ───────────
        if 'CRITICAL FATIGUE' in deduped:
            deduped = ['CRITICAL FATIGUE'] + [
                a for a in deduped
                if a != 'CRITICAL FATIGUE' and a not in self._FATIGUE_ESCALATION
            ]

        result_dict['alerts'] = deduped
        return result_dict
