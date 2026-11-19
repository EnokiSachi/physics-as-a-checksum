from .bytetrack import ByteTracker, KalmanXYAH, iou_matrix  # noqa: F401


def make_tracker():
    """Tracker of the current run configuration (common.TRACKER).
    bytetrack  two-stage association, low-score detections can keep tracks alive (Zhang et al., 2022)
    sort       one association stage with confident detections only (Bewley et al., 2016, same Kalman
               filter and life-cycle rules), so the effect of ByteTrack's low-score stage can be isolated"""
    import common as C
    return ByteTracker(use_low=(C.TRACKER == "bytetrack"))
