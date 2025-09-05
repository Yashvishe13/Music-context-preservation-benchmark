
from typing import Dict, Any

def msaf_segments(audio_path: str, feature: str = "pcp", algo: str = "sf"):
    try:
        import msaf
        try:
            est = msaf.run.process(audio_path, boundaries_id=algo, labels_id=feature)
            bounds = est["est_times"]
            labels = est.get("est_labels", None)
        except Exception:
            bounds, labels = msaf.segment(audio_path, feature=feature, algo=algo)
        return bounds, labels
    except Exception as e:
        return None, None

def mir_segment_metrics(ref_bounds, est_bounds, ref_labels=None, est_labels=None) -> Dict[str, float]:
    try:
        import mir_eval, numpy as np
    except Exception as e:
        return {"error": f"mir_eval not available: {e}"}
    det = mir_eval.segment.detection(ref_bounds, est_bounds, window=0.5, trim=True)
    out = {"boundary_f": float(det[2]), "boundary_p": float(det[0]), "boundary_r": float(det[1])}
    if ref_labels is not None and est_labels is not None:
        pair = mir_eval.segment.pairwise(ref_bounds, ref_labels, est_bounds, est_labels)
        out.update({ "pairwise_precision": float(pair[0]), "pairwise_recall": float(pair[1]), "pairwise_f": float(pair[2]) })
        try:
            ari = mir_eval.segment.ari(ref_bounds, ref_labels, est_bounds, est_labels)
            out["ari"] = float(ari)
        except Exception:
            pass
    return out

def structural_score(audio_ref: str, audio_est: str) -> Dict[str, Any]:
    rb, rl = msaf_segments(audio_ref)
    eb, el = msaf_segments(audio_est)
    if rb is None or eb is None:
        return {"error": "MSAF not installed or failed"}
    return mir_segment_metrics(rb, eb, rl, el)
