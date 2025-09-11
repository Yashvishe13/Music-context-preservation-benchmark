
import argparse, json
import numpy as np
from .harmony_tonality import harmony_score
from .rhythm_meter import rhythm_score
from .structural_form import structural_score
from .melody_motifs import melody_score
from .non_target_stems import non_target_extraneous_score    

def main():
    p = argparse.ArgumentParser(description="Two-file Music Editing Context Preservation tests")
    p.add_argument("--ref", required=True, help="Reference audio file (treated as 'truth')")
    p.add_argument("--est", required=True, help="Edited audio file (to evaluate)")
    p.add_argument("--sr", type=int, default=22050, help="(kept for compat)")
    p.add_argument("--use-essentia", action="store_true", help="Use Essentia for key/melody when available")
    p.add_argument("--skip-structure", action="store_true", help="Skip structure (MSAF) metrics")
    p.add_argument("--ref-stems", type=str, default=None, help="Optional: path to .npy reference stems (n_sources x n_samples)")
    p.add_argument("--est-stems", type=str, default=None, help="Optional: path to .npy estimated stems (n_sources x n_samples)")
    args = p.parse_args()

    out = {}
    out["harmony_tonality"] = harmony_score(args.ref, args.est)
    out["rhythm_meter"] = rhythm_score(args.ref, args.est)
    if not args.skip_structure:
        out["structural_form"] = structural_score(args.ref, args.est)
    out["melodic_content"] = melody_score(args.ref, args.est, use_essentia=args.use_essentia)
    out["non_target_score"] = non_target_extraneous_score(args.ref, args.est)

    # # Load optional stems if provided
    # ref_stems = est_stems = None
    # if args.ref_stems and args.est_stems:
    #     try:
    #         ref_stems = np.load(args.ref_stems)
    #         est_stems = np.load(args.est_stems)
    #     except Exception as e:
    #         out.setdefault("warnings", []).append(f"Failed to load stems: {e}")

    # out["non_target_stems"] = comprehensive_non_target_score(
    #     args.ref, args.est, ref_stems=ref_stems, est_stems=est_stems
    # )

    print(json.dumps(out, indent=2))
    # save to file
    with open("Evaluation/results/evaluation_results.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
