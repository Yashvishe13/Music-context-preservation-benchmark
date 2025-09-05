
import argparse, json
from cp_metrics.harmony_tonality import harmony_score
from cp_metrics.rhythm_meter import rhythm_score
from cp_metrics.structural_form import structural_score
from cp_metrics.melody_motifs import melody_score


def main():
    p = argparse.ArgumentParser(description="Two-file Music Editing Context Preservation tests")
    p.add_argument("--ref", required=True, help="Reference audio file (treated as 'truth')")
    p.add_argument("--est", required=True, help="Edited audio file (to evaluate)")
    p.add_argument("--sr", type=int, default=22050, help="(kept for compat)")
    p.add_argument("--use-essentia", action="store_true", help="Use Essentia for key/melody when available")
    p.add_argument("--skip-structure", action="store_true", help="Skip structure (MSAF) metrics")
    args = p.parse_args()

    out = {}
    out["harmony_tonality"] = harmony_score(args.ref, args.est)
    out["rhythm_meter"] = rhythm_score(args.ref, args.est)
    if not args.skip_structure:
        out["structural_form"] = structural_score(args.ref, args.est)
    out["melodic_content"] = melody_score(args.ref, args.est, use_essentia=args.use_essentia)

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
