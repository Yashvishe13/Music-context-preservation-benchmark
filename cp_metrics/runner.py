
import argparse
import json
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

from .harmony_tonality import harmony_score
from .rhythm_meter import rhythm_score
from .structural_form import structural_score
from .melody_motifs import melody_score


def process_single_pair(
    ref_path: str,
    est_path: str,
    pair_id: str = None,
    use_essentia: bool = False,
    skip_structure: bool = False,
    parallel_metrics: bool = False,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """
    Process a single (ref, est) audio pair and compute all metrics.
    
    Args:
        ref_path: Path to reference audio file
        est_path: Path to edited/estimated audio file
        pair_id: Optional identifier for this pair
        use_essentia: Use Essentia for key/melody extraction
        skip_structure: Skip structural form metrics
        parallel_metrics: Compute metrics in parallel
        timeout: Optional timeout in seconds (not implemented yet)
    
    Returns:
        Dictionary containing all metric scores
    """
    result = {
        "pair_id": pair_id or f"{Path(ref_path).stem}_{Path(est_path).stem}",
        "ref_path": ref_path,
        "est_path": est_path,
        "status": "success",
        "error": None,
        "metrics": {}
    }
    
    try:
        if parallel_metrics:
            # Parallel metric computation using threads
            metrics = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}
                
                # Submit all metric computations
                futures['harmony_tonality'] = executor.submit(harmony_score, ref_path, est_path)
                futures['rhythm_meter'] = executor.submit(rhythm_score, ref_path, est_path)
                if not skip_structure:
                    futures['structural_form'] = executor.submit(structural_score, ref_path, est_path)
                futures['melodic_content'] = executor.submit(
                    melody_score, ref_path, est_path, use_essentia=use_essentia
                )
                
                # Collect results
                for metric_name, future in futures.items():
                    try:
                        metrics[metric_name] = future.result()
                    except Exception as e:
                        metrics[metric_name] = {"error": str(e)}
            
            result["metrics"] = metrics
        else:
            # Sequential metric computation (original behavior)
            result["metrics"]["harmony_tonality"] = harmony_score(ref_path, est_path)
            result["metrics"]["rhythm_meter"] = rhythm_score(ref_path, est_path)
            if not skip_structure:
                result["metrics"]["structural_form"] = structural_score(ref_path, est_path)
            result["metrics"]["melodic_content"] = melody_score(
                ref_path, est_path, use_essentia=use_essentia
            )
    
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()
    
    return result


def process_pair_wrapper(args_tuple):
    """Wrapper for multiprocessing pool (unpacks tuple of arguments)"""
    return process_single_pair(*args_tuple)


def load_batch_json(json_path: str) -> List[Dict[str, str]]:
    """
    Load batch pairs from JSON file.
    Expected format: [{"id": "pair1", "ref": "path1.wav", "est": "path2.wav"}, ...]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    pairs = []
    for item in data:
        pairs.append({
            "id": item.get("id", item.get("name", f"pair_{len(pairs)}")),
            "ref": item["ref"],
            "est": item["est"]
        })
    return pairs


def load_batch_csv(csv_path: str) -> List[Dict[str, str]]:
    """
    Load batch pairs from CSV file.
    Expected columns: id, ref, est (or ref_path, est_path)
    """
    pairs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append({
                "id": row.get("id", row.get("name", f"pair_{len(pairs)}")),
                "ref": row.get("ref", row.get("ref_path")),
                "est": row.get("est", row.get("est_path"))
            })
    return pairs


def load_batch_dirs(ref_dir: str, est_dir: str, extensions: List[str] = None) -> List[Dict[str, str]]:
    """
    Load batch pairs by matching files in two directories.
    Files are matched by name (without extension).
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    ref_dir = Path(ref_dir)
    est_dir = Path(est_dir)
    
    # Find all audio files in ref directory
    ref_files = {}
    for ext in extensions:
        for f in ref_dir.glob(f'*{ext}'):
            ref_files[f.stem] = str(f)
    
    # Match with est directory
    pairs = []
    for ext in extensions:
        for f in est_dir.glob(f'*{ext}'):
            stem = f.stem
            if stem in ref_files:
                pairs.append({
                    "id": stem,
                    "ref": ref_files[stem],
                    "est": str(f)
                })
    
    return pairs


def process_batch(
    pairs: List[Dict[str, str]],
    output_dir: str,
    n_workers: int = None,
    use_essentia: bool = False,
    skip_structure: bool = False,
    parallel_metrics: bool = False,
    save_individual: bool = True,
    save_summary: bool = True,
    batch_size: int = None,
    resume: bool = True
) -> List[Dict[str, Any]]:
    """
    Process multiple audio pairs in parallel.
    
    Args:
        pairs: List of dicts with keys 'id', 'ref', 'est'
        output_dir: Directory to save results
        n_workers: Number of parallel workers (default: all CPUs)
        use_essentia: Use Essentia for key/melody
        skip_structure: Skip structural form metrics
        parallel_metrics: Also parallelize metrics within each pair
        save_individual: Save individual JSON files per pair
        save_summary: Save summary CSV file
        batch_size: Process pairs in chunks of this size (None = all at once)
        resume: Skip pairs that already have result files (for resuming)
    
    Returns:
        List of result dictionaries
    """
    cpu_count = multiprocessing.cpu_count()
    
    if n_workers is None:
        # Use all CPUs by default to maximize throughput
        n_workers = cpu_count
    
    # Smart parallel strategy: if pairs << CPUs, enable metric-level parallelism
    if not parallel_metrics and len(pairs) < cpu_count // 4:
        parallel_metrics = True
        print(f"Auto-enabling metric-level parallelism (pairs={len(pairs)} < CPUs={cpu_count})")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out already-processed pairs if resume is enabled
    original_count = len(pairs)
    if resume:
        pairs_to_process = []
        for pair in pairs:
            result_file = output_dir / f"{pair['id']}_results.json"
            if not result_file.exists():
                pairs_to_process.append(pair)
        
        if len(pairs_to_process) < original_count:
            print(f"\n✓ Resume mode: Found {original_count - len(pairs_to_process)} existing results, processing {len(pairs_to_process)} remaining pairs\n")
        pairs = pairs_to_process
    
    if not pairs:
        print("\n✓ All pairs already processed! Loading existing results...\n")
        # Load all existing results
        results = []
        for i in range(original_count):
            result_files = list(output_dir.glob("*_results.json"))
            for rf in result_files:
                with open(rf, 'r') as f:
                    results.append(json.load(f))
        return results
    
    # Split into batches if batch_size is specified
    if batch_size and batch_size < len(pairs):
        batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
        print(f"\n📦 Batch mode: Processing {len(pairs)} pairs in {len(batches)} batches of {batch_size}\n")
    else:
        batches = [pairs]
    
    all_results = []
    all_errors = []
    
    # Process each batch
    for batch_idx, batch in enumerate(batches, 1):
        if len(batches) > 1:
            print(f"\n{'='*60}")
            print(f"BATCH {batch_idx}/{len(batches)}")
            print(f"{'='*60}")
        
        # Prepare arguments for this batch
        args_list = [
            (pair["ref"], pair["est"], pair["id"], use_essentia, skip_structure, parallel_metrics, None)
            for pair in batch
        ]
        
        results = []
        errors = []
        
        # Print configuration
        print(f"\n{'='*60}")
        print(f"PARALLEL PROCESSING CONFIG")
        print(f"{'='*60}")
        print(f"Total CPU cores: {cpu_count}")
        print(f"Worker processes: {n_workers}")
        print(f"Pairs in this batch: {len(batch)}")
        print(f"Metric-level parallelism: {'Enabled' if parallel_metrics else 'Disabled'}")
        print(f"Output directory: {output_dir.resolve()}")
        print(f"{'='*60}\n")
        
        # Process with progress bar
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_pair_wrapper, args): args[2] for args in args_list}
            
            desc = f"Batch {batch_idx}/{len(batches)}" if len(batches) > 1 else "Processing pairs"
            with tqdm(total=len(batch), desc=desc, unit="pair") as pbar:
                for future in as_completed(futures):
                    pair_id = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Save individual result immediately
                        if save_individual:
                            output_file = output_dir / f"{result['pair_id']}_results.json"
                            with open(output_file, 'w') as f:
                                json.dump(result, f, indent=2)
                        
                        if result["status"] == "error":
                            errors.append(result)
                            pbar.set_postfix({"errors": len(errors)})
                    
                    except Exception as e:
                        error_result = {
                            "pair_id": pair_id,
                            "status": "error",
                            "error": str(e),
                            "metrics": {}
                        }
                        results.append(error_result)
                        errors.append(error_result)
                        pbar.set_postfix({"errors": len(errors)})
                    
                    pbar.update(1)
        
        all_results.extend(results)
        all_errors.extend(errors)
        
        # Save incremental summary after each batch
        if len(batches) > 1:
            print(f"\n✓ Batch {batch_idx} complete: {len(results)} pairs processed ({len(errors)} errors)")
            # Update cumulative summary
            if save_summary:
                all_existing = list(output_dir.glob("*_results.json"))
                temp_results = []
                for rf in all_existing:
                    with open(rf, 'r') as f:
                        temp_results.append(json.load(f))
                summary_file = output_dir / "summary.csv"
                save_summary_csv(temp_results, summary_file)
    
    # Save error log if any errors occurred
    if errors:
        error_file = output_dir / "errors.json"
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"\n⚠️  {len(errors)} pairs failed. See {error_file}")
    
    # Save summary CSV
    if save_summary and results:
        summary_file = output_dir / "summary.csv"
        save_summary_csv(results, summary_file)
        print(f"\n{'='*60}")
        print(f"OUTPUT FILES")
        print(f"{'='*60}")
        print(f"📁 Output directory: {output_dir.resolve()}")
        print(f"📊 Summary CSV: {summary_file.resolve()}")
        if save_individual:
            print(f"📄 Individual results: {len(results)} JSON files")
        print(f"{'='*60}")
    
    return results


def save_summary_csv(results: List[Dict[str, Any]], csv_path: Path):
    """Save a summary CSV with key metrics from all results."""
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header - removed harmony_chord_overlap as requested
        header = [
            "pair_id", "status",
            "harmony_key_distance", "harmony_chroma_dtw", "harmony_chroma_mean",
            "rhythm_tempo_delta_bpm", "rhythm_beat_fmeasure",
            "structure_boundary_f", "structure_pairwise_f", "structure_ari",
            "melody_overall_accuracy", "melody_raw_pitch_accuracy", "melody_contour_dtw"
        ]
        writer.writerow(header)
        
        # Data rows
        for result in results:
            if result["status"] != "success":
                writer.writerow([result["pair_id"], "error"] + ["N/A"] * (len(header) - 2))
                continue
            
            metrics = result["metrics"]
            
            # Extract key metrics safely
            row = [result["pair_id"], "success"]
            
            # Harmony - corrected field names to match actual data structure
            harmony = metrics.get("harmony_tonality", {})
            row.append(harmony.get("key_relatedness", {}).get("distance_norm_0to1", "N/A"))
            row.append(harmony.get("chroma_similarity", {}).get("chroma_dtw_cosine", "N/A"))
            row.append(harmony.get("chroma_similarity", {}).get("mean_chroma_cosine", "N/A"))
            
            # Rhythm - corrected field names to match actual data structure
            rhythm = metrics.get("rhythm_meter", {})
            row.append(rhythm.get("delta_bpm", "N/A"))
            row.append(rhythm.get("beat_mir_eval", {}).get("F-measure", "N/A"))
            
            # Structure - corrected field names to match actual data structure
            structure = metrics.get("structural_form", {})
            row.append(structure.get("boundary_f", "N/A"))
            row.append(structure.get("pairwise_f", "N/A"))
            row.append(structure.get("ari", "N/A"))
            
            # Melody - corrected field names to match actual data structure
            melody = metrics.get("melodic_content", {})
            row.append(melody.get("mir_melody", {}).get("overall_accuracy", "N/A"))
            row.append(melody.get("mir_melody", {}).get("raw_pitch_accuracy", "N/A"))
            row.append(melody.get("contour_dtw", {}).get("pitch_dtw_cosine", "N/A"))
            
            writer.writerow(row)


def main():
    p = argparse.ArgumentParser(
        description="Music Context Preservation Metrics - Single or Batch Processing"
    )
    
    # Single file mode (backward compatible)
    p.add_argument("--ref", help="Reference audio file (single mode)")
    p.add_argument("--est", help="Edited audio file (single mode)")
    
    # Batch mode options
    p.add_argument("--batch-json", help="JSON file with list of pairs")
    p.add_argument("--batch-csv", help="CSV file with pairs (columns: id,ref,est)")
    p.add_argument("--ref-dir", help="Directory with reference audio files")
    p.add_argument("--est-dir", help="Directory with edited audio files (matched by filename)")
    
    # Output options
    p.add_argument("--output-dir", default="./results", help="Output directory for batch mode")
    p.add_argument("--output-file", help="Output file for single mode (default: stdout)")
    
    # Processing options
    p.add_argument("--n-workers", type=int, help="Number of parallel workers (default: all CPUs)")
    p.add_argument("--parallel-metrics", action="store_true", 
                   help="Parallelize metrics within each pair (experimental)")
    p.add_argument("--no-parallel", action="store_true", 
                   help="Disable batch parallelization (sequential processing)")
    p.add_argument("--batch-size", type=int, 
                   help="Process pairs in chunks of N (e.g., 20 for large datasets)")
    p.add_argument("--no-resume", action="store_true",
                   help="Reprocess all pairs (ignore existing results)") 
    p.add_argument("--max-cpu", action="store_true",
                   help="Maximize CPU usage (enables both batch and metric parallelism)")
    p.add_argument("--no-parallel", action="store_true", 
                   help="Disable batch parallelization (sequential processing)")
    
    # Metric options
    p.add_argument("--sr", type=int, default=22050, help="Sample rate (kept for compatibility)")
    p.add_argument("--use-essentia", action="store_true", 
                   help="Use Essentia for key/melody when available")
    p.add_argument("--skip-structure", action="store_true", 
                   help="Skip structure (MSAF) metrics")
    
    args = p.parse_args()
    
    # Determine mode: single or batch
    is_batch_mode = any([args.batch_json, args.batch_csv, args.ref_dir])
    
    if not is_batch_mode:
        # Single file mode (backward compatible)
        if not args.ref or not args.est:
            p.error("Single mode requires --ref and --est arguments")
        
        print("Processing single pair...", file=sys.stderr)
        result = process_single_pair(
            ref_path=args.ref,
            est_path=args.est,
            use_essentia=args.use_essentia,
            skip_structure=args.skip_structure,
            parallel_metrics=args.parallel_metrics
        )
        
        # Output
        if result["status"] == "error":
            print(f"Error: {result['error']}", file=sys.stderr)
            sys.exit(1)
        
        output_data = result["metrics"]
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Results saved to {args.output_file}", file=sys.stderr)
        else:
            print(json.dumps(output_data, indent=2))
    
    else:
        # Batch mode
        pairs = []
        
        if args.batch_json:
            print(f"Loading pairs from JSON: {args.batch_json}", file=sys.stderr)
            pairs = load_batch_json(args.batch_json)
        elif args.batch_csv:
            print(f"Loading pairs from CSV: {args.batch_csv}", file=sys.stderr)
            pairs = load_batch_csv(args.batch_csv)
        elif args.ref_dir and args.est_dir:
            print(f"Matching files from directories: {args.ref_dir} <-> {args.est_dir}", file=sys.stderr)
            pairs = load_batch_dirs(args.ref_dir, args.est_dir)
        else:
            p.error("Batch mode requires --batch-json, --batch-csv, or both --ref-dir and --est-dir")
        
        if not pairs:
            print("No pairs found to process!", file=sys.stderr)
            sys.exit(1)
        
        print(f"Found {len(pairs)} pairs to process", file=sys.stderr)
        
        # Process batch
        n_workers = 1 if args.no_parallel else args.n_workers
        
        # Max CPU mode: force both levels of parallelism
        parallel_metrics = args.parallel_metrics
        if args.max_cpu:
            parallel_metrics = True
            if n_workers is None:
                n_workers = multiprocessing.cpu_count()
            print(f"🚀 MAX CPU mode: using all {n_workers} workers + metric parallelism", file=sys.stderr)
        
        results = process_batch(
            pairs=pairs,
            output_dir=args.output_dir,
            n_workers=n_workers,
            use_essentia=args.use_essentia,
            skip_structure=args.skip_structure,
            parallel_metrics=parallel_metrics,
            batch_size=args.batch_size,
            resume=not args.no_resume
        )
        
        success_count = sum(1 for r in results if r["status"] == "success")
        print(f"\n✓ Completed: {success_count}/{len(pairs)} pairs processed successfully")
        print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
