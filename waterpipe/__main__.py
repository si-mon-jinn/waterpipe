"""CLI entry point for the watermark testing pipeline."""
import argparse
import logging
import sys
from pathlib import Path

from .config import load_config, validate_config
from .generation import run_generation
from .detection import run_detection
from .metrics.runner import run_metrics
from .attacks.runner import run_attacks
from .attack_metrics.runner import run_attack_metrics
from .stats import compute_stats


def setup_logging(experiment_path: Path, verbose: bool):
    """Setup logging to file and optionally console."""
    log_path = Path(experiment_path) / "pipeline.log"
    
    handlers = [logging.FileHandler(log_path)]
    if verbose:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def cmd_validate(args):
    """Validate experiment config."""
    try:
        config = load_config(args.experiment_path)
        validate_config(config)
        print(f"Config valid: {args.experiment_path}")
        return 0
    except (FileNotFoundError, ValueError) as e:
        print(f"Config invalid: {e}", file=sys.stderr)
        return 1


def cmd_run(args):
    """Run the pipeline."""
    experiment_path = Path(args.experiment_path)
    
    # Validate unless skipped
    if not args.skip_validation:
        try:
            config = load_config(experiment_path)
            validate_config(config)
        except (FileNotFoundError, ValueError) as e:
            print(f"Config invalid: {e}", file=sys.stderr)
            return 1
    else:
        config = load_config(experiment_path)
    
    setup_logging(experiment_path, args.verbose)
    logging.info(f"Starting pipeline for {experiment_path}")
    
    stages = ["generation", "metrics", "detection", "attacks", "attack_metrics", "stats"]
    if args.stage:
        stages = [args.stage]
    
    try:
        for stage in stages:
            logging.info(f"Running stage: {stage}")
            
            if stage == "generation":
                run_generation(experiment_path, config, batched=args.batched, batch_size=args.batch_size)
            elif stage == "metrics":
                run_metrics(experiment_path, config)
            elif stage == "detection":
                run_detection(experiment_path, config)
            elif stage == "attacks":
                run_attacks(experiment_path, config, batch_size=args.batch_size or 8)
            elif stage == "attack_metrics":
                run_attack_metrics(experiment_path, config)
            elif stage == "stats":
                compute_stats(experiment_path, config)
        
        logging.info("Pipeline complete")
        return 0
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        print(f"Pipeline failed: {e}", file=sys.stderr)
        return 1


def cmd_stats(args):
    """Compute statistics."""
    experiment_path = Path(args.experiment_path)
    config = load_config(experiment_path)
    compute_stats(experiment_path, config)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Watermark testing pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate experiment config")
    p_validate.add_argument("experiment_path", type=Path, help="Path to experiment directory")
    p_validate.set_defaults(func=cmd_validate)

    # run
    p_run = subparsers.add_parser("run", help="Run the pipeline")
    p_run.add_argument("experiment_path", type=Path, help="Path to experiment directory")
    p_run.add_argument("--stage", choices=["generation", "metrics", "detection", "attacks", "attack_metrics", "stats"], help="Run specific stage")
    p_run.add_argument("--skip-validation", action="store_true", help="Skip config validation")
    p_run.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    p_run.add_argument("--batched", "-b", action="store_true", help="Batch all prompts in single requests")
    p_run.add_argument("--batch-size", type=int, default=None, help="Batch size for batched generation")
    p_run.set_defaults(func=cmd_run)

    # stats
    p_stats = subparsers.add_parser("stats", help="Compute statistics")
    p_stats.add_argument("experiment_path", type=Path, help="Path to experiment directory")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
