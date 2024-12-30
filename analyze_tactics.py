import json
import os
from pathlib import Path
from loguru import logger
from collections import defaultdict
from typing import Dict, List, Set
import argparse
import subprocess
import sys

from lean_dojo import *
from lean_dojo.constants import LEAN4_PACKAGES_DIR

def configure_leandojo():
    """Configure LeanDojo settings."""
    constants.logger.remove()
    constants.logger.add(sys.stderr, level="DEBUG")
    logger.info(f"Current working directory: {os.getcwd()}")

def get_mathlib4_tactics(traced_repo: TracedRepo) -> Set[str]:
    """Get all tactics used in mathlib4 dependency."""
    mathlib4_tactics = set()
    
    # Find mathlib4 in dependencies
    mathlib4_repo = None
    for name, dep in traced_repo.dependencies.items():
        if 'mathlib4' in dep.url:
            mathlib4_repo = dep
            break
            
    if not mathlib4_repo:
        logger.warning("No mathlib4 dependency found")
        return mathlib4_tactics
        
    # Get all theorems from mathlib4
    for thm in traced_repo.get_traced_theorems():
        if thm.repo == mathlib4_repo:
            for t in thm.get_traced_tactics():
                if t.state_before != "no goals" and "·" not in t.tactic:
                    mathlib4_tactics.add(t.tactic)
                    
    return mathlib4_tactics

def analyze_repo_tactics(traced_repo: TracedRepo) -> Dict:
    """Analyze tactics that are newly introduced beyond mathlib4."""
    
    # Get mathlib4 tactics
    mathlib4_tactics = get_mathlib4_tactics(traced_repo)
    logger.info(f"Found {len(mathlib4_tactics)} unique tactics in mathlib4")
    
    # Track new tactics from current repo
    new_tactics = set()
    tactic_usage = defaultdict(int)
    repo_tactics = set()
    
    # Only look at theorems from the main repo, not dependencies
    for thm in traced_repo.get_traced_theorems():
        if thm.repo == traced_repo.repo:  # Only analyze main repo
            for t in thm.get_traced_tactics():
                if t.state_before != "no goals" and "·" not in t.tactic:
                    tactic = t.tactic
                    tactic_usage[tactic] += 1
                    repo_tactics.add(tactic)
                    if tactic not in mathlib4_tactics:
                        new_tactics.add(tactic)
    
    # Prepare analysis results
    analysis = {
        "repo_url": traced_repo.repo.url,
        "repo_commit": traced_repo.repo.commit,
        "num_mathlib4_tactics": len(mathlib4_tactics),
        "num_repo_tactics": len(repo_tactics),
        "num_new_tactics": len(new_tactics),
        "new_tactics": sorted(list(new_tactics)),
        "new_tactic_usage": {t: tactic_usage[t] for t in sorted(new_tactics)},
        "mathlib4_tactics": sorted(list(mathlib4_tactics)),
        "repo_tactics": sorted(list(repo_tactics))
    }
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description="Analyze tactics in a Lean repository")
    parser.add_argument("--url", type=str, required=True, 
                       help="URL of the Lean repository to analyze")
    parser.add_argument("--commit", type=str, required=True,
                       help="Commit hash of the repository to analyze")
    parser.add_argument("--output", type=str, default="tactic_analysis.json",
                       help="Output file path for the analysis results")
    
    args = parser.parse_args()
    
    # Create repo object
    repo = LeanGitRepo(args.url, args.commit)
    logger.info(f"Analyzing repo: {repo}")
    
    # Configure LeanDojo
    configure_leandojo()
    
    try:
        # Trace the repo
        logger.info("Tracing the repo...")
        traced_repo = trace(repo)
        logger.info("Successfully traced the repo")
        
        # Analyze tactics
        analysis = analyze_repo_tactics(traced_repo)
        
        # Save results
        with open(args.output, "w") as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Analysis saved to {args.output}")
        logger.info(f"Found {analysis['num_new_tactics']} new tactics")
        
    except Exception as e:
        logger.error(f"Failed to analyze repo: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()