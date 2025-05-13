#!/usr/bin/env python
"""
Ecliphra Runner

Command-line interface for running Ecliphra experiments.

Usage:
    python run.py --experiment semantic --model semantics --steps 40
    python run.py --experiment echo --model echo --steps 30 --input-frequency 5
    python run.py --experiment noise --model semantics --noise-levels 0.1 0.2 0.3 0.4 0.5
    python run.py --experiment transition --model semantics --transition-steps 15
    python run.py --experiment all --model semantics --output-dir my_experiments
    python run.py --experiment spiral_resume --model semantics --steps 40
    python run.py --experiment photosynthesis --model photo --steps 40 --field-size 32 --output-dir photo_experiment_results
    python run.py --experiment photosynthesis --model photo --steps 40 --sequence-types noise_to_seed challenge_to_seed
    python run.py --experiment all --model photo --steps 40 --output-dir all_photo_experiments
    python run.py --experiment energy --model energy --steps 40 --field-size 32 --output-dir energy_experiment_results
    python run.py --experiment all --model energy --steps 40 --output-dir all_energy_experiments
    python run.py --experiment prefrontal --model energy --prefrontal \
    --steps 50 --field-size 32 --working-memory-size 7 --max-goals 3 \
    --output-dir prefrontal_experiments/test1
    python run.py --experiment prefrontal --model energy --prefrontal --enhanced-fatigue --fatigue-recovery 0.45 --fatigue-decay 0.98 --fatigue-visualize --fatigue-debug --steps 60 --working-memory-size 7 --max-goals 3 --output-dir prefrontal_experiments/fatigue_test


"""

import argparse
import os
import sys
from datetime import datetime
from ecliphra.models import create_ecliphra_model, VERSIONS
from ecliphra.experiments import create_experiment, PhotosynthesisExperiment, EcliphraExperiment, PrefrontalExperiment
from ecliphra.utils.curvature_riding import integrate_curvature_riding, integrate_curvature_riding_with_fatigue, implement_consolidation_phase, curvature_aware_forward
from ecliphra.utils.enhanced_fatigue_controller import EnhancedFatigueController
from ecliphra.visuals.fatigue_visualization import visualize_fatigue_dynamics, create_fatigue_debug_dashboard

def main():
    """Parse arguments and run the requested experiment"""
    parser = argparse.ArgumentParser(description='Run Ecliphra experiments')

    # Experiment selection
    parser.add_argument('--experiment', type=str, required=True,
                      choices=['semantic', 'echo', 'noise', 'transition', 'spiral_resume', 
                               'photosynthesis', 'energy','prefrontal', 'all'],
                      help='Type of experiment to run')

    # Model selection
    parser.add_argument('--model', type=str, default='semantics',
                      choices=['base', 'echo', 'semantics', 'photo', 'energy', 'legacy'],
                      help='Type of Ecliphra model to use')

    # Common parameters
    parser.add_argument('--steps', type=int, default=40,
                      help='Number of steps to run (default: 40)')
    parser.add_argument('--field-size', type=int, default=32,
                      help='Size of field (NxN) (default: 32)')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory (defaults to timestamped dir)')

    # Specific parameters
    parser.add_argument('--input-frequency', type=int, default=5,
                      help='How often to provide input (default: 5)')
    parser.add_argument('--memory-capacity', type=int, default=5,
                      help='Memory capacity for echo/semantic storage (default: 5)')
    parser.add_argument('--learning-steps', type=int, default=5,
                      help='Steps for initial pattern learning (default: 5)')
    parser.add_argument('--sequence-types', type=str, nargs='+',
                  default=['noise_to_seed', 'challenge_to_seed', 'mixed_sequence'],
                  help='Sequence types to test in photosynthesis experiment')

    # Energy experiment parameters
    parser.add_argument('--energy-analysis', action='store_true',
                      help='Perform detailed energy analysis in energy experiment')

    # Prefrontal module parameters
    parser.add_argument('--prefrontal', action='store_true',
                    help='Add prefrontal cortex capabilities to the model')
    parser.add_argument('--working-memory-size', type=int, default=5,
                    help='Size of working memory for prefrontal module (default: 5)')
    parser.add_argument('--max-goals', type=int, default=3,
                    help='Maximum number of concurrent goals (default: 3)')
    parser.add_argument('--goal-types', type=str, nargs='+',
                    default=['stability', 'energy', 'novelty'],
                    help='Types of goals to test in prefrontal experiment')
    parser.add_argument('--contradiction-test', action='store_true',
                  help='Run a test with contradictory goals to evaluate conflict resolution')
    parser.add_argument('--pattern-interference', action='store_true',
                    help='Run pattern interference test instead of standard prefrontal run')

     # Fatigue controller arguments 
    fatigue_group = parser.add_argument_group('Fatigue Controller Options')
    
    fatigue_group.add_argument('--enhanced-fatigue', action='store_true',
                          help='Use the enhanced fatigue controller')
    
    fatigue_group.add_argument('--fatigue-recovery', type=float, default=0.4,
                          help='Recovery factor for goal completion (0-1)')
    
    fatigue_group.add_argument('--fatigue-decay', type=float, default=0.98,
                          help='Natural decay rate of fatigue (0-1)')
    
    fatigue_group.add_argument('--fatigue-visualize', action='store_true',
                          help='Create visualizations of fatigue dynamics')
    
    fatigue_group.add_argument('--fatigue-debug', action='store_true',
                          help='Create debug dashboard for fatigue analysis')


    # Noise experiment parameters
    parser.add_argument('--noise-levels', type=float, nargs='+',
                      default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                      help='Noise levels to test (default: 0.1-0.8)')

    # Transition experiment parameters
    parser.add_argument('--transition-steps', type=int, default=10,
                      help='Steps for transition between patterns (default: 10)')
    parser.add_argument('--start-transition', type=int, default=10,
                      help='Step to begin transition (default: 10)')

    args = parser.parse_args()

    # Create a timestamp for output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.experiment == 'all':
            args.output_dir = f"ecliphra_results/all_{timestamp}"
        else:
            args.output_dir = f"ecliphra_results/{args.experiment}_{timestamp}"

    # Create the directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Print version info
    model_version = VERSIONS.get(args.model, 'unknown')
    print(f"Ecliphra {args.model.capitalize()} v{model_version}")

    # Run all experiment types if requested
    if args.experiment == 'all':
        run_all_experiments(args)
    else:
        # Run the specific requested experiment
        run_single_experiment(args)

def run_single_experiment(args):
    """Run a single experiment type with the given parameters"""
    # Create the model
    import inspect
    print(inspect.getsource(create_ecliphra_model))

    print(f"Creating {args.model} model...")
    model = create_ecliphra_model(
        model_type=args.model,
        field_dim=(args.field_size, args.field_size),
        memory_capacity=args.memory_capacity,
        prefrontal=args.prefrontal,  # Pass the prefrontal flag
        working_memory_size=args.working_memory_size,
        max_goals=args.max_goals,
        enhanced_fatigue=args.enhanced_fatigue,
        fatigue_recovery=args.fatigue_recovery,
        fatigue_decay=args.fatigue_decay

    )
     # Debug curvature riding integration
    if args.prefrontal and args.model == 'energy' and model is not None:
        print("\n=== Curvature Riding Integration Debug ===")
        print("Integrating curvature riding capabilities...")
        
        # Check required methods/attributes on model
        print(f"Model type: {type(model)}")
        print(f"Has field: {hasattr(model, 'field')}")
        print(f"Has prefrontal: {hasattr(model, 'prefrontal')}")
        print(f"Has compute_laplacian: {hasattr(model, 'compute_laplacian')}")
        
        original_model = model
        
        try:

            try:
                from ecliphra.curvature_riding import integrate_curvature_riding, integrate_curvature_riding_with_fatigue
                print("Successfully imported integrate_curvature_riding")
            except ImportError as ie:
                print(f"Failed to import curvature_riding: {ie}")
                raise
                
            # Try the integration
            model_with_curvature = integrate_curvature_riding_with_fatigue(model)
            
            # Check the result
            if model_with_curvature is not None:
                print("integrate_curvature_riding returned a model")
                model = model_with_curvature
            else:
                print("integrate_curvature_riding returned None")
                # Implement our own simplified integration
                print("Attempting simplified curvature integration...")
                model = implement_simplified_curvature_riding(model)
        except Exception as e:
            print(f"Error during curvature integration: {e}")
            import traceback
            traceback.print_exc()
            model = original_model
            
        print(f"Final model type: {type(model)}")
        print("=== End Curvature Riding Debug ===\n")
    
    # Verify 
    if model is None:
        print("Model creation failed. Cannot proceed with experiment.")
        return {"error": "Model creation failed", "status": "failed"}
    


    print(f"Setting up {args.experiment} experiment...")
    experiment = create_experiment(
        experiment_type=args.experiment,
        model=model,
        output_dir=args.output_dir,
        field_size=args.field_size
    )

    print(f"Running experiment with {args.steps} steps...")

    
    if args.experiment == 'semantic':
        results = experiment.run(
            steps=args.steps,
            input_frequency=args.input_frequency,
            learning_steps=args.learning_steps
        )
    elif args.experiment == 'echo':
        results = experiment.run(
            steps=args.steps,
            input_frequency=args.input_frequency,
            memory_capacity=args.memory_capacity
        )
    elif args.experiment == 'noise':
        results = experiment.run(
            steps=args.steps,
            noise_levels=args.noise_levels
        )
    elif args.experiment == 'transition':
        results = experiment.run(
            steps=args.steps,
            transition_steps=args.transition_steps,
            start_transition=args.start_transition
        )
    elif args.experiment == 'spiral_resume':
        results = experiment.run(
            steps=args.steps
        )
    elif args.experiment == 'photosynthesis':
        results = experiment.run(
            steps=args.steps,
            noise_levels=args.noise_levels,
            sequence_types=args.sequence_types
        )
    elif args.experiment == 'energy':
        results = experiment.run(
            steps=args.steps,
            energy_analysis=args.energy_analysis
        )
    elif args.experiment == 'prefrontal':
        if args.pattern_interference:
            print("Running pattern interference test...")
            results = experiment.run_pattern_interference_test(
                steps=args.steps,
                visualization_steps=10  # might make this configurable later
            )
        else:
            results = experiment.run(
                steps=args.steps,
                goal_count=args.max_goals,
                goal_types=args.goal_types if hasattr(args, 'goal_types') else None,
                run_contradiction_test=args.contradiction_test
            )

    if args.prefrontal and args.enhanced_fatigue:
        # fatigue visualizations if requested
        if args.fatigue_visualize:
            if hasattr(model, 'prefrontal'):
                print("Creating fatigue visualizations...")
                output_dir = os.path.join(experiment.output_dir, "fatigue_visualizations")
                visualize_fatigue_dynamics(model.prefrontal, output_dir)
            else:
                print("WARNING: Cannot create fatigue visualizations - no prefrontal module found.")
                
        if args.fatigue_debug:
            if hasattr(model, 'prefrontal'):
                print("Creating fatigue debug dashboard...")
                output_dir = os.path.join(experiment.output_dir, "fatigue_debug")
                create_fatigue_debug_dashboard(model.prefrontal, output_dir)
            else:
                print("WARNING: Cannot create fatigue debug dashboard - no prefrontal module found.")


    print(f"Experiment complete. Results saved to {experiment.output_dir}")
    return results


if __name__ == "__main__":
    main()

def run_all_experiments(args):
    """Run all experiment types in sequence"""
   
    # Determine which experiments to run based on model type
    if args.model == 'energy':
        experiments = ['energy', 'echo', 'noise', 'transition', 'prefrontal']
    else:
        # For other models, run standard experiments
        experiments = ['semantic', 'echo', 'noise', 'transition']
        
        if args.model == 'photo':
            experiments.append('photosynthesis')

    print(f"Running all experiment types. Results will be saved to {args.output_dir}")

    for exp_type in experiments:
        os.makedirs(f"{args.output_dir}/{exp_type}", exist_ok=True)

    for exp_type in experiments:
        print(f"\n{'='*50}")
        print(f"RUNNING {exp_type.upper()} EXPERIMENT")
        print(f"{'='*50}\n")

        exp_args = argparse.Namespace(**vars(args))
        exp_args.experiment = exp_type
        exp_args.output_dir = f"{args.output_dir}/{exp_type}"

        # Adjust steps for noise, shorter
        if exp_type == 'noise':
            exp_args.steps = max(20, args.steps // 2)

        run_single_experiment(exp_args)

    print(f"\nAll experiments complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
