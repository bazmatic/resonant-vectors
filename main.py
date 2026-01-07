def run_trainer():
    import signal
    import sys
    from Trainer import Trainer
    from settings import TRIALS_PER_EXPERIMENT
    
    # Check for --no-clear flag
    clear_collection = "--no-clear" not in sys.argv
    
    trainer = Trainer("lander3", clear_collection=clear_collection)
    
    # Handle Ctrl+C gracefully - save metrics before exiting
    def signal_handler(sig, frame):
        print("\n\nInterrupted! Saving metrics before exit...")
        trainer.save_metrics("training_metrics_interrupted.json")
        stats = trainer.get_summary_stats()
        print(f"\nSaved metrics from {stats['total_trials']} trials")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Run for 1000 trials with reporting every 100 trials
        # Metrics are automatically saved every 100 trials as backup
        trainer.train(TRIALS_PER_EXPERIMENT, report_interval=100, save_interval=100)
        
        # Save metrics at the end
        trainer.save_metrics("training_metrics.json")
        
        # Print final summary
        stats = trainer.get_summary_stats()
        print("\n" + "="*60)
        print("FINAL TRAINING SUMMARY")
        print("="*60)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        print("="*60)
        print(f"\nMetrics saved to: training_metrics.json")
        
    except KeyboardInterrupt:
        signal_handler(None, None)

run_trainer()