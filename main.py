def run_trainer():
    import signal
    import sys
    import gymnasium as gym
    import settings
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
        # Close environment and cleanup
        if trainer.env is not None:
            trainer.env.close()
        if trainer.action_output_surface is not None:
            import pygame
            pygame.quit()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Run for specified number of trials with reporting every 100 trials
        # Metrics are automatically saved every 100 trials as backup
        trainer.train(TRIALS_PER_EXPERIMENT, report_interval=100, save_interval=100)
        
        # Save metrics at the end of training
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
        
        # Enable display and continue running indefinitely
        print("\n" + "="*60)
        print("TRAINING COMPLETE - ENABLING DISPLAY")
        print("="*60)
        print("Continuing with display enabled. Press Ctrl+C to exit.")
        print("="*60 + "\n")
        
        # Enable display in settings
        settings.DISPLAY = True
        
        # Recreate environment with display enabled
        if trainer.env is not None:
            trainer.env.close()
        trainer.env = gym.make("LunarLander-v3", render_mode="human")
        
        # Initialize pygame for action output if SHOW_ACTION_OUTPUT is enabled
        if settings.SHOW_ACTION_OUTPUT == True:
            import pygame
            if trainer.action_output_surface is None:
                pygame.init()
                window_width = 200
                window_height = 50
                trainer.action_output_surface = pygame.display.set_mode((window_width, window_height), pygame.DOUBLEBUF)
                pygame.display.set_caption("Action Output")
                trainer.action_output_background = pygame.Surface((window_width, window_height))
                trainer.action_output_font = pygame.font.Font(None, 36)
                trainer.action_output_clock = pygame.time.Clock()
        
        # Continue running trials indefinitely
        display_trial_count = 0
        while True:
            display_trial_count += 1
            trainer.trial()
            if display_trial_count % 10 == 0:
                print(f"\nDisplay mode: Completed {display_trial_count} trials (Press Ctrl+C to exit)\n")
        
    except KeyboardInterrupt:
        signal_handler(None, None)

run_trainer()