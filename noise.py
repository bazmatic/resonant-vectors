"""
Centralized noise calculation for the training process.

Noise decreases over the total episode (all trials combined), not within individual trials.
This encourages exploration early in training and exploitation later.
"""

from math import exp
from settings import (
    NOISE_START,
    NOISE_END,
    NOISE_DECAY_RATE,
    PANIC_MAX_NOISE,
)


def calculate_base_noise(episode_progress: float) -> float:
    """
    Calculate base noise level based on episode progress (without panic factor).
    
    Args:
        episode_progress: Progress through total episode (0.0 at start, approaches 1.0 at end)
                         This is across ALL trials, not within a single trial.
    
    Returns:
        Base noise value with exponential decay (no panic scaling)
    """
    # Calculate episode noise using exponential decay
    # Noise starts at NOISE_START and asymptotically approaches NOISE_END
    return NOISE_END + (NOISE_START - NOISE_END) * exp(-NOISE_DECAY_RATE * episode_progress)


def calculate_noise(episode_progress: float, panic_factor: float = 0.0) -> float:
    """
    Calculate noise level based on episode progress and panic factor.
    
    Args:
        episode_progress: Progress through total episode (0.0 at start, approaches 1.0 at end)
                         This is across ALL trials, not within a single trial.
        panic_factor: Panic factor multiplier (0.0 to 1.0, default 0.0)
    
    Returns:
        Final noise value with exponential decay and optional panic scaling
    """
    # Calculate base episode noise using exponential decay
    base_noise = calculate_base_noise(episode_progress)
    
    # Apply panic_factor multiplicatively to scale the current episode noise
    if panic_factor > 0.0:
        # Multiplicative scaling: panic multiplies the current episode noise
        # This maintains relative panic effect throughout the episode
        noise = base_noise * (1.0 + panic_factor * (PANIC_MAX_NOISE / NOISE_START - 1.0))
    else:
        noise = base_noise
    
    return noise
