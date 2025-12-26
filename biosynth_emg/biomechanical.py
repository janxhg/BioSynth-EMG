import numpy as np
from scipy import signal
from typing import Tuple, Optional


class BiomechanicalLayer:
    """
    Biomechanical layer implementing motor unit firing rate simulation
    and muscle force-length-velocity relationship using Hill's equation.
    """
    
    def __init__(self, sampling_rate: int = 2000, num_motor_units: int = 100):
        """
        Initialize biomechanical layer.
        
        Args:
            sampling_rate: Sampling rate in Hz
            num_motor_units: Number of motor units to simulate
        """
        self.sampling_rate = sampling_rate
        self.num_motor_units = num_motor_units
        self.dt = 1.0 / sampling_rate
        
        # Motor unit properties - optimized for target frequency range
        self.firing_rates = np.random.uniform(40, 100, num_motor_units)  # Hz
        self.recruitment_thresholds = np.sort(np.random.uniform(0.1, 1.0, num_motor_units))
        self.muscle_fiber_types = np.random.choice(['type_I', 'type_IIa', 'type_IIb'], 
                                                 num_motor_units, p=[0.5, 0.3, 0.2])
        
        # Hill's equation parameters
        self.v_max = 0.5  # Maximum contraction velocity (m/s)
        self.f_max = 1.0  # Maximum isometric force
        self.l_opt = 1.0  # Optimal fiber length
        self.a_hill = 0.25  # Hill's constant for force-velocity relationship
        
    def generate_motor_unit_spikes(self, activation: float, duration: float, 
                                 fatigue_factor: float = 1.0) -> np.ndarray:
        """
        Generate motor unit spike trains using Poisson process.
        
        Args:
            activation: Neural activation level (0-1)
            fatigue_factor: Fatigue reduction factor (0-1)
            
        Returns:
            Array of spike trains shape (num_motor_units, num_samples)
        """
        num_samples = int(duration * self.sampling_rate)
        spike_trains = np.zeros((self.num_motor_units, num_samples))
        
        for i in range(self.num_motor_units):
            # Recruitment based on activation level
            if activation > self.recruitment_thresholds[i]:
                # Adjust firing rate based on fiber type and fatigue
                base_rate = self.firing_rates[i]
                if self.muscle_fiber_types[i] == 'type_I':
                    rate_factor = 1.0
                elif self.muscle_fiber_types[i] == 'type_IIa':
                    rate_factor = 1.2
                else:  # type_IIb
                    rate_factor = 1.5
                
                effective_rate = base_rate * rate_factor * fatigue_factor
                
                # Generate Poisson spikes
                spike_prob = effective_rate * self.dt
                spikes = np.random.random(num_samples) < spike_prob
                spike_trains[i] = spikes.astype(float)
        
        return spike_trains
    
    def hill_force_length_velocity(self, muscle_length: float, 
                                 muscle_velocity: float) -> float:
        """
        Calculate muscle force using Hill's force-length-velocity relationship.
        
        Args:
            muscle_length: Current muscle length (normalized to optimal length)
            muscle_velocity: Muscle contraction velocity (normalized)
            
        Returns:
            Normalized muscle force (0-1)
        """
        # Force-length relationship (Gaussian-like)
        length_factor = np.exp(-((muscle_length - self.l_opt) ** 2) / (2 * 0.3 ** 2))
        
        # Force-velocity relationship (Hill's equation)
        if muscle_velocity >= 0:  # Contraction
            velocity_factor = (self.f_max - self.a_hill * muscle_velocity) / (self.f_max + muscle_velocity)
        else:  # Extension
            velocity_factor = (self.f_max + self.a_hill * abs(muscle_velocity)) / (self.f_max - abs(muscle_velocity))
        
        # Combined force
        force = length_factor * velocity_factor
        return np.clip(force, 0, 1)
    
    def simulate_muscle_dynamics(self, activation: float, duration: float,
                               movement_pattern: Optional[np.ndarray] = None,
                               fatigue_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate complete muscle dynamics.
        
        Args:
            activation: Neural activation level (0-1)
            duration: Simulation duration in seconds
            movement_pattern: Optional muscle length/velocity over time
            fatigue_factor: Fatigue reduction factor (0-1)
            
        Returns:
            Tuple of (spike_trains, muscle_force)
        """
        # Generate spike trains
        spike_trains = self.generate_motor_unit_spikes(activation, duration, fatigue_factor)
        
        # Generate muscle force trajectory
        num_samples = int(duration * self.sampling_rate)
        muscle_force = np.zeros(num_samples)
        
        if movement_pattern is None:
            # Static contraction with small variations
            muscle_length = np.ones(num_samples) + 0.1 * np.sin(2 * np.pi * 0.5 * np.arange(num_samples) / self.sampling_rate)
            muscle_velocity = np.gradient(muscle_length) * self.sampling_rate
        else:
            muscle_length = movement_pattern[0]
            muscle_velocity = movement_pattern[1]
        
        for t in range(num_samples):
            # Calculate instantaneous force
            force = self.hill_force_length_velocity(muscle_length[t], muscle_velocity[t])
            
            # Add fatigue effects
            fatigue = 1.0 - 0.1 * (t / num_samples)  # Simple fatigue model
            muscle_force[t] = force * activation * fatigue
        
        return spike_trains, muscle_force
