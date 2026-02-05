from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

@dataclass
class ECGArtifactRemovalConfig:
    """Base configuration for ECG artifact removal."""
    pass

class ECGArtifactRemover(ABC):
    """Abstract base class for ECG artifact removers."""
    
    def __init__(self, config: ECGArtifactRemovalConfig):
        self.config = config

    @abstractmethod
    def clean(self, df: pd.DataFrame, fs: float) -> pd.DataFrame:
        """
        Apply ECG artifact removal to a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame provided by PerceptParser (Time index, Channel columns).
            fs (float): Sampling frequency in Hz.
            
        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        pass
