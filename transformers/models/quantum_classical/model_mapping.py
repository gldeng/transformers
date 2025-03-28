"""
This module patches the transformers library to include our custom model.
"""

# Import the necessary modules from transformers
from transformers.models.auto.configuration_auto import MODEL_NAMES_MAPPING

# Add our model to the MODEL_NAMES_MAPPING
MODEL_NAMES_MAPPING["quantum_classical"] = "Quantum Classical" 