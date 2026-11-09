# levels/level_manager.py

import json
import os

class LevelManager:
    """Gestiona la carga y validación de los archivos JSON de los niveles."""
    def __init__(self, level_paths):
        self.level_paths = level_paths
        
    def load_level(self, stage_idx):
        if stage_idx < 0 or stage_idx >= len(self.level_paths):
            raise ValueError(f"Índice de etapa {stage_idx} fuera de rango.")
            
        filepath = self.level_paths[stage_idx]
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el archivo del nivel en {filepath}")
            
        with open(filepath, 'r') as f:
            config = json.load(f)
            
        return config
