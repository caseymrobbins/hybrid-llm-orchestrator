# src/config.py

import os
import yaml
from pathlib import Path
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv

# Load environment variables from.env file
load_dotenv()

class ModuleConfig(BaseModel):
    """Pydantic model for validating a module's configuration."""
    aspect: str
    instructions: str
    # llm_address and api_key are now optional, as the router will manage model selection.
    # They can be used to override the router for specific modules if needed.
    llm_address: Optional[str] = None
    api_key: Optional[str] = "local"
    fallback: Optional[str] = None
    
    @validator('api_key', pre=True, always=True)
    def resolve_api_key_from_env(cls, v: Optional[str]) -> Optional[str]:
        """Resolves API key if it's an environment variable placeholder like ${VAR_NAME}."""
        if v and v.startswith('${') and v.endswith('}'):
            var_name = v[2:-1]
            key = os.getenv(var_name)
            if not key:
                raise ValueError(f"API key environment variable '{var_name}' not found.")
            return key
        return v

class WorkflowConfig(BaseModel):
    """Pydantic model for validating the main workflow configuration."""
    name: str
    description: str
    execution_plan: List]

class ConfigLoader:
    """Loads and validates all configurations for the orchestrator."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.workflow_config = self._load_workflow()
        self.module_configs = self._load_modules()
        self._validate_plan_modules()

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Helper to load a single YAML file."""
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {file_path}: {e}")

    def _load_workflow(self) -> WorkflowConfig:
        """Loads and validates the main workflow.yaml."""
        workflow_path = self.config_path / "workflow.yaml"
        data = self._load_yaml(workflow_path)
        return WorkflowConfig(**data)

    def _load_modules(self) -> Dict[str, ModuleConfig]:
        """Loads and validates all module configurations from the modules/ directory."""
        modules_path = self.config_path / "modules"
        configs = {}
        if not modules_path.is_dir():
            raise FileNotFoundError(f"Modules directory not found: {modules_path}")
            
        for yaml_file in modules_path.glob("*.yaml"):
            data = self._load_yaml(yaml_file)
            module_config = ModuleConfig(**data)
            if module_config.aspect in configs:
                raise ValueError(f"Duplicate module aspect found: '{module_config.aspect}'")
            configs[module_config.aspect] = module_config
        return configs

    def _validate_plan_modules(self):
        """Ensures all modules mentioned in the execution plan have a loaded config."""
        plan_modules = {step['module'] for step in self.workflow_config.execution_plan}
        loaded_modules = set(self.module_configs.keys())
        
        missing_modules = plan_modules - loaded_modules
        if missing_modules:
            raise ValueError(f"The following modules are defined in workflow.yaml but have no corresponding config file in configs/modules/: {', '.join(missing_modules)}")
