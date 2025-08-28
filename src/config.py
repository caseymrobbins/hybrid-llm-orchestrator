# src/config.py

import os
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator, root_validator

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

class ExecutionStep(BaseModel):
    """Model for individual execution steps in the workflow."""
    module: str = Field(..., description="Name of the module to execute")
    dependencies: List[str] = Field(default_factory=list, description="List of required dependencies")
    parallel: bool = Field(default=False, description="Whether this step can run in parallel")
    timeout: Optional[int] = Field(default=None, description="Timeout in seconds for this step")
    
    @validator('module')
    def validate_module_name(cls, v):
        """Ensure module name is not empty and follows naming conventions."""
        if not v or not v.strip():
            raise ValueError("Module name cannot be empty")
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Module name must contain only alphanumeric characters, hyphens, and underscores")
        return v.strip()

class ModuleConfig(BaseModel):
    """Pydantic model for validating a module's configuration."""
    aspect: str = Field(..., description="The aspect or purpose of this module")
    instructions: str = Field(..., description="Instructions/prompt template for the module")
    llm_address: Optional[str] = Field(None, description="LLM provider address (e.g., 'openai:gpt-4')")
    api_key: Optional[str] = Field("local", description="API key or environment variable reference")
    fallback: Optional[str] = Field(None, description="Fallback model if primary fails")
    
    # Generation parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=1024, ge=1, le=8192, description="Maximum tokens to generate")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    
    # Module-specific settings
    timeout: int = Field(default=30, ge=1, le=300, description="Request timeout in seconds")
    retries: int = Field(default=3, ge=0, le=10, description="Number of retry attempts")
    cache_enabled: bool = Field(default=True, description="Whether to use semantic caching")
    
    @validator('aspect')
    def validate_aspect(cls, v):
        """Ensure aspect is not empty."""
        if not v or not v.strip():
            raise ValueError("Aspect cannot be empty")
        return v.strip()
    
    @validator('instructions')
    def validate_instructions(cls, v):
        """Ensure instructions contain valid template variables."""
        if not v or not v.strip():
            raise ValueError("Instructions cannot be empty")
        
        # Check for basic template syntax issues
        if '{' in v and '}' in v:
            # This is a basic check - more sophisticated validation could be added
            try:
                # Test if template variables are properly formatted
                import re
                template_vars = re.findall(r'\{([^}]+)\}', v)
                for var in template_vars:
                    if not var.isidentifier():
                        logger.warning(f"Template variable '{var}' may not be a valid identifier")
            except Exception as e:
                logger.warning(f"Template validation warning: {e}")
        
        return v.strip()
    
    @validator('llm_address')
    def validate_llm_address(cls, v):
        """Validate LLM address format."""
        if v is None:
            return v
            
        v = v.strip()
        if not v:
            return None
            
        # Expected format: "provider:model" or just "provider"
        if ':' in v:
            provider, model = v.split(':', 1)
            if not provider or not model:
                raise ValueError("LLM address format should be 'provider:model'")
        else:
            # Just provider name
            if not v.replace('-', '').replace('_', '').isalnum():
                raise ValueError("Provider name should contain only alphanumeric characters, hyphens, and underscores")
        
        return v
    
    @validator('api_key', pre=True, always=True)
    def resolve_api_key_from_env(cls, v: Optional[str]) -> Optional[str]:
        """Resolve API key from environment variables."""
        if v and isinstance(v, str):
            v = v.strip()
            if v.startswith('${') and v.endswith('}'):
                var_name = v[2:-1]
                resolved_key = os.getenv(var_name)
                if not resolved_key:
                    raise ValueError(f"Environment variable '{var_name}' not found or empty")
                return resolved_key
            elif v.startswith('$'):
                # Handle $VAR_NAME format
                var_name = v[1:]
                resolved_key = os.getenv(var_name)
                if not resolved_key:
                    raise ValueError(f"Environment variable '{var_name}' not found or empty")
                return resolved_key
        return v
    
    def get_generation_params(self) -> Dict[str, Any]:
        """Get parameters for text generation."""
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback."""
        return getattr(self, key, default)

class WorkflowConfig(BaseModel):
    """Pydantic model for validating the main workflow configuration."""
    name: str = Field(..., description="Name of the workflow")
    description: str = Field(..., description="Description of the workflow's purpose")
    version: str = Field(default="1.0.0", description="Workflow version")
    execution_plan: List[ExecutionStep] = Field(..., description="List of execution steps")
    
    # Global settings
    max_concurrent_steps: int = Field(default=5, ge=1, le=20, description="Maximum concurrent parallel steps")
    global_timeout: int = Field(default=300, ge=30, le=3600, description="Global workflow timeout in seconds")
    
    @validator('name')
    def validate_name(cls, v):
        """Validate workflow name."""
        if not v or not v.strip():
            raise ValueError("Workflow name cannot be empty")
        return v.strip()
    
    @validator('execution_plan')
    def validate_execution_plan(cls, v):
        """Validate execution plan consistency."""
        if not v:
            raise ValueError("Execution plan cannot be empty")
        
        # Check for duplicate module names
        module_names = [step.module for step in v]
        if len(module_names) != len(set(module_names)):
            duplicates = [name for name in set(module_names) if module_names.count(name) > 1]
            raise ValueError(f"Duplicate modules in execution plan: {duplicates}")
        
        # Validate dependencies exist in the plan
        all_modules = set(module_names)
        for step in v:
            for dep in step.dependencies:
                if dep not in all_modules:
                    raise ValueError(f"Module '{step.module}' depends on '{dep}' which is not in the execution plan")
        
        return v
    
    @root_validator
    def validate_dependency_cycles(cls, values):
        """Check for circular dependencies in the execution plan."""
        execution_plan = values.get('execution_plan', [])
        if not execution_plan:
            return values
        
        # Build dependency graph
        deps = {step.module: step.dependencies for step in execution_plan}
        
        # Check for cycles using DFS
        def has_cycle(node, visiting, visited):
            if node in visiting:
                return True
            if node in visited:
                return False
            
            visiting.add(node)
            for neighbor in deps.get(node, []):
                if has_cycle(neighbor, visiting, visited):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False
        
        visited = set()
        for module in deps:
            if module not in visited:
                if has_cycle(module, set(), visited):
                    raise ValueError(f"Circular dependency detected involving module '{module}'")
        
        return values

class ConfigLoader:
    """Loads and validates all configurations for the orchestrator."""
    
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.workflow_config: Optional[WorkflowConfig] = None
        self.module_configs: Dict[str, ModuleConfig] = {}
        
        # Load and validate all configurations
        self._load_all_configs()
    
    def _load_all_configs(self) -> None:
        """Load and validate all configuration files."""
        try:
            logger.info(f"Loading configurations from: {self.config_path}")
            
            # Load workflow configuration
            self.workflow_config = self._load_workflow()
            logger.info(f"Loaded workflow: {self.workflow_config.name}")
            
            # Load module configurations
            self.module_configs = self._load_modules()
            logger.info(f"Loaded {len(self.module_configs)} module configurations")
            
            # Cross-validate
            self._validate_configuration_consistency()
            
            logger.info("Configuration loading completed successfully")
            
        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            raise ConfigurationError(f"Failed to load configurations: {e}")
    
    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse a YAML file safely."""
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
                
            if content is None:
                raise ValueError(f"YAML file is empty or invalid: {file_path}")
                
            return content
            
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {file_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration file {file_path}: {e}")
    
    def _load_workflow(self) -> WorkflowConfig:
        """Load and validate the main workflow configuration."""
        workflow_path = self.config_path / "workflow.yaml"
        
        try:
            data = self._load_yaml(workflow_path)
            return WorkflowConfig(**data)
        except Exception as e:
            raise ConfigurationError(f"Failed to load workflow configuration: {e}")
    
    def _load_modules(self) -> Dict[str, ModuleConfig]:
        """Load and validate all module configurations."""
        modules_path = self.config_path / "modules"
        
        if not modules_path.exists():
            raise FileNotFoundError(f"Modules directory not found: {modules_path}")
        
        if not modules_path.is_dir():
            raise NotADirectoryError(f"Modules path is not a directory: {modules_path}")
        
        configs = {}
        yaml_files = list(modules_path.glob("*.yaml")) + list(modules_path.glob("*.yml"))
        
        if not yaml_files:
            logger.warning(f"No YAML files found in modules directory: {modules_path}")
            return configs
        
        for yaml_file in yaml_files:
            try:
                logger.debug(f"Loading module config: {yaml_file.name}")
                data = self._load_yaml(yaml_file)
                module_config = ModuleConfig(**data)
                
                # Use aspect as the key
                aspect = module_config.aspect
                if aspect in configs:
                    raise ValueError(f"Duplicate module aspect '{aspect}' found in file: {yaml_file.name}")
                
                configs[aspect] = module_config
                logger.debug(f"Successfully loaded module: {aspect}")
                
            except Exception as e:
                raise ConfigurationError(f"Failed to load module config from {yaml_file.name}: {e}")
        
        return configs
    
    def _validate_configuration_consistency(self) -> None:
        """Validate consistency between workflow and module configurations."""
        if not self.workflow_config:
            raise ConfigurationError("Workflow configuration not loaded")
        
        # Get all modules referenced in execution plan
        plan_modules = {step.module for step in self.workflow_config.execution_plan}
        loaded_modules = set(self.module_configs.keys())
        
        # Check for missing module configurations
        missing_modules = plan_modules - loaded_modules
        if missing_modules:
            raise ConfigurationError(
                f"Modules referenced in workflow but missing configuration files: {', '.join(sorted(missing_modules))}\n"
                f"Expected location: {self.config_path / 'modules'}/*.yaml"
            )
        
        # Check for unused module configurations (warning only)
        unused_modules = loaded_modules - plan_modules
        if unused_modules:
            logger.warning(f"Module configurations loaded but not used in workflow: {', '.join(sorted(unused_modules))}")
        
        # Validate that all modules have required dependencies available
        for step in self.workflow_config.execution_plan:
            for dep in step.dependencies:
                if dep not in loaded_modules:
                    raise ConfigurationError(f"Module '{step.module}' depends on '{dep}' which has no configuration")
    
    def get_module_config(self, module_name: str) -> Optional[ModuleConfig]:
        """Get configuration for a specific module."""
        return self.module_configs.get(module_name)
    
    def get_execution_order(self) -> List[List[str]]:
        """Get modules grouped by execution level (for parallel execution)."""
        if not self.workflow_config:
            return []
        
        # Topological sort with parallel execution support
        execution_levels = []
        remaining_modules = {step.module: step.dependencies for step in self.workflow_config.execution_plan}
        completed = set()
        
        while remaining_modules:
            # Find modules with no remaining dependencies
            ready_modules = []
            for module, deps in remaining_modules.items():
                if all(dep in completed for dep in deps):
                    ready_modules.append(module)
            
            if not ready_modules:
                # This should not happen due to cycle detection, but just in case
                raise ConfigurationError("Unable to resolve execution order - possible circular dependency")
            
            # Add ready modules to current execution level
            execution_levels.append(ready_modules)
            
            # Mark these modules as completed
            completed.update(ready_modules)
            
            # Remove completed modules from remaining
            for module in ready_modules:
                del remaining_modules[module]
        
        return execution_levels
    
    def reload(self) -> None:
        """Reload all configurations from disk."""
        logger.info("Reloading configurations...")
        self._load_all_configs()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of loaded configurations."""
        if not self.workflow_config:
            return {"status": "not_loaded"}
        
        return {
            "workflow": {
                "name": self.workflow_config.name,
                "version": self.workflow_config.version,
                "steps": len(self.workflow_config.execution_plan),
                "max_concurrent": self.workflow_config.max_concurrent_steps
            },
            "modules": {
                "count": len(self.module_configs),
                "aspects": list(self.module_configs.keys()),
                "providers": list(set(
                    config.llm_address.split(':')[0] if config.llm_address and ':' in config.llm_address else 'local'
                    for config in self.module_configs.values()
                ))
            }
        }
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate that all required environment variables are available."""
        issues = []
        required_vars = set()
        
        # Collect all environment variables referenced in configurations
        for config in self.module_configs.values():
            if config.api_key and isinstance(config.api_key, str):
                if config.api_key.startswith('${') and config.api_key.endswith('}'):
                    var_name = config.api_key[2:-1]
                    required_vars.add(var_name)
                elif config.api_key.startswith('$'):
                    var_name = config.api_key[1:]
                    required_vars.add(var_name)
        
        # Check if variables are available
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            issues.append(f"Missing environment variables: {', '.join(missing_vars)}")
        
        return {
            "status": "valid" if not issues else "invalid",
            "issues": issues,
            "required_variables": list(required_vars),
            "missing_variables": missing_vars
        }