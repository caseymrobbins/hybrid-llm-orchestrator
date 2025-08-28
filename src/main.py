#!/usr/bin/env python3
# src/main.py

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Handle imports based on package structure
try:
    from .orchestrator import Orchestrator
    from .config import ConfigLoader, ConfigurationError
except ImportError:
    # Fallback for direct execution
    from orchestrator import Orchestrator
    from config import ConfigLoader, ConfigurationError

# Set up logging configuration
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure application logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logging.info(f"Logging to file: {log_file}")
        except Exception as e:
            logging.warning(f"Failed to set up file logging: {e}")

class ApplicationError(Exception):
    """Custom exception for application-level errors."""
    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code

class HybridLLMApp:
    """Main application class for the Hybrid LLM Orchestrator."""
    
    def __init__(self):
        self.orchestrator: Optional[Orchestrator] = None
        self.shutdown_requested = False
        self.logger = logging.getLogger(__name__)
        
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            self.logger.info(f"Received {signal_name}, initiating graceful shutdown...")
            self.shutdown_requested = True
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except (ValueError, OSError) as e:
            # Signal handling might not be available in all environments
            self.logger.warning(f"Could not set up signal handlers: {e}")
    
    def parse_arguments(self) -> argparse.Namespace:
        """Parse and validate command line arguments."""
        parser = argparse.ArgumentParser(
            description="Hybrid LLM Orchestrator - A production-grade AI workflow system",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s --query "What are the ethical implications of AI?"
  %(prog)s --query "Analyze this data" --config-path /path/to/configs --verbose
  %(prog)s --health-check --config-path ./configs
            """
        )
        
        # Main operation arguments
        parser.add_argument(
            "--query",
            type=str,
            help="The user query to process through the workflow."
        )
        
        parser.add_argument(
            "--config-path",
            type=str,
            default="./configs",
            help="Path to the configuration directory (default: ./configs)"
        )
        
        # Operation modes
        parser.add_argument(
            "--health-check",
            action="store_true",
            help="Perform a health check and exit"
        )
        
        parser.add_argument(
            "--validate-config",
            action="store_true",
            help="Validate configuration files and exit"
        )
        
        parser.add_argument(
            "--config-summary",
            action="store_true",
            help="Show configuration summary and exit"
        )
        
        # Logging options
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
            help="Set the logging level (default: INFO)"
        )
        
        parser.add_argument(
            "--log-file",
            type=str,
            help="Path to log file (default: console only)"
        )
        
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose output (equivalent to --log-level DEBUG)"
        )
        
        # Performance options
        parser.add_argument(
            "--timeout",
            type=int,
            default=300,
            help="Global timeout for workflow execution in seconds (default: 300)"
        )
        
        args = parser.parse_args()
        
        # Validate arguments
        self._validate_arguments(args)
        
        return args
    
    def _validate_arguments(self, args: argparse.Namespace) -> None:
        """Validate parsed arguments."""
        # Check if at least one operation is specified
        if not any([args.query, args.health_check, args.validate_config, args.config_summary]):
            raise ApplicationError("At least one operation must be specified (--query, --health-check, --validate-config, or --config-summary)", 2)
        
        # Validate query if provided
        if args.query and not args.query.strip():
            raise ApplicationError("Query cannot be empty", 2)
        
        # Validate config path
        config_path = Path(args.config_path)
        if not config_path.exists():
            raise ApplicationError(f"Configuration path does not exist: {config_path}", 2)
        
        if not config_path.is_dir():
            raise ApplicationError(f"Configuration path must be a directory: {config_path}", 2)
        
        # Validate timeout
        if args.timeout <= 0:
            raise ApplicationError("Timeout must be positive", 2)
        
        # Adjust log level for verbose mode
        if args.verbose:
            args.log_level = "DEBUG"
    
    async def validate_configuration(self, config_path: str) -> None:
        """Validate configuration files."""
        self.logger.info("Validating configuration files...")
        
        try:
            config_loader = ConfigLoader(Path(config_path))
            
            # Validate environment
            env_status = config_loader.validate_environment()
            if env_status["status"] != "valid":
                self.logger.error("Environment validation failed:")
                for issue in env_status["issues"]:
                    self.logger.error(f"  - {issue}")
                raise ApplicationError("Configuration validation failed", 3)
            
            # Show configuration summary
            summary = config_loader.get_config_summary()
            self.logger.info("Configuration validation successful:")
            self.logger.info(f"  Workflow: {summary['workflow']['name']} v{summary['workflow']['version']}")
            self.logger.info(f"  Modules: {summary['modules']['count']} loaded")
            self.logger.info(f"  Providers: {', '.join(summary['modules']['providers'])}")
            
        except ConfigurationError as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise ApplicationError(f"Invalid configuration: {e}", 3)
    
    async def show_config_summary(self, config_path: str) -> None:
        """Show detailed configuration summary."""
        try:
            config_loader = ConfigLoader(Path(config_path))
            summary = config_loader.get_config_summary()
            
            print("\n" + "="*60)
            print("CONFIGURATION SUMMARY")
            print("="*60)
            
            print(f"\nWorkflow Information:")
            print(f"  Name: {summary['workflow']['name']}")
            print(f"  Version: {summary['workflow']['version']}")
            print(f"  Steps: {summary['workflow']['steps']}")
            print(f"  Max Concurrent: {summary['workflow']['max_concurrent']}")
            
            print(f"\nModule Information:")
            print(f"  Total Modules: {summary['modules']['count']}")
            print(f"  Available Aspects: {', '.join(sorted(summary['modules']['aspects']))}")
            print(f"  LLM Providers: {', '.join(sorted(summary['modules']['providers']))}")
            
            # Environment status
            env_status = config_loader.validate_environment()
            print(f"\nEnvironment Status: {env_status['status'].upper()}")
            if env_status['required_variables']:
                print(f"  Required Variables: {', '.join(env_status['required_variables'])}")
            if env_status['missing_variables']:
                print(f"  Missing Variables: {', '.join(env_status['missing_variables'])}")
            
            print("="*60)
            
        except Exception as e:
            raise ApplicationError(f"Failed to show configuration summary: {e}", 3)
    
    async def perform_health_check(self, config_path: str) -> None:
        """Perform comprehensive system health check."""
        self.logger.info("Performing system health check...")
        
        start_time = time.time()
        
        try:
            # Initialize orchestrator for health check
            async with Orchestrator(config_path).managed_execution() as orchestrator:
                health_status = await orchestrator.health_check()
                
            check_duration = time.time() - start_time
            
            print("\n" + "="*60)
            print("SYSTEM HEALTH CHECK")
            print("="*60)
            
            print(f"\nOverall Status: {health_status['overall_status'].upper()}")
            print(f"Check Duration: {check_duration:.2f} seconds")
            
            # Database status
            db_status = health_status.get('components', {}).get('database', {})
            print(f"\nDatabase: {db_status.get('status', 'unknown').upper()}")
            if db_status.get('details'):
                print(f"  Details: {db_status['details']}")
            
            # External APIs
            api_status = health_status.get('components', {}).get('external_apis', {})
            print(f"\nExternal APIs:")
            for api, status in api_status.items():
                print(f"  {api}: {status.upper()}")
            
            # Circuit breakers
            breaker_status = health_status.get('components', {}).get('circuit_breakers', {})
            if breaker_status:
                print(f"\nCircuit Breakers:")
                for breaker, info in breaker_status.items():
                    state = info.get('state', 'unknown') if isinstance(info, dict) else str(info)
                    print(f"  {breaker}: {state}")
            
            # Clients
            client_status = health_status.get('components', {}).get('clients', {})
            if client_status:
                print(f"\nLLM Clients:")
                for client, status in client_status.items():
                    print(f"  {client}: {status.upper()}")
            
            # System features
            components = health_status.get('components', {})
            print(f"\nSystem Features:")
            print(f"  Semantic Cache: {components.get('cache', 'unknown')}")
            print(f"  Router: {components.get('router', 'unknown')}")
            
            print("="*60)
            
            if health_status['overall_status'] != 'healthy':
                raise ApplicationError("System health check failed", 4)
                
        except Exception as e:
            if isinstance(e, ApplicationError):
                raise
            raise ApplicationError(f"Health check failed: {e}", 4)
    
    async def process_query(self, query: str, config_path: str, timeout: int) -> str:
        """Process user query through the workflow."""
        self.logger.info(f"Processing query: {query[:100]}{'...' if len(query) > 100 else ''}")
        
        start_time = time.time()
        
        try:
            # Initialize orchestrator with timeout
            async with asyncio.timeout(timeout):
                async with Orchestrator(config_path).managed_execution() as orchestrator:
                    self.orchestrator = orchestrator
                    
                    # Check if shutdown was requested
                    if self.shutdown_requested:
                        raise ApplicationError("Shutdown requested before processing", 130)
                    
                    # Process the query
                    result = await orchestrator.execute_workflow(query)
                    
            processing_time = time.time() - start_time
            self.logger.info(f"Query processed successfully in {processing_time:.2f} seconds")
            
            return result
            
        except asyncio.TimeoutError:
            raise ApplicationError(f"Query processing timed out after {timeout} seconds", 5)
        except KeyboardInterrupt:
            raise ApplicationError("Processing interrupted by user", 130)
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Query processing failed after {processing_time:.2f} seconds: {e}")
            raise ApplicationError(f"Query processing failed: {e}", 6)
    
    def display_result(self, result: str) -> None:
        """Display the final result with formatting."""
        print("\n" + "="*20 + " FINAL RESPONSE " + "="*20)
        print(result)
        print("="*58)
    
    async def run(self) -> int:
        """Main application run method."""
        try:
            # Parse arguments
            args = self.parse_arguments()
            
            # Set up logging
            setup_logging(args.log_level, args.log_file)
            
            self.logger.info("Starting Hybrid LLM Orchestrator")
            self.logger.debug(f"Arguments: {vars(args)}")
            
            # Handle different operation modes
            if args.validate_config:
                await self.validate_configuration(args.config_path)
                print("✅ Configuration validation successful")
                return 0
            
            elif args.config_summary:
                await self.show_config_summary(args.config_path)
                return 0
            
            elif args.health_check:
                await self.perform_health_check(args.config_path)
                print("✅ System health check passed")
                return 0
            
            elif args.query:
                result = await self.process_query(args.query, args.config_path, args.timeout)
                self.display_result(result)
                return 0
            
            else:
                # This should not happen due to argument validation
                raise ApplicationError("No operation specified", 2)
                
        except ApplicationError as e:
            self.logger.error(f"Application error: {e}")
            return e.exit_code
            
        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
            return 130
            
        except Exception as e:
            self.logger.exception(f"Unexpected error: {e}")
            return 1
        
        finally:
            if self.orchestrator:
                try:
                    await self.orchestrator.cleanup()
                except Exception as e:
                    self.logger.warning(f"Error during cleanup: {e}")
            
            self.logger.info("Hybrid LLM Orchestrator shutdown complete")

async def main():
    """Async main entry point."""
    app = HybridLLMApp()
    exit_code = await app.run()
    sys.exit(exit_code)

def sync_main():
    """Synchronous entry point for setuptools."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sync_main()