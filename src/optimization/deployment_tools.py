"""
Production Deployment Tools - Refactored Module

This module now serves as a backward-compatible wrapper around the refactored deployment package.
All original functionality is preserved through imports from submodules.

DEPRECATION NOTICE:
This file is maintained for backward compatibility only.
New code should import directly from the deployment package:
    from .deployment import ProductionDeploymentTools, ContainerManager, etc.

Original file has been split into:
- deployment/models.py: Data models and exceptions
- deployment/strategies.py: Deployment strategies
- deployment/container_manager.py: Docker container management
- deployment/health_check.py: Health monitoring system
- deployment/backup_manager.py: Backup and recovery operations
- deployment/deployment_tools.py: Main orchestrator system
"""

# Backward compatibility imports
from .deployment import (
    DeploymentError,
    DeploymentConfig,
    ServiceHealth,
    DeploymentResult,
    BaseDeploymentStrategy,
    RollingDeploymentStrategy,
    ContainerManager,
    HealthCheckManager,
    BackupManager,
    ProductionDeploymentTools
)

# Re-export all classes for backward compatibility
__all__ = [
    "DeploymentError",
    "DeploymentConfig",
    "ServiceHealth",
    "DeploymentResult",
    "BaseDeploymentStrategy",
    "RollingDeploymentStrategy",
    "ContainerManager",
    "HealthCheckManager",
    "BackupManager",
    "ProductionDeploymentTools"
]

# Deprecation warning for direct imports
import warnings
warnings.warn(
    "Importing from deployment_tools.py is deprecated. "
    "Use 'from .deployment import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)