"""
Deployment strategies for different deployment patterns.

This module provides various deployment strategies including rolling deployments,
blue-green deployments, and canary deployments.
"""

import logging
import time
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime

from .models import DeploymentConfig, DeploymentResult, DeploymentError


class BaseDeploymentStrategy(ABC):
    """
    Base class for deployment strategies.

    Provides common interface for different deployment strategies.
    """

    def __init__(self, name: str):
        """Initialize deployment strategy."""
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def deploy(self, config: DeploymentConfig, **kwargs) -> DeploymentResult:
        """Execute deployment using this strategy."""
        pass

    @abstractmethod
    def rollback(self, deployment_id: str) -> bool:
        """Rollback deployment."""
        pass

    def validate_prerequisites(self, config: DeploymentConfig) -> bool:
        """Validate deployment prerequisites."""
        return config.validate()


class RollingDeploymentStrategy(BaseDeploymentStrategy):
    """
    Rolling deployment strategy.

    Deploys new version gradually, replacing old instances one by one.
    """

    def __init__(self):
        """Initialize rolling deployment strategy."""
        super().__init__("RollingDeployment")

    def deploy(self, config: DeploymentConfig, **kwargs) -> DeploymentResult:
        """Execute rolling deployment."""
        deployment_id = self._generate_deployment_id(config)
        result = DeploymentResult(
            deployment_id=deployment_id,
            service_name=config.service_name,
            version=config.version,
            environment=config.environment,
            status="in_progress"
        )

        try:
            result.add_log(f"Starting rolling deployment for {config.service_name} v{config.version}")

            # Validate prerequisites
            if not self.validate_prerequisites(config):
                raise DeploymentError("Prerequisites validation failed")

            # Simulate rolling deployment steps
            self._deploy_containers(config, result)
            self._update_load_balancer(config, result)
            self._verify_deployment(config, result)

            result.complete_deployment(success=True)
            result.add_log("Rolling deployment completed successfully")
            result.rollback_available = True

        except Exception as e:
            result.complete_deployment(success=False)
            result.add_log(f"Deployment failed: {str(e)}")
            raise DeploymentError(f"Rolling deployment failed: {e}")

        return result

    def rollback(self, deployment_id: str) -> bool:
        """Rollback rolling deployment."""
        try:
            self.logger.info(f"Rolling back deployment {deployment_id}")
            # Simulate rollback logic
            time.sleep(1)  # Simulate rollback time
            self.logger.info(f"Deployment {deployment_id} rolled back successfully")
            return True
        except Exception as e:
            self.logger.error(f"Rollback failed for {deployment_id}: {e}")
            return False

    def _deploy_containers(self, config: DeploymentConfig, result: DeploymentResult):
        """Deploy container instances."""
        result.add_log(f"Deploying {config.replicas} container instances")

        for i in range(config.replicas):
            result.add_log(f"Deploying container {i+1}/{config.replicas}")
            time.sleep(0.1)  # Simulate deployment time

        result.add_log("All containers deployed successfully")

    def _update_load_balancer(self, config: DeploymentConfig, result: DeploymentResult):
        """Update load balancer configuration."""
        result.add_log("Updating load balancer configuration")
        time.sleep(0.1)  # Simulate update time
        result.add_log("Load balancer updated successfully")

    def _verify_deployment(self, config: DeploymentConfig, result: DeploymentResult):
        """Verify deployment health."""
        result.add_log("Verifying deployment health")

        if config.health_check_enabled:
            result.add_log("Running health checks")
            time.sleep(0.1)  # Simulate health check time
            result.add_log("Health checks passed")

        result.add_log("Deployment verification completed")

    def _generate_deployment_id(self, config: DeploymentConfig) -> str:
        """Generate unique deployment ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")  # Include microseconds
        service_hash = hashlib.md5(config.service_name.encode()).hexdigest()[:8]
        return f"{config.service_name}_{timestamp}_{service_hash}"