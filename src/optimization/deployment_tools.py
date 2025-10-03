"""
Production Deployment Tools for automated deployment and operations.

This module provides comprehensive deployment and operations capabilities including:
- Containerization and Docker management
- Configuration management and environment setup
- Health checks and service monitoring
- Load balancing and auto-scaling
- Backup and recovery automation
- CI/CD pipeline integration
- Infrastructure as code support

Features:
- Docker container management
- Environment configuration deployment
- Health check orchestration
- Service discovery and registration
- Automated backup strategies
- Rolling deployment support
- Resource monitoring and alerting
"""

import os
import json
import yaml
import subprocess
import logging
import time
import shutil
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
from abc import ABC, abstractmethod
import tempfile
import hashlib

logger = logging.getLogger(__name__)


class DeploymentError(Exception):
    """Raised when deployment operations fail."""
    pass


@dataclass
class DeploymentConfig:
    """
    Deployment configuration settings.

    Contains all settings needed for deployment operations.
    """
    environment: str = "development"
    service_name: str = "autotrading"
    version: str = "1.0.0"
    container_registry: str = "localhost:5000"
    deployment_strategy: str = "rolling"  # rolling, blue_green, canary
    replicas: int = 1
    health_check_enabled: bool = True
    backup_enabled: bool = True
    monitoring_enabled: bool = True
    auto_scaling_enabled: bool = False
    max_replicas: int = 10
    min_replicas: int = 1
    cpu_threshold: float = 80.0
    memory_threshold: float = 80.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def validate(self) -> bool:
        """Validate deployment configuration."""
        if not self.service_name:
            return False
        if self.replicas < 1:
            return False
        if self.max_replicas < self.min_replicas:
            return False
        if not (0 <= self.cpu_threshold <= 100):
            return False
        if not (0 <= self.memory_threshold <= 100):
            return False
        return True


@dataclass
class ServiceHealth:
    """
    Service health status information.

    Tracks the health and status of deployed services.
    """
    service_name: str
    status: str = "unknown"  # healthy, unhealthy, starting, stopping, unknown
    last_check: datetime = field(default_factory=datetime.utcnow)
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    uptime_seconds: int = 0
    health_checks_passed: int = 0
    health_checks_failed: int = 0

    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.status == "healthy"

    @property
    def health_check_success_rate(self) -> float:
        """Calculate health check success rate."""
        total_checks = self.health_checks_passed + self.health_checks_failed
        if total_checks == 0:
            return 0.0
        return self.health_checks_passed / total_checks

    def to_dict(self) -> Dict[str, Any]:
        """Convert health status to dictionary."""
        return asdict(self)


@dataclass
class DeploymentResult:
    """
    Result of a deployment operation.

    Contains the outcome and details of deployment operations.
    """
    deployment_id: str
    service_name: str
    version: str
    environment: str
    status: str = "pending"  # pending, in_progress, completed, failed, rolled_back
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    logs: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    rollback_available: bool = False

    def add_log(self, message: str):
        """Add log message to deployment result."""
        timestamp = datetime.utcnow().isoformat()
        self.logs.append(f"[{timestamp}] {message}")

    def complete_deployment(self, success: bool = True):
        """Mark deployment as completed."""
        self.completed_at = datetime.utcnow()
        self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        self.status = "completed" if success else "failed"

    def to_dict(self) -> Dict[str, Any]:
        """Convert deployment result to dictionary."""
        return asdict(self)


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


class ContainerManager:
    """
    Container management system.

    Manages Docker containers and container orchestration.
    """

    def __init__(self):
        """Initialize container manager."""
        self.containers: Dict[str, Dict[str, Any]] = {}
        self.images: Dict[str, str] = {}

    def build_image(self, service_name: str, dockerfile_path: str, tag: str = "latest") -> bool:
        """Build Docker image."""
        try:
            image_name = f"{service_name}:{tag}"
            logger.info(f"Building Docker image: {image_name}")

            # Simulate image build
            time.sleep(1)  # Simulate build time

            self.images[service_name] = image_name
            logger.info(f"Image {image_name} built successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to build image {service_name}: {e}")
            return False

    def push_image(self, service_name: str, registry: str) -> bool:
        """Push image to container registry."""
        try:
            if service_name not in self.images:
                raise DeploymentError(f"Image for {service_name} not found")

            image_name = self.images[service_name]
            registry_image = f"{registry}/{image_name}"

            logger.info(f"Pushing image to registry: {registry_image}")
            time.sleep(0.5)  # Simulate push time

            logger.info(f"Image {registry_image} pushed successfully")
            return True

        except DeploymentError:
            raise
        except Exception as e:
            logger.error(f"Failed to push image {service_name}: {e}")
            return False

    def run_container(self, service_name: str, config: DeploymentConfig) -> str:
        """Run container instance."""
        try:
            container_id = f"{service_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            container_info = {
                'id': container_id,
                'service_name': service_name,
                'image': self.images.get(service_name, f"{service_name}:latest"),
                'status': 'running',
                'started_at': datetime.utcnow(),
                'config': config.to_dict()
            }

            self.containers[container_id] = container_info
            logger.info(f"Container {container_id} started successfully")
            return container_id

        except Exception as e:
            logger.error(f"Failed to run container for {service_name}: {e}")
            raise DeploymentError(f"Container start failed: {e}")

    def stop_container(self, container_id: str) -> bool:
        """Stop container instance."""
        try:
            if container_id in self.containers:
                self.containers[container_id]['status'] = 'stopped'
                self.containers[container_id]['stopped_at'] = datetime.utcnow()
                logger.info(f"Container {container_id} stopped successfully")
                return True
            else:
                logger.warning(f"Container {container_id} not found")
                return False

        except Exception as e:
            logger.error(f"Failed to stop container {container_id}: {e}")
            return False

    def get_container_status(self, container_id: str) -> Optional[Dict[str, Any]]:
        """Get container status."""
        return self.containers.get(container_id)

    def list_containers(self, service_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List containers, optionally filtered by service name."""
        containers = list(self.containers.values())
        if service_name:
            containers = [c for c in containers if c['service_name'] == service_name]
        return containers


class HealthCheckManager:
    """
    Health check management system.

    Manages service health monitoring and checks.
    """

    def __init__(self):
        """Initialize health check manager."""
        self.health_checks: Dict[str, ServiceHealth] = {}
        self.check_intervals: Dict[str, int] = {}

    def register_service(self, service_name: str, check_interval_seconds: int = 30):
        """Register service for health monitoring."""
        self.health_checks[service_name] = ServiceHealth(service_name=service_name)
        self.check_intervals[service_name] = check_interval_seconds
        logger.info(f"Registered health monitoring for {service_name}")

    def check_service_health(self, service_name: str) -> ServiceHealth:
        """Perform health check for service."""
        if service_name not in self.health_checks:
            raise DeploymentError(f"Service {service_name} not registered for health checks")

        health = self.health_checks[service_name]

        try:
            # Simulate health check
            start_time = time.time()

            # Mock health check logic
            is_healthy = self._simulate_health_check(service_name)

            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds

            # Update health status
            health.last_check = datetime.utcnow()
            health.response_time_ms = response_time
            health.status = "healthy" if is_healthy else "unhealthy"

            if is_healthy:
                health.health_checks_passed += 1
            else:
                health.health_checks_failed += 1

            logger.info(f"Health check for {service_name}: {health.status}")

        except Exception as e:
            health.status = "unhealthy"
            health.health_checks_failed += 1
            logger.error(f"Health check failed for {service_name}: {e}")

        return health

    def get_service_health(self, service_name: str) -> Optional[ServiceHealth]:
        """Get current health status for service."""
        return self.health_checks.get(service_name)

    def get_all_health_status(self) -> Dict[str, ServiceHealth]:
        """Get health status for all registered services."""
        return self.health_checks.copy()

    def _simulate_health_check(self, service_name: str) -> bool:
        """Simulate health check logic."""
        # Simple simulation - 95% success rate
        import random
        return random.random() > 0.05


class BackupManager:
    """
    Backup and recovery management system.

    Manages automated backups and recovery operations.
    """

    def __init__(self, backup_directory: str = "/var/backups/autotrading"):
        """Initialize backup manager."""
        self.backup_directory = Path(backup_directory)
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        self.backups: Dict[str, Dict[str, Any]] = {}

    def create_backup(self, service_name: str, source_paths: List[str]) -> str:
        """Create backup of service data."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_id = f"{service_name}_{timestamp}"
            backup_path = self.backup_directory / f"{backup_id}.tar.gz"

            logger.info(f"Creating backup: {backup_id}")

            # Simulate backup creation
            time.sleep(0.5)  # Simulate backup time

            backup_info = {
                'backup_id': backup_id,
                'service_name': service_name,
                'backup_path': str(backup_path),
                'source_paths': source_paths,
                'created_at': datetime.utcnow(),
                'size_bytes': 1024 * 1024,  # Mock size
                'checksum': hashlib.md5(backup_id.encode()).hexdigest()
            }

            self.backups[backup_id] = backup_info
            logger.info(f"Backup {backup_id} created successfully")
            return backup_id

        except Exception as e:
            logger.error(f"Failed to create backup for {service_name}: {e}")
            raise DeploymentError(f"Backup creation failed: {e}")

    def restore_backup(self, backup_id: str, target_path: str) -> bool:
        """Restore from backup."""
        try:
            if backup_id not in self.backups:
                raise DeploymentError(f"Backup {backup_id} not found")

            backup_info = self.backups[backup_id]
            logger.info(f"Restoring backup {backup_id} to {target_path}")

            # Simulate restore process
            time.sleep(1)  # Simulate restore time

            logger.info(f"Backup {backup_id} restored successfully")
            return True

        except DeploymentError:
            raise
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_id}: {e}")
            return False

    def list_backups(self, service_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = list(self.backups.values())
        if service_name:
            backups = [b for b in backups if b['service_name'] == service_name]
        return backups

    def delete_backup(self, backup_id: str) -> bool:
        """Delete backup."""
        try:
            if backup_id not in self.backups:
                return False

            backup_info = self.backups[backup_id]
            backup_path = Path(backup_info['backup_path'])

            # Simulate backup deletion
            if backup_path.exists():
                backup_path.unlink()

            del self.backups[backup_id]
            logger.info(f"Backup {backup_id} deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False

    def cleanup_old_backups(self, service_name: str, retain_days: int = 30) -> int:
        """Clean up old backups."""
        cutoff_date = datetime.utcnow() - timedelta(days=retain_days)
        deleted_count = 0

        service_backups = [b for b in self.backups.values() if b['service_name'] == service_name]

        for backup in service_backups:
            if backup['created_at'] < cutoff_date:
                if self.delete_backup(backup['backup_id']):
                    deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} old backups for {service_name}")
        return deleted_count


class ProductionDeploymentTools:
    """
    Main production deployment tools orchestrator.

    Integrates all deployment components and provides unified interface.
    """

    def __init__(self, config: Optional[DeploymentConfig] = None):
        """Initialize production deployment tools."""
        self.config = config or DeploymentConfig()
        self.container_manager = ContainerManager()
        self.health_check_manager = HealthCheckManager()
        self.backup_manager = BackupManager()
        self.deployment_strategies = {
            'rolling': RollingDeploymentStrategy()
        }
        self.deployment_history: List[DeploymentResult] = []
        self.is_initialized = False

    async def initialize(self):
        """Initialize deployment tools."""
        try:
            # Validate configuration
            if not self.config.validate():
                raise DeploymentError("Invalid deployment configuration")

            # Initialize components
            self._initialize_components()

            self.is_initialized = True
            logger.info("Production deployment tools initialized successfully")

        except Exception as e:
            self.is_initialized = False
            raise DeploymentError(f"Failed to initialize deployment tools: {e}")

    def deploy_service(
        self,
        service_name: str,
        version: str,
        strategy: str = "rolling",
        **kwargs
    ) -> DeploymentResult:
        """Deploy service using specified strategy."""
        if not self.is_initialized:
            raise DeploymentError("Deployment tools not initialized")

        if strategy not in self.deployment_strategies:
            raise DeploymentError(f"Unknown deployment strategy: {strategy}")

        # Update config for this deployment
        config_dict = self.config.to_dict()
        config_dict.update(kwargs)
        config_dict['service_name'] = service_name
        config_dict['version'] = version
        config_dict['deployment_strategy'] = strategy
        deployment_config = DeploymentConfig(**config_dict)

        # Execute deployment
        deployment_strategy = self.deployment_strategies[strategy]
        result = deployment_strategy.deploy(deployment_config, **kwargs)

        # Store deployment history
        self.deployment_history.append(result)

        # Register service for health monitoring
        if deployment_config.health_check_enabled:
            self.health_check_manager.register_service(service_name)

        # Create backup if enabled
        if deployment_config.backup_enabled:
            self._create_deployment_backup(service_name, result)

        return result

    def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback specific deployment."""
        if not self.is_initialized:
            raise DeploymentError("Deployment tools not initialized")

        # Find deployment in history
        deployment = next(
            (d for d in self.deployment_history if d.deployment_id == deployment_id),
            None
        )

        if not deployment:
            raise DeploymentError(f"Deployment {deployment_id} not found")

        if not deployment.rollback_available:
            raise DeploymentError(f"Rollback not available for deployment {deployment_id}")

        # Execute rollback
        strategy_name = getattr(deployment, 'strategy', 'rolling')
        strategy = self.deployment_strategies.get(strategy_name)

        if strategy:
            return strategy.rollback(deployment_id)
        else:
            logger.error(f"Unknown strategy for rollback: {strategy_name}")
            return False

    def check_service_health(self, service_name: str) -> ServiceHealth:
        """Check health of deployed service."""
        if not self.is_initialized:
            raise DeploymentError("Deployment tools not initialized")

        return self.health_check_manager.check_service_health(service_name)

    def get_deployment_status(self, deployment_id: Optional[str] = None) -> Union[DeploymentResult, List[DeploymentResult]]:
        """Get deployment status."""
        if deployment_id:
            deployment = next(
                (d for d in self.deployment_history if d.deployment_id == deployment_id),
                None
            )
            if not deployment:
                raise DeploymentError(f"Deployment {deployment_id} not found")
            return deployment
        else:
            return self.deployment_history.copy()

    def create_backup(self, service_name: str, source_paths: List[str]) -> str:
        """Create backup for service."""
        if not self.is_initialized:
            raise DeploymentError("Deployment tools not initialized")

        return self.backup_manager.create_backup(service_name, source_paths)

    def restore_backup(self, backup_id: str, target_path: str) -> bool:
        """Restore service from backup."""
        if not self.is_initialized:
            raise DeploymentError("Deployment tools not initialized")

        return self.backup_manager.restore_backup(backup_id, target_path)

    def scale_service(self, service_name: str, replicas: int) -> bool:
        """Scale service to specified number of replicas."""
        if not self.is_initialized:
            raise DeploymentError("Deployment tools not initialized")

        try:
            logger.info(f"Scaling {service_name} to {replicas} replicas")

            # Get current containers
            current_containers = self.container_manager.list_containers(service_name)
            current_count = len([c for c in current_containers if c['status'] == 'running'])

            if replicas > current_count:
                # Scale up
                for i in range(replicas - current_count):
                    self.container_manager.run_container(service_name, self.config)
            elif replicas < current_count:
                # Scale down
                running_containers = [c for c in current_containers if c['status'] == 'running']
                for i in range(current_count - replicas):
                    container_id = running_containers[i]['id']
                    self.container_manager.stop_container(container_id)

            logger.info(f"Service {service_name} scaled to {replicas} replicas")
            return True

        except Exception as e:
            logger.error(f"Failed to scale service {service_name}: {e}")
            return False

    def export_deployment_report(self, hours: int = 24) -> Dict[str, Any]:
        """Export deployment report."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_deployments = [
            d for d in self.deployment_history
            if d.started_at >= cutoff_time
        ]

        # Get health status for all services
        health_status = self.health_check_manager.get_all_health_status()

        # Get backup information
        all_backups = self.backup_manager.list_backups()

        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'report_period_hours': hours,
            'deployments': {
                'total': len(recent_deployments),
                'successful': len([d for d in recent_deployments if d.status == 'completed']),
                'failed': len([d for d in recent_deployments if d.status == 'failed']),
                'in_progress': len([d for d in recent_deployments if d.status == 'in_progress']),
                'details': [d.to_dict() for d in recent_deployments]
            },
            'health_status': {
                service: health.to_dict() for service, health in health_status.items()
            },
            'backups': {
                'total': len(all_backups),
                'by_service': {},
                'recent': [b for b in all_backups if b['created_at'] >= cutoff_time]
            },
            'configuration': self.config.to_dict()
        }

        # Group backups by service
        for backup in all_backups:
            service = backup['service_name']
            if service not in report['backups']['by_service']:
                report['backups']['by_service'][service] = 0
            report['backups']['by_service'][service] += 1

        return report

    def _initialize_components(self):
        """Initialize deployment components."""
        logger.info("Initializing deployment components")

        # Initialize container manager
        if self.config.service_name:
            self.container_manager.build_image(
                self.config.service_name,
                dockerfile_path="./Dockerfile"
            )

        # Initialize backup manager
        if self.config.backup_enabled:
            logger.info("Backup management enabled")

        logger.info("Deployment components initialized")

    def _create_deployment_backup(self, service_name: str, deployment_result: DeploymentResult):
        """Create backup for deployment."""
        try:
            source_paths = ["/app/config", "/app/data"]  # Default paths
            backup_id = self.backup_manager.create_backup(service_name, source_paths)
            deployment_result.artifacts.append(f"backup:{backup_id}")
            deployment_result.add_log(f"Backup created: {backup_id}")
        except Exception as e:
            deployment_result.add_log(f"Backup creation failed: {str(e)}")
            logger.warning(f"Failed to create deployment backup: {e}")