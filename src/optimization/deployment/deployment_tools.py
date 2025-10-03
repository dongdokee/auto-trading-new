"""
Main production deployment tools orchestrator.

This module provides the main orchestrator for all deployment operations,
integrating container management, health checking, backup operations, and
deployment strategies into a unified interface.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

from .models import DeploymentConfig, DeploymentResult, ServiceHealth, DeploymentError
from .container_manager import ContainerManager
from .health_check import HealthCheckManager
from .backup_manager import BackupManager
from .strategies import RollingDeploymentStrategy

logger = logging.getLogger(__name__)


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