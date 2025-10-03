"""
Comprehensive tests for the Production Deployment Tools.

Tests cover:
- Deployment configuration and validation
- Container management and orchestration
- Health checking and monitoring
- Backup and recovery operations
- Deployment strategies and rollbacks
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.optimization.deployment_tools import (
    DeploymentConfig,
    ServiceHealth,
    DeploymentResult,
    RollingDeploymentStrategy,
    ContainerManager,
    HealthCheckManager,
    BackupManager,
    ProductionDeploymentTools,
    DeploymentError
)


class TestDeploymentConfig:
    """Test DeploymentConfig data structure."""

    def test_should_initialize_with_defaults(self):
        """Test that DeploymentConfig initializes with default values."""
        config = DeploymentConfig()

        assert config.environment == "development"
        assert config.service_name == "autotrading"
        assert config.version == "1.0.0"
        assert config.deployment_strategy == "rolling"
        assert config.replicas == 1
        assert config.health_check_enabled is True
        assert config.backup_enabled is True

    def test_should_initialize_with_custom_values(self):
        """Test that DeploymentConfig can initialize with custom values."""
        config = DeploymentConfig(
            environment="production",
            service_name="trading-engine",
            version="2.1.0",
            replicas=3
        )

        assert config.environment == "production"
        assert config.service_name == "trading-engine"
        assert config.version == "2.1.0"
        assert config.replicas == 3

    def test_should_convert_to_dictionary(self):
        """Test that DeploymentConfig can convert to dictionary."""
        config = DeploymentConfig(
            environment="staging",
            replicas=2
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['environment'] == "staging"
        assert config_dict['replicas'] == 2

    def test_should_validate_configuration(self):
        """Test that DeploymentConfig validates correctly."""
        # Valid configuration
        valid_config = DeploymentConfig(
            service_name="test-service",
            replicas=2,
            min_replicas=1,
            max_replicas=5
        )
        assert valid_config.validate() is True

        # Invalid configuration - empty service name
        invalid_config1 = DeploymentConfig(service_name="")
        assert invalid_config1.validate() is False

        # Invalid configuration - negative replicas
        invalid_config2 = DeploymentConfig(replicas=-1)
        assert invalid_config2.validate() is False

        # Invalid configuration - max < min replicas
        invalid_config3 = DeploymentConfig(min_replicas=5, max_replicas=2)
        assert invalid_config3.validate() is False

        # Invalid configuration - invalid threshold
        invalid_config4 = DeploymentConfig(cpu_threshold=150.0)
        assert invalid_config4.validate() is False


class TestServiceHealth:
    """Test ServiceHealth data structure."""

    def test_should_initialize_with_service_name(self):
        """Test that ServiceHealth initializes with service name."""
        health = ServiceHealth(service_name="test-service")

        assert health.service_name == "test-service"
        assert health.status == "unknown"
        assert health.health_checks_passed == 0
        assert health.health_checks_failed == 0

    def test_should_check_healthy_status(self):
        """Test that ServiceHealth correctly checks healthy status."""
        health = ServiceHealth(service_name="test-service")

        # Initially not healthy
        assert health.is_healthy is False

        # Set to healthy
        health.status = "healthy"
        assert health.is_healthy is True

        # Set to unhealthy
        health.status = "unhealthy"
        assert health.is_healthy is False

    def test_should_calculate_success_rate(self):
        """Test that ServiceHealth calculates success rate correctly."""
        health = ServiceHealth(service_name="test-service")

        # No checks yet
        assert health.health_check_success_rate == 0.0

        # Some checks
        health.health_checks_passed = 8
        health.health_checks_failed = 2
        assert health.health_check_success_rate == 0.8

        # All passed
        health.health_checks_failed = 0
        assert health.health_check_success_rate == 1.0

    def test_should_convert_to_dictionary(self):
        """Test that ServiceHealth can convert to dictionary."""
        health = ServiceHealth(
            service_name="test-service",
            status="healthy",
            response_time_ms=50.0
        )

        health_dict = health.to_dict()

        assert isinstance(health_dict, dict)
        assert health_dict['service_name'] == "test-service"
        assert health_dict['status'] == "healthy"
        assert health_dict['response_time_ms'] == 50.0


class TestDeploymentResult:
    """Test DeploymentResult data structure."""

    def test_should_initialize_with_required_fields(self):
        """Test that DeploymentResult initializes with required fields."""
        result = DeploymentResult(
            deployment_id="deploy-123",
            service_name="test-service",
            version="1.0.0",
            environment="development"
        )

        assert result.deployment_id == "deploy-123"
        assert result.service_name == "test-service"
        assert result.version == "1.0.0"
        assert result.environment == "development"
        assert result.status == "pending"
        assert result.logs == []
        assert result.artifacts == []

    def test_should_add_log_messages(self):
        """Test that DeploymentResult can add log messages."""
        result = DeploymentResult(
            deployment_id="deploy-123",
            service_name="test-service",
            version="1.0.0",
            environment="development"
        )

        result.add_log("Starting deployment")
        result.add_log("Deployment completed")

        assert len(result.logs) == 2
        assert "Starting deployment" in result.logs[0]
        assert "Deployment completed" in result.logs[1]

    def test_should_complete_deployment_successfully(self):
        """Test that DeploymentResult can complete deployment successfully."""
        result = DeploymentResult(
            deployment_id="deploy-123",
            service_name="test-service",
            version="1.0.0",
            environment="development"
        )

        assert result.status == "pending"
        assert result.completed_at is None

        import time
        time.sleep(0.01)  # Ensure some duration
        result.complete_deployment(success=True)

        assert result.status == "completed"
        assert result.completed_at is not None
        assert result.duration_seconds > 0

    def test_should_complete_deployment_with_failure(self):
        """Test that DeploymentResult can complete deployment with failure."""
        result = DeploymentResult(
            deployment_id="deploy-123",
            service_name="test-service",
            version="1.0.0",
            environment="development"
        )

        result.complete_deployment(success=False)

        assert result.status == "failed"
        assert result.completed_at is not None

    def test_should_convert_to_dictionary(self):
        """Test that DeploymentResult can convert to dictionary."""
        result = DeploymentResult(
            deployment_id="deploy-123",
            service_name="test-service",
            version="1.0.0",
            environment="development"
        )
        result.add_log("Test log")

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict['deployment_id'] == "deploy-123"
        assert len(result_dict['logs']) == 1


class TestRollingDeploymentStrategy:
    """Test RollingDeploymentStrategy functionality."""

    @pytest.fixture
    def rolling_strategy(self):
        """Create rolling deployment strategy for testing."""
        return RollingDeploymentStrategy()

    @pytest.fixture
    def deployment_config(self):
        """Create deployment config for testing."""
        return DeploymentConfig(
            service_name="test-service",
            version="1.0.0",
            replicas=3
        )

    def test_should_initialize_correctly(self, rolling_strategy):
        """Test that RollingDeploymentStrategy initializes correctly."""
        assert rolling_strategy.name == "RollingDeployment"

    def test_should_validate_prerequisites(self, rolling_strategy, deployment_config):
        """Test that RollingDeploymentStrategy validates prerequisites."""
        # Valid config
        assert rolling_strategy.validate_prerequisites(deployment_config) is True

        # Invalid config
        invalid_config = DeploymentConfig(service_name="", replicas=-1)
        assert rolling_strategy.validate_prerequisites(invalid_config) is False

    def test_should_execute_deployment(self, rolling_strategy, deployment_config):
        """Test that RollingDeploymentStrategy can execute deployment."""
        result = rolling_strategy.deploy(deployment_config)

        assert isinstance(result, DeploymentResult)
        assert result.service_name == "test-service"
        assert result.version == "1.0.0"
        assert result.status == "completed"
        assert result.rollback_available is True
        assert len(result.logs) > 0

    def test_should_handle_deployment_failure(self, rolling_strategy):
        """Test that RollingDeploymentStrategy handles deployment failures."""
        # Create invalid config to trigger failure
        invalid_config = DeploymentConfig(service_name="", replicas=-1)

        with pytest.raises(DeploymentError, match="Rolling deployment failed"):
            rolling_strategy.deploy(invalid_config)

    def test_should_perform_rollback(self, rolling_strategy):
        """Test that RollingDeploymentStrategy can perform rollback."""
        result = rolling_strategy.rollback("deployment-123")
        assert result is True

    def test_should_generate_unique_deployment_id(self, rolling_strategy, deployment_config):
        """Test that RollingDeploymentStrategy generates unique deployment IDs."""
        result1 = rolling_strategy.deploy(deployment_config)
        time.sleep(0.01)  # Ensure different timestamp
        result2 = rolling_strategy.deploy(deployment_config)

        assert result1.deployment_id != result2.deployment_id
        assert "test-service" in result1.deployment_id
        assert "test-service" in result2.deployment_id


class TestContainerManager:
    """Test ContainerManager functionality."""

    @pytest.fixture
    def container_manager(self):
        """Create container manager for testing."""
        return ContainerManager()

    @pytest.fixture
    def deployment_config(self):
        """Create deployment config for testing."""
        return DeploymentConfig(service_name="test-service")

    def test_should_initialize_correctly(self, container_manager):
        """Test that ContainerManager initializes correctly."""
        assert container_manager.containers == {}
        assert container_manager.images == {}

    def test_should_build_image(self, container_manager):
        """Test that ContainerManager can build images."""
        result = container_manager.build_image("test-service", "./Dockerfile", "v1.0.0")

        assert result is True
        assert "test-service" in container_manager.images
        assert container_manager.images["test-service"] == "test-service:v1.0.0"

    def test_should_push_image(self, container_manager):
        """Test that ContainerManager can push images."""
        # Build image first
        container_manager.build_image("test-service", "./Dockerfile")

        # Push image
        result = container_manager.push_image("test-service", "localhost:5000")

        assert result is True

    def test_should_handle_push_without_image(self, container_manager):
        """Test that ContainerManager handles push without built image."""
        with pytest.raises(DeploymentError, match="Image for nonexistent-service not found"):
            container_manager.push_image("nonexistent-service", "localhost:5000")

    def test_should_run_container(self, container_manager, deployment_config):
        """Test that ContainerManager can run containers."""
        container_id = container_manager.run_container("test-service", deployment_config)

        assert isinstance(container_id, str)
        assert container_id in container_manager.containers
        assert container_manager.containers[container_id]['status'] == 'running'

    def test_should_stop_container(self, container_manager, deployment_config):
        """Test that ContainerManager can stop containers."""
        # Run container first
        container_id = container_manager.run_container("test-service", deployment_config)

        # Stop container
        result = container_manager.stop_container(container_id)

        assert result is True
        assert container_manager.containers[container_id]['status'] == 'stopped'

    def test_should_handle_stop_nonexistent_container(self, container_manager):
        """Test that ContainerManager handles stopping nonexistent containers."""
        result = container_manager.stop_container("nonexistent-container")
        assert result is False

    def test_should_get_container_status(self, container_manager, deployment_config):
        """Test that ContainerManager can get container status."""
        container_id = container_manager.run_container("test-service", deployment_config)

        status = container_manager.get_container_status(container_id)

        assert status is not None
        assert status['id'] == container_id
        assert status['service_name'] == "test-service"

    def test_should_list_containers(self, container_manager, deployment_config):
        """Test that ContainerManager can list containers."""
        # Run multiple containers
        container1 = container_manager.run_container("service1", deployment_config)
        container2 = container_manager.run_container("service2", deployment_config)

        # List all containers
        all_containers = container_manager.list_containers()
        assert len(all_containers) == 2

        # List containers for specific service
        service1_containers = container_manager.list_containers("service1")
        assert len(service1_containers) == 1
        assert service1_containers[0]['service_name'] == "service1"


class TestHealthCheckManager:
    """Test HealthCheckManager functionality."""

    @pytest.fixture
    def health_manager(self):
        """Create health check manager for testing."""
        return HealthCheckManager()

    def test_should_initialize_correctly(self, health_manager):
        """Test that HealthCheckManager initializes correctly."""
        assert health_manager.health_checks == {}
        assert health_manager.check_intervals == {}

    def test_should_register_service(self, health_manager):
        """Test that HealthCheckManager can register services."""
        health_manager.register_service("test-service", check_interval_seconds=60)

        assert "test-service" in health_manager.health_checks
        assert health_manager.check_intervals["test-service"] == 60

    def test_should_check_service_health(self, health_manager):
        """Test that HealthCheckManager can check service health."""
        # Register service first
        health_manager.register_service("test-service")

        # Check health
        health = health_manager.check_service_health("test-service")

        assert isinstance(health, ServiceHealth)
        assert health.service_name == "test-service"
        assert health.status in ["healthy", "unhealthy"]
        assert health.response_time_ms >= 0

    def test_should_handle_unregistered_service_health_check(self, health_manager):
        """Test that HealthCheckManager handles unregistered service health checks."""
        with pytest.raises(DeploymentError, match="Service unregistered-service not registered"):
            health_manager.check_service_health("unregistered-service")

    def test_should_get_service_health(self, health_manager):
        """Test that HealthCheckManager can get service health."""
        # Register and check service
        health_manager.register_service("test-service")
        health_manager.check_service_health("test-service")

        # Get health status
        health = health_manager.get_service_health("test-service")

        assert health is not None
        assert health.service_name == "test-service"

    def test_should_get_all_health_status(self, health_manager):
        """Test that HealthCheckManager can get all health status."""
        # Register multiple services
        health_manager.register_service("service1")
        health_manager.register_service("service2")

        # Check health for both
        health_manager.check_service_health("service1")
        health_manager.check_service_health("service2")

        # Get all health status
        all_health = health_manager.get_all_health_status()

        assert len(all_health) == 2
        assert "service1" in all_health
        assert "service2" in all_health


class TestBackupManager:
    """Test BackupManager functionality."""

    @pytest.fixture
    def backup_manager(self, tmp_path):
        """Create backup manager for testing."""
        return BackupManager(backup_directory=str(tmp_path / "backups"))

    def test_should_initialize_correctly(self, backup_manager):
        """Test that BackupManager initializes correctly."""
        assert backup_manager.backups == {}
        assert backup_manager.backup_directory.exists()

    def test_should_create_backup(self, backup_manager):
        """Test that BackupManager can create backups."""
        source_paths = ["/app/config", "/app/data"]
        backup_id = backup_manager.create_backup("test-service", source_paths)

        assert isinstance(backup_id, str)
        assert backup_id in backup_manager.backups
        assert backup_manager.backups[backup_id]['service_name'] == "test-service"
        assert backup_manager.backups[backup_id]['source_paths'] == source_paths

    def test_should_restore_backup(self, backup_manager):
        """Test that BackupManager can restore backups."""
        # Create backup first
        source_paths = ["/app/config"]
        backup_id = backup_manager.create_backup("test-service", source_paths)

        # Restore backup
        result = backup_manager.restore_backup(backup_id, "/restore/path")

        assert result is True

    def test_should_handle_restore_nonexistent_backup(self, backup_manager):
        """Test that BackupManager handles restoring nonexistent backups."""
        with pytest.raises(DeploymentError, match="Backup nonexistent-backup not found"):
            backup_manager.restore_backup("nonexistent-backup", "/restore/path")

    def test_should_list_backups(self, backup_manager):
        """Test that BackupManager can list backups."""
        # Create multiple backups
        backup1 = backup_manager.create_backup("service1", ["/path1"])
        backup2 = backup_manager.create_backup("service2", ["/path2"])

        # List all backups
        all_backups = backup_manager.list_backups()
        assert len(all_backups) == 2

        # List backups for specific service
        service1_backups = backup_manager.list_backups("service1")
        assert len(service1_backups) == 1
        assert service1_backups[0]['service_name'] == "service1"

    def test_should_delete_backup(self, backup_manager):
        """Test that BackupManager can delete backups."""
        # Create backup first
        backup_id = backup_manager.create_backup("test-service", ["/path"])

        # Delete backup
        result = backup_manager.delete_backup(backup_id)

        assert result is True
        assert backup_id not in backup_manager.backups

    def test_should_handle_delete_nonexistent_backup(self, backup_manager):
        """Test that BackupManager handles deleting nonexistent backups."""
        result = backup_manager.delete_backup("nonexistent-backup")
        assert result is False

    def test_should_cleanup_old_backups(self, backup_manager):
        """Test that BackupManager can cleanup old backups."""
        # Create backup and manually set old timestamp
        backup_id = backup_manager.create_backup("test-service", ["/path"])
        old_timestamp = datetime.utcnow() - timedelta(days=35)
        backup_manager.backups[backup_id]['created_at'] = old_timestamp

        # Cleanup old backups
        deleted_count = backup_manager.cleanup_old_backups("test-service", retain_days=30)

        assert deleted_count == 1
        assert backup_id not in backup_manager.backups


class TestProductionDeploymentTools:
    """Test ProductionDeploymentTools integration."""

    @pytest.fixture
    def deployment_tools(self):
        """Create production deployment tools for testing."""
        config = DeploymentConfig(service_name="test-service")
        return ProductionDeploymentTools(config)

    @pytest.mark.asyncio
    async def test_should_initialize_successfully(self, deployment_tools):
        """Test that ProductionDeploymentTools initializes successfully."""
        await deployment_tools.initialize()

        assert deployment_tools.is_initialized is True
        assert isinstance(deployment_tools.container_manager, ContainerManager)
        assert isinstance(deployment_tools.health_check_manager, HealthCheckManager)
        assert isinstance(deployment_tools.backup_manager, BackupManager)

    @pytest.mark.asyncio
    async def test_should_handle_invalid_configuration(self):
        """Test that ProductionDeploymentTools handles invalid configuration."""
        invalid_config = DeploymentConfig(service_name="", replicas=-1)
        deployment_tools = ProductionDeploymentTools(invalid_config)

        with pytest.raises(DeploymentError, match="Invalid deployment configuration"):
            await deployment_tools.initialize()

    def test_should_require_initialization(self, deployment_tools):
        """Test that ProductionDeploymentTools requires initialization."""
        with pytest.raises(DeploymentError, match="Deployment tools not initialized"):
            deployment_tools.deploy_service("test-service", "1.0.0")

    @pytest.mark.asyncio
    async def test_should_deploy_service(self, deployment_tools):
        """Test that ProductionDeploymentTools can deploy services."""
        await deployment_tools.initialize()

        result = deployment_tools.deploy_service("test-service", "1.0.0", strategy="rolling")

        assert isinstance(result, DeploymentResult)
        assert result.service_name == "test-service"
        assert result.version == "1.0.0"
        assert result.status == "completed"
        assert len(deployment_tools.deployment_history) == 1

    @pytest.mark.asyncio
    async def test_should_handle_unknown_deployment_strategy(self, deployment_tools):
        """Test that ProductionDeploymentTools handles unknown deployment strategies."""
        await deployment_tools.initialize()

        with pytest.raises(DeploymentError, match="Unknown deployment strategy: unknown"):
            deployment_tools.deploy_service("test-service", "1.0.0", strategy="unknown")

    @pytest.mark.asyncio
    async def test_should_rollback_deployment(self, deployment_tools):
        """Test that ProductionDeploymentTools can rollback deployments."""
        await deployment_tools.initialize()

        # Deploy service first
        result = deployment_tools.deploy_service("test-service", "1.0.0")
        deployment_id = result.deployment_id

        # Rollback deployment
        rollback_result = deployment_tools.rollback_deployment(deployment_id)

        assert rollback_result is True

    @pytest.mark.asyncio
    async def test_should_handle_rollback_nonexistent_deployment(self, deployment_tools):
        """Test that ProductionDeploymentTools handles rollback of nonexistent deployments."""
        await deployment_tools.initialize()

        with pytest.raises(DeploymentError, match="Deployment nonexistent-deployment not found"):
            deployment_tools.rollback_deployment("nonexistent-deployment")

    @pytest.mark.asyncio
    async def test_should_check_service_health(self, deployment_tools):
        """Test that ProductionDeploymentTools can check service health."""
        await deployment_tools.initialize()

        # Deploy service first to register it
        deployment_tools.deploy_service("test-service", "1.0.0")

        # Check service health
        health = deployment_tools.check_service_health("test-service")

        assert isinstance(health, ServiceHealth)
        assert health.service_name == "test-service"

    @pytest.mark.asyncio
    async def test_should_get_deployment_status(self, deployment_tools):
        """Test that ProductionDeploymentTools can get deployment status."""
        await deployment_tools.initialize()

        # Deploy service
        result = deployment_tools.deploy_service("test-service", "1.0.0")
        deployment_id = result.deployment_id

        # Get specific deployment status
        status = deployment_tools.get_deployment_status(deployment_id)
        assert isinstance(status, DeploymentResult)
        assert status.deployment_id == deployment_id

        # Get all deployment status
        all_status = deployment_tools.get_deployment_status()
        assert isinstance(all_status, list)
        assert len(all_status) == 1

    @pytest.mark.asyncio
    async def test_should_create_and_restore_backup(self, deployment_tools):
        """Test that ProductionDeploymentTools can create and restore backups."""
        await deployment_tools.initialize()

        # Create backup
        source_paths = ["/app/config", "/app/data"]
        backup_id = deployment_tools.create_backup("test-service", source_paths)

        assert isinstance(backup_id, str)

        # Restore backup
        result = deployment_tools.restore_backup(backup_id, "/restore/path")

        assert result is True

    @pytest.mark.asyncio
    async def test_should_scale_service(self, deployment_tools):
        """Test that ProductionDeploymentTools can scale services."""
        await deployment_tools.initialize()

        # Scale service
        result = deployment_tools.scale_service("test-service", replicas=3)

        assert result is True

    @pytest.mark.asyncio
    async def test_should_export_deployment_report(self, deployment_tools):
        """Test that ProductionDeploymentTools can export deployment reports."""
        await deployment_tools.initialize()

        # Deploy service to have some data
        deployment_tools.deploy_service("test-service", "1.0.0")

        # Export report
        report = deployment_tools.export_deployment_report(hours=24)

        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'deployments' in report
        assert 'health_status' in report
        assert 'backups' in report
        assert 'configuration' in report
        assert report['deployments']['total'] == 1

    @pytest.mark.asyncio
    async def test_should_handle_initialization_errors(self):
        """Test that ProductionDeploymentTools handles initialization errors."""
        deployment_tools = ProductionDeploymentTools()

        # Simulate error during initialization
        with patch.object(deployment_tools, '_initialize_components', side_effect=Exception("Test error")):
            with pytest.raises(DeploymentError, match="Failed to initialize deployment tools"):
                await deployment_tools.initialize()

            assert deployment_tools.is_initialized is False

    @pytest.mark.asyncio
    async def test_should_integrate_all_components(self, deployment_tools):
        """Test that ProductionDeploymentTools integrates all components correctly."""
        await deployment_tools.initialize()

        # Deploy service
        deployment_result = deployment_tools.deploy_service(
            "integration-test-service",
            "1.2.3",
            strategy="rolling",
            replicas=2
        )

        assert deployment_result.status == "completed"
        assert deployment_result.service_name == "integration-test-service"

        # Check service health
        health = deployment_tools.check_service_health("integration-test-service")
        assert health.service_name == "integration-test-service"

        # Scale service
        scale_result = deployment_tools.scale_service("integration-test-service", 4)
        assert scale_result is True

        # Create backup
        backup_id = deployment_tools.create_backup("integration-test-service", ["/app"])
        assert backup_id is not None

        # Get comprehensive status
        all_deployments = deployment_tools.get_deployment_status()
        assert len(all_deployments) == 1

        # Export comprehensive report
        report = deployment_tools.export_deployment_report()
        assert report['deployments']['total'] == 1
        assert 'integration-test-service' in report['health_status']
        assert report['backups']['total'] >= 1