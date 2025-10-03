"""
Data models and exceptions for deployment operations.

This module contains all the data classes and exception types used throughout
the deployment package for configuration, health tracking, and result reporting.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict


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