"""
Health check management system for service monitoring.

This module provides comprehensive health monitoring capabilities including
service registration, health status tracking, and automated health checks.
"""

import logging
import time
import random
from typing import Dict, Optional
from datetime import datetime

from .models import ServiceHealth, DeploymentError

logger = logging.getLogger(__name__)


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
        return random.random() > 0.05