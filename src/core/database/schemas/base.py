"""
Base database schema models and mixins for the AutoTrading system.
"""

import uuid
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field


class BaseModel:
    """
    Base model for all database entities.
    Provides common fields like ID and basic functionality.
    """
    def __init__(self):
        self.id: Optional[int] = None
        self.uuid: str = str(uuid.uuid4())
        if self.id is None:
            # In a real implementation, this would be handled by the database
            # For testing purposes, we'll assign a temporary ID
            self.id = hash(self.uuid) % 1000000


class TimestampMixin:
    """
    Mixin for models that need created_at and updated_at timestamps.
    """
    def __init__(self):
        self.created_at: datetime = datetime.utcnow()
        self.updated_at: datetime = datetime.utcnow()

    def update_timestamp(self):
        """Update the updated_at timestamp"""
        self.updated_at = datetime.utcnow()


class AuditMixin:
    """
    Mixin for models that need audit trail functionality.
    """
    def __init__(self):
        self.created_by: Optional[str] = None
        self.updated_by: Optional[str] = None
        self.version: int = 1

    def increment_version(self):
        """Increment version for optimistic locking"""
        self.version += 1


@dataclass
class SoftDeleteMixin:
    """
    Mixin for models that support soft deletion.
    """
    deleted_at: Optional[datetime] = None
    is_deleted: bool = field(default=False)

    def soft_delete(self):
        """Mark the record as deleted"""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()

    def restore(self):
        """Restore a soft-deleted record"""
        self.is_deleted = False
        self.deleted_at = None