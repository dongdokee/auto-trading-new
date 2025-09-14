"""
Base repository implementation providing common CRUD operations.
Following the Repository pattern for data access abstraction.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Type
import logging

# Type variable for entity models
T = TypeVar('T')


class RepositoryError(Exception):
    """Base exception for repository operations"""
    pass


class BaseRepository(Generic[T]):
    """
    Base repository providing common CRUD operations for any entity model.

    This implementation provides:
    - Create, Read, Update, Delete operations
    - Query filtering and searching
    - Transaction management
    - Error handling
    """

    def __init__(self, model: Type[T], session: Any):
        """
        Initialize repository with model class and database session.

        Args:
            model: The entity model class
            session: Database session or connection
        """
        self.model = model
        self.session = session
        self.logger = logging.getLogger(self.__class__.__name__)

    async def create(self, data: Dict[str, Any]) -> T:
        """
        Create new entity from data dictionary.

        Args:
            data: Dictionary of entity attributes

        Returns:
            Created entity instance

        Raises:
            RepositoryError: If creation fails
        """
        try:
            # Create entity instance from data
            entity = self.model(**data)

            # Add to session
            await self.session.add(entity)
            await self.session.commit()

            self.logger.info(f"Created {self.model.__name__} with data: {data}")
            return entity

        except Exception as e:
            await self.session.rollback()
            error_msg = f"Failed to create {self.model.__name__}: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e

    async def get_by_id(self, entity_id: int) -> Optional[T]:
        """
        Get entity by ID.

        Args:
            entity_id: Entity ID

        Returns:
            Entity instance or None if not found
        """
        try:
            entity = await self.session.get(self.model, entity_id)

            if entity:
                self.logger.debug(f"Retrieved {self.model.__name__} with id: {entity_id}")
            else:
                self.logger.debug(f"No {self.model.__name__} found with id: {entity_id}")

            return entity

        except Exception as e:
            error_msg = f"Failed to get {self.model.__name__} by id {entity_id}: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e

    async def update(self, entity_id: int, updates: Dict[str, Any]) -> T:
        """
        Update existing entity.

        Args:
            entity_id: Entity ID to update
            updates: Dictionary of fields to update

        Returns:
            Updated entity instance

        Raises:
            RepositoryError: If entity not found or update fails
        """
        try:
            entity = await self.get_by_id(entity_id)

            if not entity:
                raise RepositoryError(f"Entity with id {entity_id} not found")

            # Update entity attributes
            for key, value in updates.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)
                else:
                    self.logger.warning(f"Attribute '{key}' not found on {self.model.__name__}")

            await self.session.commit()

            self.logger.info(f"Updated {self.model.__name__} id {entity_id} with: {updates}")
            return entity

        except RepositoryError:
            raise
        except Exception as e:
            await self.session.rollback()
            error_msg = f"Failed to update {self.model.__name__} id {entity_id}: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e

    async def delete(self, entity_id: int) -> bool:
        """
        Delete entity by ID.

        Args:
            entity_id: Entity ID to delete

        Returns:
            True if deleted, False if not found

        Raises:
            RepositoryError: If deletion fails
        """
        try:
            entity = await self.get_by_id(entity_id)

            if not entity:
                self.logger.debug(f"No {self.model.__name__} found with id {entity_id} to delete")
                return False

            # Remove entity from session data (mock implementation)
            if hasattr(self.session, 'data') and entity_id in self.session.data:
                del self.session.data[entity_id]

            await self.session.commit()

            self.logger.info(f"Deleted {self.model.__name__} with id: {entity_id}")
            return True

        except Exception as e:
            await self.session.rollback()
            error_msg = f"Failed to delete {self.model.__name__} id {entity_id}: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e

    async def list_all(self, limit: Optional[int] = None) -> List[T]:
        """
        List all entities.

        Args:
            limit: Optional limit on number of results

        Returns:
            List of entity instances
        """
        try:
            query = await self.session.query(self.model)

            if limit:
                query = await query.limit(limit)

            entities = await query.all()

            self.logger.debug(f"Retrieved {len(entities)} {self.model.__name__} entities")
            return entities

        except Exception as e:
            error_msg = f"Failed to list {self.model.__name__} entities: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e

    async def find_by(self, **kwargs) -> List[T]:
        """
        Find entities by criteria.

        Args:
            **kwargs: Search criteria as keyword arguments

        Returns:
            List of matching entity instances
        """
        try:
            query = await self.session.query(self.model)
            query = await query.filter(**kwargs)

            entities = await query.all()

            self.logger.debug(f"Found {len(entities)} {self.model.__name__} entities matching: {kwargs}")
            return entities

        except Exception as e:
            error_msg = f"Failed to find {self.model.__name__} by criteria {kwargs}: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e

    async def count(self, **kwargs) -> int:
        """
        Count entities matching criteria.

        Args:
            **kwargs: Search criteria as keyword arguments

        Returns:
            Count of matching entities
        """
        try:
            entities = await self.find_by(**kwargs)
            count = len(entities)

            self.logger.debug(f"Counted {count} {self.model.__name__} entities matching: {kwargs}")
            return count

        except Exception as e:
            error_msg = f"Failed to count {self.model.__name__} entities: {str(e)}"
            self.logger.error(error_msg)
            raise RepositoryError(error_msg) from e

    async def exists(self, entity_id: int) -> bool:
        """
        Check if entity exists by ID.

        Args:
            entity_id: Entity ID to check

        Returns:
            True if entity exists, False otherwise
        """
        try:
            entity = await self.get_by_id(entity_id)
            exists = entity is not None

            self.logger.debug(f"{self.model.__name__} id {entity_id} exists: {exists}")
            return exists

        except RepositoryError:
            return False