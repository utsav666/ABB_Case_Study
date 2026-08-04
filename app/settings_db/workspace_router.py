from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.auth.dependencies import get_current_user_id
from app.settings_db.gateway import SettingsDatabaseGateway
from app.settings_db.orm_models import WorkspaceEntity, WorkspaceMemberEntity

logger = logging.getLogger(__name__)

workspace_router = APIRouter(
    prefix="/workspaces",
    tags=["workspaces"],
)


class CreateWorkspaceRequest(BaseModel):
    """Request body for creating a workspace."""

    name: str
    slug: str


@workspace_router.get("")
def list_workspaces() -> dict[str, Any]:
    """Get all available workspaces."""
    gateway = SettingsDatabaseGateway()

    statement = select(
        WorkspaceEntity.id,
        WorkspaceEntity.name,
        WorkspaceEntity.slug,
    ).where(WorkspaceEntity.deleted_at.is_(None))

    with gateway.get_session() as session:
        row_mappings = session.execute(statement).mappings().all()

    workspaces = [
        {
            "id": str(row["id"]),
            "name": row["name"],
            "slug": row["slug"],
        }
        for row in row_mappings
    ]

    return {"workspaces": workspaces, "count": len(workspaces)}


@workspace_router.get("/{workspace_id}")
def get_workspace(workspace_id: str) -> dict[str, Any]:
    """Get a specific workspace by ID."""
    try:
        parsed_id = UUID(workspace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid workspace ID")

    gateway = SettingsDatabaseGateway()

    statement = select(
        WorkspaceEntity.id,
        WorkspaceEntity.name,
        WorkspaceEntity.slug,
    ).where(
        WorkspaceEntity.id == parsed_id,
        WorkspaceEntity.deleted_at.is_(None),
    )

    with gateway.get_session() as session:
        row = session.execute(statement).mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="Workspace not found")

    return {
        "id": str(row["id"]),
        "name": row["name"],
        "slug": row["slug"],
    }


@workspace_router.post("")
async def create_workspace(
    payload: CreateWorkspaceRequest,
    user_id: UUID | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    """Create a new workspace and add creator as owner."""
    logger.info("Creating workspace with user_id: %s", user_id)

    gateway = SettingsDatabaseGateway()

    # Create workspace with owner_id
    workspace_statement = insert(WorkspaceEntity).values(
        name=payload.name,
        slug=payload.slug,
        owner_id=user_id,  # Will be None if user not authenticated
    )

    logger.info(
        "Workspace statement values - name: %s, slug: %s, owner_id: %s",
        payload.name,
        payload.slug,
        user_id,
    )
    workspace_statement = workspace_statement.returning(
        WorkspaceEntity.id,
        WorkspaceEntity.name,
        WorkspaceEntity.slug,
        WorkspaceEntity.owner_id,
    )

    try:
        with gateway.get_session() as session:
            workspace_row = session.execute(workspace_statement).mappings().first()

            if not workspace_row:
                raise HTTPException(
                    status_code=500, detail="Failed to create workspace"
                )

            workspace_id = workspace_row["id"]
            logger.info(
                "Workspace created with id: %s, owner_id from DB: %s",
                workspace_id,
                workspace_row.get("owner_id"),
            )

            # Create workspace_member entry if user is authenticated
            if user_id:
                logger.info("Creating workspace_member entry for user_id: %s", user_id)
                # Try common role enum values
                # Common values: ADMIN, EDITOR, VIEWER, OWNER, MEMBER
                member_statement = insert(WorkspaceMemberEntity).values(
                    workspace_id=workspace_id,
                    user_id=user_id,
                    role="ADMIN",  # Changed from OWNER to ADMIN
                )
                session.execute(member_statement)
                logger.info("Workspace member entry created with role ADMIN")
            else:
                logger.warning(
                    "No user_id provided, skipping workspace_member creation"
                )

            session.commit()
            logger.info("Transaction committed successfully")

        return {
            "workspace": {
                "id": str(workspace_id),
                "name": workspace_row["name"],
                "slug": workspace_row["slug"],
            }
        }
    except Exception as e:
        if "unique constraint" in str(e).lower():
            raise HTTPException(
                status_code=400, detail="Workspace with this slug already exists"
            )
        raise HTTPException(status_code=500, detail=str(e))
