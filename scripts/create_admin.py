#!/usr/bin/env python
"""CLI script to create an admin user.

Usage:
    python scripts/create_admin.py --email admin@example.com --name "Admin Name" --password secret123

Or interactively (will prompt for password):
    python scripts/create_admin.py --email admin@example.com --name "Admin Name"
"""

import argparse
import asyncio
import getpass
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.services.admin_service import create_admin


async def main():
    parser = argparse.ArgumentParser(description="Create an admin user")
    parser.add_argument("--email", required=True, help="Admin email address")
    parser.add_argument("--name", required=True, help="Admin full name")
    parser.add_argument("--password", help="Admin password (will prompt if not provided)")
    parser.add_argument(
        "--role",
        default="super_admin",
        choices=["super_admin"],
        help="Admin role (default: super_admin)",
    )

    args = parser.parse_args()

    # Get password interactively if not provided
    password = args.password
    if not password:
        password = getpass.getpass("Enter password: ")
        password_confirm = getpass.getpass("Confirm password: ")
        if password != password_confirm:
            print("Error: Passwords do not match")
            sys.exit(1)

    if len(password) < 8:
        print("Error: Password must be at least 8 characters")
        sys.exit(1)

    # Create database engine
    engine = create_async_engine(settings.database_url, echo=False)

    async with AsyncSession(engine) as session:
        try:
            admin = await create_admin(
                session=session,
                email=args.email,
                password=password,
                full_name=args.name,
                role=args.role,
            )
            print(f"✅ Admin creado exitosamente:")
            print(f"   ID: {admin.id}")
            print(f"   Email: {admin.email}")
            print(f"   Nombre: {admin.full_name}")
            print(f"   Rol: {admin.role}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        finally:
            await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
