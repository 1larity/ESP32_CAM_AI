# AI/enrollment/__init__.py
# Public entry points for the enrollment subsystem.

from __future__ import annotations

from .service import EnrollmentService


def get_enrollment_service() -> EnrollmentService:
    """Convenience accessor for the singleton EnrollmentService."""
    return EnrollmentService.instance()
