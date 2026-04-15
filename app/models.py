from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Category(Base):
    __tablename__ = "categories"

    name: Mapped[str] = mapped_column(Text, primary_key=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    color: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    notes: Mapped[list["CategoryNote"]] = relationship(back_populates="category_rel")
    tasks: Mapped[list["Task"]] = relationship(back_populates="category_rel")


class Context(Base):
    __tablename__ = "contexts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    kind: Mapped[str | None] = mapped_column(Text)  # srl|asbl|independant|interne|perso
    notes: Mapped[str | None] = mapped_column(Text)
    aliases: Mapped[str | None] = mapped_column(Text)  # JSON array
    archived: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    tasks: Mapped[list["Task"]] = relationship(back_populates="context")


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(Text, ForeignKey("categories.name"))
    context_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("contexts.id"))
    urgency: Mapped[str | None] = mapped_column(Text)  # critique|haute|normale|basse
    due_date: Mapped[date | None] = mapped_column(Date)
    estimated_minutes: Mapped[int | None] = mapped_column(Integer)
    actual_minutes: Mapped[int | None] = mapped_column(Integer)
    tags: Mapped[str | None] = mapped_column(Text)  # JSON array
    status: Mapped[str] = mapped_column(Text, default="open")  # open|doing|waiting|done|cancelled
    waiting_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    touched_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    llm_raw_response: Mapped[str | None] = mapped_column(Text)
    llm_confidence: Mapped[float | None] = mapped_column(Float)
    llm_reasoning: Mapped[str | None] = mapped_column(Text)
    was_corrected: Mapped[int] = mapped_column(Integer, default=0)
    postponed_count: Mapped[int] = mapped_column(Integer, default=0)
    needs_review: Mapped[int] = mapped_column(Integer, default=0)
    llm_pending: Mapped[bool] = mapped_column(Boolean, default=False)

    category_rel: Mapped["Category | None"] = relationship(back_populates="tasks")
    context: Mapped["Context | None"] = relationship(back_populates="tasks")
    corrections: Mapped[list["Correction"]] = relationship(back_populates="task")
    pending_prompts: Mapped[list["PendingPrompt"]] = relationship(back_populates="task")

    __table_args__ = (
        CheckConstraint(
            "urgency IN ('critique','haute','normale','basse') OR urgency IS NULL",
            name="ck_tasks_urgency",
        ),
        CheckConstraint(
            "status IN ('open','doing','waiting','done','cancelled')",
            name="ck_tasks_status",
        ),
        Index("idx_tasks_status", "status"),
        Index("idx_tasks_due", "due_date"),
        Index("idx_tasks_context", "context_id"),
        Index("idx_tasks_touched", "touched_at"),
    )


class Correction(Base):
    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("tasks.id"))
    field: Mapped[str] = mapped_column(Text, nullable=False)  # category|urgency|due_date|context_id|reasoning
    old_value: Mapped[str | None] = mapped_column(Text)
    new_value: Mapped[str | None] = mapped_column(Text)
    task_title: Mapped[str | None] = mapped_column(Text)
    task_description: Mapped[str | None] = mapped_column(Text)
    corrected_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    task: Mapped["Task | None"] = relationship(back_populates="corrections")

    __table_args__ = (Index("idx_corrections_recent", "corrected_at"),)


class CategoryNote(Base):
    __tablename__ = "category_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category: Mapped[str | None] = mapped_column(Text, ForeignKey("categories.name"))
    note: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    category_rel: Mapped["Category | None"] = relationship(back_populates="notes")


class PendingPrompt(Base):
    __tablename__ = "pending_prompts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("tasks.id"))
    kind: Mapped[str] = mapped_column(Text, nullable=False)  # ask_due_date|zombie_check|estimate_confirm
    message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime)
    resolution: Mapped[str | None] = mapped_column(Text)  # JSON

    task: Mapped["Task | None"] = relationship(back_populates="pending_prompts")


class Digest(Base):
    __tablename__ = "digests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, unique=True, nullable=False)
    content_html: Mapped[str] = mapped_column(Text, nullable=False)
    content_text: Mapped[str] = mapped_column(Text, nullable=False)
    task_ids: Mapped[str | None] = mapped_column(Text)  # JSON array


class Goal(Base):
    __tablename__ = "goals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    starts_at: Mapped[date | None] = mapped_column(Date)
    ends_at: Mapped[date | None] = mapped_column(Date)
    active: Mapped[int] = mapped_column(Integer, default=1)


class DurationCalibration(Base):
    __tablename__ = "duration_calibration"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category: Mapped[str | None] = mapped_column(Text)
    context_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("contexts.id"))
    keyword: Mapped[str | None] = mapped_column(Text)
    llm_estimate: Mapped[int | None] = mapped_column(Integer)
    actual_minutes: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Setting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
