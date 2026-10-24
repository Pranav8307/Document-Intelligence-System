import os
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    document: Mapped[Document] = relationship(back_populates="chunks")


DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None
SessionLocal = sessionmaker(bind=engine) if engine else None


def init_database() -> None:
    if engine:
        Base.metadata.create_all(engine)


def save_document(document_id: str, filename: str, chunks: list[str]) -> None:
    if not SessionLocal:
        return
    with SessionLocal.begin() as session:
        session.add(
            Document(
                id=document_id,
                filename=filename,
                created_at=datetime.now(timezone.utc),
                chunks=[
                    DocumentChunk(chunk_index=index, text=text)
                    for index, text in enumerate(chunks)
                ],
            )
        )


def load_all_documents() -> list[tuple[str, list[str]]]:
    if not SessionLocal:
        return []
    with SessionLocal() as session:
        documents = session.scalars(select(Document)).all()
        return [
            (document.id, [chunk.text for chunk in sorted(document.chunks, key=lambda item: item.chunk_index)])
            for document in documents
        ]


def document_exists(document_id: str) -> bool:
    if not SessionLocal:
        return False
    with SessionLocal() as session:
        return session.get(Document, document_id) is not None