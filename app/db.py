import json
import asyncio
import psycopg2
from psycopg2.extras import Json
from app.config import config
from app.logger import logger

def initialize_database():
    """
    Connects to the Neon PostgreSQL instance and initializes the schema.
    Creates the `caption_runs` and `pipeline_errors` tables if they do not exist.
    """
    db_url = config.neon_db_url
    if not db_url:
        logger.warning("NEON_DB_URL is not set. Database logging is disabled.")
        return False

    logger.info("Initializing Neon PostgreSQL Database schemas...")
    conn = None
    try:
        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            # 1. Main execution logs table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS caption_runs (
                    id SERIAL PRIMARY KEY,
                    run_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    video_url TEXT NOT NULL,
                    inference_type VARCHAR(50) NOT NULL,
                    video_inference TEXT,
                    captions JSONB,
                    metadata JSONB
                );
            """)
            # 2. Execution errors logging table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_errors (
                    id SERIAL PRIMARY KEY,
                    error_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    video_url TEXT,
                    stage VARCHAR(50) NOT NULL,
                    error_message TEXT NOT NULL,
                    metadata JSONB
                );
            """)
            conn.commit()
        logger.info("Neon database schemas initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


def _insert_run_sync(video_url: str, inference_type: str, video_inference: str, captions: dict, metadata: dict):
    """Synchronous Postgres insert helper, run within thread context."""
    db_url = config.neon_db_url
    if not db_url:
        return

    conn = None
    try:
        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO caption_runs (video_url, inference_type, video_inference, captions, metadata)
                VALUES (%s, %s, %s, %s, %s);
            """, (
                video_url,
                inference_type,
                video_inference,
                Json(captions),
                Json(metadata)
            ))
            conn.commit()
        logger.info(f"Successfully logged run metadata to Neon DB for video: {video_url}")
    except Exception as e:
        logger.error(f"Neon database insert failed for {video_url}: {e}")
        if conn:
            conn.rollback()
        # Log the insert failure itself to the pipeline_errors table
        _insert_error_sync(
            video_url=video_url,
            stage="db_insert",
            error_message=str(e),
            metadata={"failed_insert": {"inference_type": inference_type, "metadata": metadata}}
        )
    finally:
        if conn:
            conn.close()


def _insert_error_sync(video_url: str, stage: str, error_message: str, metadata: dict):
    """Synchronous Postgres error logging helper."""
    db_url = config.neon_db_url
    if not db_url:
        return

    conn = None
    try:
        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO pipeline_errors (video_url, stage, error_message, metadata)
                VALUES (%s, %s, %s, %s);
            """, (
                video_url,
                stage,
                error_message,
                Json(metadata)
            ))
            conn.commit()
        logger.info(f"Logged pipeline stage error ({stage}) to database for: {video_url}")
    except Exception as e:
        logger.error(f"Failed to log error to Neon database: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


async def log_caption_run(video_url: str, inference_type: str, video_inference: str, captions: dict, metadata: dict):
    """
    Asynchronously logs a completed video inference and its style captions to the Neon DB.
    Uses asyncio.to_thread to run the blocking DB insert in a separate worker thread.
    """
    if config.is_local:
        logger.info("Local environment detected: skipping database captions logging.")
        return

    db_url = config.neon_db_url
    if not db_url:
        return

    try:
        await asyncio.to_thread(
            _insert_run_sync,
            video_url,
            inference_type,
            video_inference,
            captions,
            metadata
        )
    except Exception as e:
        logger.error(f"Failed to schedule background database insert: {e}")


async def log_pipeline_error(video_url: str, stage: str, error_message: str, metadata: dict):
    """
    Asynchronously logs a pipeline execution error to the database.
    """
    if config.is_local:
        logger.info(f"Local environment detected: skipping database error logging for stage {stage}.")
        return

    db_url = config.neon_db_url
    if not db_url:
        return

    try:
        await asyncio.to_thread(
            _insert_error_sync,
            video_url,
            stage,
            error_message,
            metadata
        )
    except Exception as e:
        logger.error(f"Failed to schedule background error logging: {e}")
