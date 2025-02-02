import os
import asyncio
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
import numpy as np
from pinecone.grpc import PineconeGRPC as Pinecone
from typing import List
import textract
import psycopg2
from datetime import datetime

# Database configuration (replace with your credentials)
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Configure logging to file and console
log_file_path = "upload_status.log"  # Path to the log file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),  # Log to file
        logging.StreamHandler(),  # Log to console
    ],
)
logger = logging.getLogger(__name__)

WORKING_DIR = "./ac"
UPLOAD_FOLDER = "/Users/kishorereddy/Documents/amaravati_chamber_rag_store/filtered"  # Folder containing multiple files to upload

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
    logger.warning(f"Upload folder '{UPLOAD_FOLDER}' does not exist. Created it.")

async def pinecone_embedding(texts: List[str], model_name: str = "multilingual-e5-large") -> List[List[float]]:
    """
    Custom embedding function using Pinecone's API.
    :param texts: List of text strings to embed.
    :param model_name: Name of the Pinecone embedding model.
    :return: List of embedding vectors.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Generate embeddings using Pinecone's inference.embed method
    embeddings = pc.inference.embed(
        model=model_name,
        inputs=texts,
        parameters={"input_type": "passage", "truncate": "END"}
    )
    
    return [e['values'] for e in embeddings.data]


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "qwen/qwen-2.5-7b-instruct",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        **kwargs,
    )


embedding_func = EmbeddingFunc(
    embedding_dim=os.getenv("EMBEDDING_DIM"),
    max_token_size=os.getenv("MAX_EMBED_TOKENS"),
    func=pinecone_embedding  # Use the custom Pinecone embedding function
)


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    # Convert the list of embeddings to a NumPy array
    embedding_array = np.array(embedding)
    # Get the embedding dimension from the shape of the array
    embedding_dim = embedding_array.shape[1]
    return embedding_dim

# Function to connect to the PostgreSQL database
async def connect_db():
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        return conn
    except psycopg2.Error as e:
        logger.error(f"Error connecting to database: {e}")
        return None

# Function to create the upload status table (run once)
async def create_upload_status_table(conn):
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS upload_status (
                id SERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL,  -- "pending", "success", "failed"
                last_attempt TIMESTAMP,
                error_message TEXT
            );
        """
        )
        conn.commit()
        cur.close()
    except psycopg2.Error as e:
        logger.error(f"Error creating table: {e}")

# Function to update upload status in the database
async def update_upload_status(conn, filename, status, error_message=None):
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO upload_status (filename, status, last_attempt, error_message)
            VALUES (%s, %s, NOW(), %s)
            ON CONFLICT (filename) DO UPDATE SET status = %s, last_attempt = NOW(), error_message = %s;
        """,
            (filename, status, error_message, status, error_message),
        )
        conn.commit()
        cur.close()
    except psycopg2.Error as e:
        logger.error(f"Error updating upload status: {e}")

async def upload_folder_to_rag(rag: LightRAG, folder_path: str, conn):  # Add db connection
    """Upload files and track status in the database."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            await update_upload_status(conn, filename, "pending")  # Initial status
            try:
                if filename.lower().endswith(".pdf"):
                    text_content = textract.process(file_path).decode("utf-8")
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text_content = f.read()
                await rag.ainsert(text_content)
                await update_upload_status(conn, filename, "success")
                logger.info(f"Successfully uploaded file: {filename}")
            except Exception as e:
                logger.error(f"Failed to upload file {filename}: {e}")
                await update_upload_status(conn, filename, "failed", str(e))
        else:
            logger.warning(f"Skipping non-file item: {filename}")


async def main():
    conn = await connect_db()
    if conn:
        await create_upload_status_table(conn)

        try:
            embedding_dimension = await get_embedding_dim()
            logger.info(f"Detected embedding dimension: {embedding_dimension}")

            rag = LightRAG(
                working_dir=WORKING_DIR,
                llm_model_func=llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=embedding_dimension,
                    max_token_size=os.getenv("MAX_EMBED_TOKENS"),
                    func=embedding_func,
                ),
            )

            await upload_folder_to_rag(rag, UPLOAD_FOLDER, conn)  # Pass db connection

        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
            conn.close()  # Close the connection in the finally block
    else:
        logger.error("Failed to connect to the database. Exiting.")


if __name__ == "__main__":
    asyncio.run(main())
