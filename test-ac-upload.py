import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from typing import List
import os


WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

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


# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)


# asyncio.run(test_funcs())


async def main():
    try:
        embedding_dimension = await get_embedding_dim()
        print(f"Detected embedding dimension: {embedding_dimension}")

        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=8192,
                func=embedding_func,
            ),
        )

        with open("./book.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        # Perform naive search
        print(
            await rag.aquery(
                "What are the top themes in this story?", param=QueryParam(mode="naive")
            )
        )

        # Perform local search
        print(
            await rag.aquery(
                "What are the top themes in this story?", param=QueryParam(mode="local")
            )
        )

        # Perform global search
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="global"),
            )
        )

        # Perform hybrid search
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="hybrid"),
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
