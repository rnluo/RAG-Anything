import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path

from evaluation_prompts import PROMPTS
import base64
import json
import io
import sys
import pandas as pd
import pickle
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image
import tempfile
import shutil


sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig
from raganything.utils import separate_content

from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


def configure_logging():
    """Configure logging for the application"""
    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_example.log"))

    print(f"\nRAGAnything example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


async def process_with_rag(
    #file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    parser: str = None,
):
    """
    Process document with RAGAnything

    Args:
        file_path: Path to the document
        output_dir: Output directory for RAG results
        api_key: OpenAI API key
        base_url: Optional base URL for API
        working_dir: Working directory for RAG storage
    """

    # Create RAGAnything configuration
    config = RAGAnythingConfig(
        working_dir=working_dir or "./rag_storage",
        parser=parser,  # Parser selection: mineru or docling
        parse_method="auto",  # Parse method: auto, ocr, or txt
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Define LLM model function
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    # Define vision model function for image processing
    def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=[],
        image_data=None,
        messages=None,
        **kwargs,
    ):
        # If messages format is provided (for multimodal VLM enhanced query), use it directly
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                #system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Traditional single image format
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    #{"role": "system", "content": system_prompt}
                    #if system_prompt
                    #else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
                            },
                        ],
                    }
                    if image_data
                    else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Pure text format
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # Define embedding function
    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-large",
            api_key=api_key,
            base_url=base_url,
        ),
    )
    
    # Load all annotations in dataset/extracted_questions
    annotations = []
    extracted_questions_dir = Path("dataset/extracted_questions")
    for jsonl_file in extracted_questions_dir.glob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                annotations.append(data)
    
    # Load document data
    parquet_path = "dataset/MMDocIR_pages.parquet"
    table = pq.read_table(parquet_path)
    df_pages = table.to_pandas()

    # Extract data for each question
    questions_data = []
    for annot in annotations:
        doc_name = annot['doc_name']
        start_idx, end_idx = annot['page_indices']

        question = annot["Q"]
        answer = annot["A"]
        question_type = annot["type"]
        page_ids = annot["page_id"]
        layout_mapping = annot["layout_mapping"]

        image_data_list = []
        question_pages_df = df_pages.iloc[start_idx:end_idx]
        for page_id in page_ids:
            row_df = question_pages_df[question_pages_df["passage_id"] == str(page_id)]
            if not row_df.empty:
                row = row_df.iloc[0]
                image_binary = row['image_binary']
                image_path = row['image_path']
                ocr_text = row['ocr_text']
                image_data_list.append(
                    {
                        'page_id': page_id,
                        'image_binary': image_binary,
                        'image_path': image_path,
                        'ocr_text': ocr_text
                    }
                )
        questions_data.append({
                'doc_name': doc_name,
                'question': question,
                'answer': answer,
                'type': question_type,
                'page_ids': page_ids,
                'layout_mapping': layout_mapping,
                'image_data': image_data_list
            })

    # Collect RAG-Anything responses to the questions
    for i, question_data in enumerate(questions_data):
        logger.info(f"Collecting response to question {i+1}/{len(questions_data)}")

        # Remove the knowledge base
        if os.path.exists(config.working_dir):
            shutil.rmtree(config.working_dir)

        # Reinitialize RAGAnything for each independent question
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        question = question_data["question"]
        question_type = question_data["type"]
        image_data_list = question_data["image_data"]
        layout_mapping = question_data["layout_mapping"]

        
        temp_files = []
        # For pure-text questions, pass OCR text
        if "Pure-text" in question_type: 
            full_ocr_text = "\n\n".join([image_data["ocr_text"] for image_data in image_data_list])
            
            # Write OCR text to a temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as temp_file:
                temp_file.write(full_ocr_text)
                temp_files.append(temp_file.name)
            
        # For multimodal questions, pass cropped original images
        else:
            page_id_to_image_binary = {page['page_id']: page['image_binary'] for page in image_data_list}
            for layout_info in layout_mapping:
                page_num = layout_info['page']
                bbox = layout_info['bbox']
                if page_num in page_id_to_image_binary:
                    try:
                        image = Image.open(io.BytesIO(page_id_to_image_binary[page_num]))
                        cropped_image = image.crop(bbox)
                        
                        # Write cropped image to a temporary file
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img_file:
                            cropped_image.save(temp_img_file.name, "PNG")
                            temp_files.append(temp_img_file.name)   
                    except Exception as e:
                        logger.error(f"Failed to crop image for page {page_num}: {e}")

        logger.info(f"{len(temp_files)}")

        # Add each temp file (text or image) to the RAG knowledge base
        if temp_files:
            for temp_file_path in temp_files:
                await rag.process_document_complete(file_path=temp_file_path, output_dir=output_dir)

            # Query RAG-Anything
            result = await rag.aquery(question, mode="hybrid")
            question_data["rag_response"] = result
            logger.info(f"RAG Response: {result}")
        else:
            question_data["rag_response"] = "<SKIP>"
            logger.warning(f"No documents found for the question: {question}")
        
        # Cache RAG response after each question
        cache_path = os.path.join(output_dir, "rag_responses_cache.pkl")
        try:
            # 'image_binary' is not serializable to JSON, so we must exclude it.
            cached_data = question_data.copy()
            if 'image_data' in cached_data:
                for item in cached_data['image_data']:
                    item.pop('image_binary', None)

            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(cached_data) + "\n")
        except Exception as e:
            logger.error(f"Error caching RAG response: {e}")

        for path in temp_files:
            try:
                os.remove(path)
            except OSError as e:
                logger.error(f"Error removing temp file {path}: {e}")

    # LLM evaluation
    evaluation_scores = {}
    for question_data in questions_data:
            if question_data.get("rag_response") == "<SKIP>":
                continue

            evaluation_prompt_template = PROMPTS["evaluation_prompt"]
            evaluation_prompt = evaluation_prompt_template.format(
                instruction=question_data.get("question"),
                response=question_data.get("rag_response"),
                reference_answer=question_data.get("answer")
            )

            result = await llm_model_func(evaluation_prompt)
            
            try:
                score = result.split("[RESULT]")[-1].strip()
                question_type = question_data["type"]
                if question_type not in evaluation_scores.keys():
                    evaluation_scores[question_type] = []
                evaluation_scores[question_type].append(score)
            except (ValueError, IndexError) as e:
                logger.error(f"Failed to parse score from {result}. Error: {e}")


    for type, type_scores in evaluation_scores.items():
        int_scores = []
        for score in type_scores:
            try:
                int_scores.append(int(score))
            except (ValueError, TypeError):
                logger.error(f"Failed to convert score {score}. Error: {e}")

        avg_score = sum(int_scores) / len(int_scores) if int_scores else -1
        logger.info(f"Type: {type}\nItem count: {len(int_scores)}\nAverage score: {avg_score}")

def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="MinerU RAG Example")
    #parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument(
        "--working_dir", "-w", default="./rag_storage", help="Working directory path"
    )
    parser.add_argument(
        "--output", "-o", default="./output", help="Output directory path"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_BINDING_API_KEY"),
        help="OpenAI API key (defaults to LLM_BINDING_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="Optional base URL for API",
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        help="Optional base URL for API",
    )

    args = parser.parse_args()

    # Check if API key is provided
    if not args.api_key:
        logger.error("Error: OpenAI API key is required")
        logger.error("Set api key environment variable or use --api-key option")
        return

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Process with RAG
    asyncio.run(
        process_with_rag(
            #args.file_path,
            args.output,
            args.api_key,
            args.base_url,
            args.working_dir,
            args.parser,
        )
    )


if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    print("RAGAnything Example")
    print("=" * 30)
    print("Processing document with multimodal RAG pipeline")
    print("=" * 30)

    main()
