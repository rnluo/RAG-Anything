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
    file_path: str,
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

    # Initialize RAGAnything with new dataclass structure
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # Process document
    content_list, _ = await rag.parse_document(
        file_path=file_path, output_dir=output_dir, parse_method="auto"
    )
    text_content, multimodal_items = separate_content(content_list)

    for item in multimodal_items:
        print(item)

    # Try to load QA pairs from cache
    cache_path = os.path.join(output_dir, "qa_cache.pkl")
    if os.path.exists(cache_path):
        logger.info(f"Loading QA pairs from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            evaluation_data = pickle.load(f)
    else:
        # Generate text QA pairs
        text_qa_pairs = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text_content)
        prompt_template = PROMPTS["qa_generation_prompt"]
        for chunk in chunks:
            prompt = prompt_template.format(
                type="text",
                context=chunk
            )
            response = await llm_model_func(prompt)
            try:
                response_json = json.loads(response)
                text_qa_pairs.append({
                    "context": chunk,
                    "question": response_json["question"],
                    "ground_truth_answer": response_json["answer"]
                })
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.error(f"Failed to parse text QA response: {response}")

        # Generate multimodal QA pairs
        table_qa_pairs = []
        image_qa_pairs = []
        equation_qa_pairs = []
        for item in multimodal_items:
            type = item.get("type")
            if type == "table":
                df = pd.read_html(io.StringIO(item.get('table_body')))[0]
                context = f"Caption: {item.get('table_caption')}, Body: {df.to_markdown(index=False)}, Footnote: {item.get('table_footnote')}."
                prompt = prompt_template.format(
                    type=type,
                    context=context
                )
                response = await llm_model_func(prompt)
                fixed_response = response.replace('\\', '\\\\')
                try:
                    response_json = json.loads(fixed_response)
                    table_qa_pairs.append({
                        "context": context,
                        "question": response_json["question"],
                        "ground_truth_answer": response_json["answer"]
                    })
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.error(f"Failed to parse {type} QA response: {response}")


            elif type == "image":
                img_path = item.get("img_path")
                if img_path and os.path.exists(img_path):
                    with open(img_path, "rb") as image_file:
                        image_data_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                    context = f"Caption: {item.get('image_caption')}, Footnote: {item.get('image_footnote')}."
                    prompt = prompt_template.format(
                        type=type,
                        context=context
                        )
                    response = await vision_model_func(prompt, image_data=image_data_base64)
                    try:
                        response_json = json.loads(response)
                        image_qa_pairs.append({
                            "context": context,
                            "question": response_json["question"],
                            "ground_truth_answer": response_json["answer"]
                        })
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.error(f"Failed to parse {type} QA response: {response}")


            elif type == "equation":
                context = item.get("text")
                prompt = prompt_template.format(
                    type=type,
                    context=context
                    )
                response = await llm_model_func(prompt)
                fixed_response = response.replace('\\', '\\\\')
                try:
                    response_json = json.loads(fixed_response)
                    equation_qa_pairs.append({
                        "context": context,
                        "question": response_json["question"],
                        "ground_truth_answer": response_json["answer"]
                    })
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.error(f"Failed to parse {type} QA response: {response}")

        evaluation_data = {
            "text": text_qa_pairs,
            "table": table_qa_pairs, 
            "image": image_qa_pairs, 
            "equation": equation_qa_pairs
        }

        logger.info(f"Saving QA pairs to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(evaluation_data, f)

    await rag.process_document_complete(file_path=file_path, output_dir=output_dir)

    # Evaluate questions & collect RAG-Anything responses
    for type, eval_items in evaluation_data.items():
        logger.info(f"Collecting answers for {len(eval_items)} questions with type: {type}")
        for eval_item in eval_items:
            context = eval_item["context"]
            question = eval_item["question"]
            if not question:
                continue
            
            if False:
                # Filter out low-quality questions
                critique_prompt_templates = {
                    "groundness": PROMPTS["question_groundness_critique_prompt"],
                    "relevance": PROMPTS["question_relevance_critique_prompt"],
                    "standalone": PROMPTS["question_standalone_critique_prompt"]
                }
                low_quality = False
                for critique_type, critique_prompt_template in critique_prompt_templates.items():
                    if low_quality:
                        break
                    if critique_type == "groundness":
                        critique_prompt = critique_prompt_template.format(
                            context=context, 
                            question=question
                        )
                    else:
                        critique_prompt = critique_prompt_template.format(
                            question=question
                        )
                    response = await llm_model_func(critique_prompt)
                    fixed_response = response.replace('\\', '\\\\')
                    try:
                        response_json = json.loads(fixed_response)
                        score = int(response_json.get("rating", 0))
                        if score < 2:
                            eval_item["rag_response"] = "[SKIP]"
                            low_quality = True
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.error(f"Failed to parse critique: {response}")
                        low_quality = True
                if low_quality:
                    continue

            logger.info(f"\n[Text Query]: {question}")
            result = await rag.aquery(question, mode="hybrid")
            eval_item["rag_response"] = result
            logger.info(f"Answer: {result}")

    for type_data in evaluation_data.values():
        print("=========================================")
        print(type_data[:2])
        print("-----------------------------------------")

    # LLM evaluation
    evaluation_scores = {
        "text": [],
        "table": [], 
        "image": [], 
        "equation": []
    }
    for type, eval_items in evaluation_data.items():
        logger.info(f"Evaluating {len(eval_items)} questions for type: {type}")
        for eval_item in eval_items:
            if eval_item.get("rag_response") == "[SKIP]":
                continue

            evaluation_prompt_template = PROMPTS["evaluation_prompt"]
            evaluation_prompt = evaluation_prompt_template.format(
                instruction=eval_item.get("question"),
                response=eval_item.get("rag_response"),
                reference_answer=eval_item.get("ground_truth_answer")
            )

            result = await llm_model_func(evaluation_prompt)
            
            try:
                score = result.split("[RESULT]")[-1].strip()
                evaluation_scores[type].append(score)
            except (ValueError, IndexError) as e:
                logger.error(f"Failed to parse {result}. Error: {e}")


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
    parser.add_argument("file_path", help="Path to the document to process")
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
            args.file_path,
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
