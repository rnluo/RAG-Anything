# Adapted from https://huggingface.co/learn/cookbook/en/rag_evaluation

from typing import Any

PROMPTS: dict[str, Any] = {}

PROMPTS["qa_generation_prompt"] = """
Your task is to write a factoid question and an answer given a {type} context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Note that your factoid question will be evaluated in three dimentions:
1. groundness, i.e., how well one can answer the given question unambiguously with the given context.
2. relevance, i.e., how useful this question is for others to gain insight into the original material.
3. standalone, i.e., how context-independent is the question, without additional information to be understood.
Try to provide longer and more specific questions to match these expected qualities.

Provide your answer as a raw JSON object, without any markdown formatting (i.e. no ```json).
The JSON object should be structured as follows:
Output:::
{{
"question": (your factoid question)
"answer": (your answer to the factoid question)
}}

Now here is the context.

context: {context}\n
Output:::"""

PROMPTS["question_groundness_critique_prompt"] = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Note that the focus is to evaluate whether the question is answerable (a quality of the question), not whether the context is clear for the question (a quality of the context).

Provide your answer to be json decodable as follows:

Answer:::
{{
"evaluation": (your rationale for the rating, as a text)
"rating": (your rating, as a number between 1 and 5)
}}

You MUST provide values for 'Evaluation:' and 'Rating:' in your answer.

Now here are the question and context.

question: {question}\n
context: {context}\n
Answer::: """

PROMPTS["question_relevance_critique_prompt"] = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question is for others to gain insight into the original material.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer to be json decodable as follows:

Answer:::
{{
"evaluation": (your rationale for the rating, as a text)
"rating": (your rating, as a number between 1 and 5)
}}

You MUST provide values for 'Evaluation:' and 'Rating:' in your answer.

Now here is the question.

question: {question}\n
Answer::: """

PROMPTS["question_standalone_critique_prompt"] = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independent this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independent from the context.

Provide your answer to be json decodable as follows:

Answer:::
{{
"evaluation": (your rationale for the rating, as a text)
"rating": (your rating, as a number between 1 and 5)
}}

You MUST provide values for 'Evaluation:' and 'Rating:' in your answer.

Now here is the question.

question: {question}\n
Answer::: """

PROMPTS["evaluation_prompt"] = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""
