from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains.qa_with_sources.stuff_prompt import template

custom_qa_prompt_template = """You are a domain-neutral research assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Instructions:
- Base your answer strictly on the provided context
- If the context doesn't contain the answer, say "I don't have enough information to answer that"
- Include specific names, numbers, dates, metrics, and definitions when available
- Cite sources when making factual claims

Answer:"""

new_template = custom_qa_prompt_template + template

PROMPT = PromptTemplate(template=new_template, input_variables=["summaries", "question"])
