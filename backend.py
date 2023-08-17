import time
import torch
# from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.prompts import PromptTemplate

from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    pipeline,
)

from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY


def load_model(model_id, model_basename=None):

    if model_basename is not None:
        if ".ggml" in model_basename:
            model_path = hf_hub_download(
                repo_id=model_id,
                filename=model_basename
            )
            max_ctx_size = 2048
            kwargs = {"model_path": model_path,
                      "n_ctx": max_ctx_size,
                      "max_tokens": max_ctx_size,
                      "n_gpu_layers": 1000,
                      "n_batch": max_ctx_size}
            return LlamaCpp(**kwargs)

        # else:
        #     if ".safetensors" in model_basename:
        #         model_basename = model_basename.replace(".safetensors", "")

        #     tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        #     model = AutoGPTQForCausalLM.from_quantized(
        #         model_id,
        #         model_basename=model_basename,
        #         use_safetensors=True,
        #         trust_remote_code=True,
        #         device="cuda:0",
        #         use_triton=False,
        #         quantize_config=None,
        #     )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.tie_weights()

    generation_config = GenerationConfig.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        min_length=512,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    return HuggingFacePipeline(pipeline=pipe)


def load_qa():

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"}
    )
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()

    model_id = "psmathur/orca_mini_3b"
    model_basename = None

    # model_id = "TheBloke/vicuna-7B-1.1-HF"
    # model_basename = None

    template = """You are an AI assistant for answering questions about logotherapy. You are given the following
    extracted parts of a annual academic journal. Provide a very detailed comprehensive academic answer. If you don't
    know the answer, just say "I'm not sure." Don't try to make up an answer. If the question is not about the
    psychotherapy and not directly in the given context, politely inform them that you are tuned to only answer
    questions about logotherapy. Question: {question} ========= {context} ========= Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    llm = load_model(
        model_id=model_id,
        model_basename=model_basename
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return qa
