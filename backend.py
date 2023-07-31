import time
import streamlit as st
import torch
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from constants import EMBEDDING_MODEL_NAME, db_all, db_frankl, db_inst
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)


@st.cache_resource(show_spinner=False)
def load_model(device_type, model_id):
    if device_type.lower() == "cuda":  # cuda
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.tie_weights()
    else:  # cpu
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    generation_config = GenerationConfig.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.5,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm


@st.cache_data(persist=True, show_spinner=False)
def get_embeddings():
    if torch.cuda.is_available():
        device_type = "cuda"
    else:
        device_type = "cpu"

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type}
    )

    return embeddings


def get_llm():
    if torch.cuda.is_available():
        device_type = "cuda"
    else:
        device_type = "cpu"

    model_id = "psmathur/orca_mini_3b"

    llm = load_model(
        device_type,
        model_id=model_id
    )
    return llm


def load_qa(db_option):
    db_instance = db_inst

    match db_option:
        case "All":
            db_instance = db_all
            print("loading db_all")
        case "Frankl's Works":
            db_instance = db_frankl
            print("loading db_frankl")
        case "Journal of Search for Meaning":
            db_instance = db_inst
            print("loading db_inst")

    load_start = time.time()

    embeddings = get_embeddings()

    db = Chroma(
        persist_directory=db_instance.get_directory(),
        embedding_function=embeddings,
        client_settings=db_instance.get_chroma_settings(),
    )

    retriever = db.as_retriever()

    llm = get_llm()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    load_end = time.time()
    print(f"\n> Completed Initial Load (took {round(load_end - load_start, 2)} s.)")

    return qa
