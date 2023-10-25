import os
import openai
import gradio
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import LLMPredictor, ServiceContext
from langchain.chat_models import ChatOpenAI
from llama_index import Prompt
from llama_index import StorageContext, load_index_from_storage

openai.api_key = ""  # Reemplaza con tu clave de API de OpenAI
os.environ["OPENAI_API_KEY"] = "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"  # Poner la clave de API de OpenAI

# Cargar documentos de un directorio (opcional)
documents = SimpleDirectoryReader('book').load_data()

# Crear un índice a partir de los documentos (opcional)
index = VectorStoreIndex.from_documents(documents)

# Persistir el índice en disco (opcional)
index.storage_context.persist("naval_index")

# Rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="naval_index")

# Load index from storage
new_index = load_index_from_storage(storage_context)

# Create query engine from index
new_query_engine = new_index.as_query_engine()

# Create predictor using a custom model
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))

# Create service context with a custom predictor
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# Create index using the service context
custom_llm_index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

# Define a custom prompt
template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "eres una experto analizador de libros, busca en cada uno de los libros entregados y da tu respuesta con mucho detalle sin importar el largo de la respuesta"
    "Por favor, proporciona una respuesta completa y exhaustiva que incluya todos los detalles relevantes"
    "Al final de tu respuesta, por favor, enumera los puntos importantes y proporciona una conclusión general"
    "Given this information, please answer the question and each answer should start with code word Lucy: {query_str}\n"
)
qa_template = Prompt(template)

# Use the custom prompt when querying
query_engine = custom_llm_index.as_query_engine(text_qa_template=qa_template)

# Define a callable function for the gradio interface
def query_function(query_str):
    response = query_engine.query(query_str)
    return response

# Create Gradio interface
app = gradio.Interface(
    query_function,
    inputs="text",
    outputs="text",
    title="SUPER! Asistente de interpretador de textos",
    description="Este asistente te ayuda a encontrar información sobre libros y relacionar esa información. Los textos fueron cargados previamente. En este momento tiene 4 libros: 'Las aventuras de Huckleberry Finn', 'Las aventuras de Tom Sawyer', 'Tom Sawyer en el extranjero' y 'Tom Sawyer Detective.'",
)

# Run the app
app.launch(share=True)
