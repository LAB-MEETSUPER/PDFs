import os
import openai
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import LLMPredictor, ServiceContext
from langchain.chat_models import ChatOpenAI
from llama_index import Prompt
from llama_index import StorageContext, load_index_from_storage

#openai.api_key = ""  # Replace with your OpenAI API key
#os.environ["OPENAI_API_KEY"] = ""  # Poner la OpenAI API key


# Load documents from a directory
documents = SimpleDirectoryReader('book').load_data()

# Create an index from the documents
index = VectorStoreIndex.from_documents(documents)

# Create a query engine from the index
query_engine = index.as_query_engine()


#print(response)
# Persist index to disk
index.storage_context.persist("naval_index")



# Rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="naval_index") #<- aca se hace el storage.

# Load index from the storage context
new_index = load_index_from_storage(storage_context) #<-aca se sube el storage

new_query_engine = new_index.as_query_engine() #<- crea un motor de consulta a partir del indice
response = new_query_engine.query("who is this text about?")



# Create a predictor using a custom model
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))

# Create a service context with the custom predictor
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# Create an index using the service context / crea un contexto de servicio con un predictor personalizado
custom_llm_index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

custom_llm_query_engine = custom_llm_index.as_query_engine()



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

# Print a welcome message
print("Hola, soy un asistente de interpretador de textos basado en documentos especialmente proporcionados por el usuario. ¿Cuál es tu consulta?")

# Get the user's query
while True:
    consulta = input("ingrese su consulta: \n")
    if (consulta == "salir"):
        break

    # Get the user's response
    response = query_engine.query(consulta)

    # Print the user's query
    print("Lucy: " + consulta)

    # Print the AI's response
    print(response)

    # Ask the user if they have another question
    print("¿Tienes otra consulta?")