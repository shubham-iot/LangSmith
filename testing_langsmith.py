import openai, os 

from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
#from langchain.vectorstores.chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter 

from langchain.document_loaders.pdf import PyPDFDirectoryLoader 
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from ragas import EvaluationDataset

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import AspectCritic
from langsmith import traceable
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

os.environ["OPENAI_API_TYPE"] = "azure"
project_name = "AI10570_Solution_Metrics_Quality"
# os.environ["OPENAI_API_BASE"] = f"https://aigateway-prod.apps-1.gp-1-prod.openshift.cignacloud.com/api/v1/ai/{project_name}/OAI"
os.environ["OPENAI_API_KEY"] = "cM4KTUeqzWgdTsSoakfgs9mwwSukSmc68dHRXry7294="
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ai-gateway-api.sys.cigna.com/api/v1/AI/{project_name}/OAI"
os.environ["AZURE_OPENAI_ENDPOINT"] =f"https://aigateway-prod.apps-1.gp-1-prod.openshift.cignacloud.com/api/v1/ai/{project_name}/OAI"

os.environ["SSL_CERT_FILE"] = "C:\\Users\\C8k4nn\\AppData\\Local\\anaconda3\\envs\\solution1\\Lib\\site-packages\\certifi\\cacert.pem"


# from uuid import uuid4

# unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "shadow_monarch-v1"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"

LANGSMITH_TRACING='true'
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="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"
LANGSMITH_PROJECT="shadow_monarch-v1"

from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment="ai-coe-gpt4o:analyze", #"aoai:ai-coe-gpt4o", #"gpt-35-turbo",  # or your deployment
    api_version= "2023-05-15",# "2023-06-01-preview",  # or your api version
    temperature=0,
    max_tokens=4009,
    timeout=None,
    max_retries=2,
    # other params...
)
print(llm.invoke("Hi"))
print("^*&^*&^*&^*&^*&^*&^*&^*&*&************")


dataset = [
    {
        "user_input": "Which CEO is widely recognized for democratizing AI education through platforms like Coursera?",
        "retrieved_contexts": [
            "Andrew Ng, CEO of Landing AI, is known for his pioneering work in deep learning and for democratizing AI education through Coursera."
        ],
        "response": "Andrew Ng is widely recognized for democratizing AI education through platforms like Coursera.",
        "reference": "Andrew Ng, CEO of Landing AI, is known for democratizing AI education through Coursera.",
    },
    {
        "user_input": "Who is Sam Altman?",
        "retrieved_contexts": [
            "Sam Altman, CEO of OpenAI, has advanced AI research and advocates for safe, beneficial AI technologies."
        ],
        "response": "Sam Altman is the CEO of OpenAI and advocates for safe, beneficial AI technologies.",
        "reference": "Sam Altman, CEO of OpenAI, has advanced AI research and advocates for safe AI.",
    },
    {
        "user_input": "Who is Demis Hassabis and how did he gain prominence?",
        "retrieved_contexts": [
            "Demis Hassabis, CEO of DeepMind, is known for developing systems like AlphaGo that master complex games."
        ],
        "response": "Demis Hassabis is the CEO of DeepMind, known for developing systems like AlphaGo.",
        "reference": "Demis Hassabis, CEO of DeepMind, is known for developing AlphaGo.",
    },
    {
        "user_input": "Who is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's product ecosystem?",
        "retrieved_contexts": [
            "Sundar Pichai, CEO of Google and Alphabet Inc., leads innovation across Google's product ecosystem."
        ],
        "response": "Sundar Pichai is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's product ecosystem.",
        "reference": "Sundar Pichai, CEO of Google and Alphabet Inc., leads innovation across Google's product ecosystem.",
    },
    {
        "user_input": "How did Arvind Krishna transform IBM?",
        "retrieved_contexts": [
            "Arvind Krishna, CEO of IBM, transformed the company by focusing on cloud computing and AI solutions."
        ],
        "response": "Arvind Krishna transformed IBM by focusing on cloud computing and AI solutions.",
        "reference": "Arvind Krishna, CEO of IBM, transformed the company through cloud computing and AI.",
    },
]

evaluation_dataset = EvaluationDataset.from_list(dataset)
evaluator_llm = LangchainLLMWrapper(llm)
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm,
)
print ( result )
