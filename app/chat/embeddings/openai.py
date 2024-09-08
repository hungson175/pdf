from langchain.embeddings import OpenAIEmbeddings
# disable the deprecated warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
embeddings = OpenAIEmbeddings()