import traceback
import sys

print("Testing imports...")
for stmt in [
    "import torch",
    "import transformers",
    "from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification",
    "from transformers import PreTrainedModel",
    "import sentence_transformers",
    "from sentence_transformers import SentenceTransformer",
    "import bertopic",
    "from bertopic import BERTopic",
]:
    try:
        exec(stmt)
        print(f"SUCCESS: {stmt}")
    except Exception as e:
        print(f"FAILED: {stmt}")
        print(f"Error: {e}")
        traceback.print_exc()
