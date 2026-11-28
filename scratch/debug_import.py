import logging
import transformers
from transformers.utils import logging as tf_logging

tf_logging.set_verbosity_debug()
logging.basicConfig(level=logging.DEBUG)

print("Attempting to import from transformers...")
try:
    from transformers import PreTrainedModel
    print("PreTrainedModel imported successfully:", PreTrainedModel)
except Exception as e:
    print("Error importing PreTrainedModel:", type(e), e)

try:
    from sentence_transformers import SentenceTransformer
    print("SentenceTransformer imported successfully")
except Exception as e:
    print("Error importing SentenceTransformer:", type(e), e)

try:
    from bertopic import BERTopic
    print("BERTopic imported successfully")
except Exception as e:
    print("Error importing BERTopic:", type(e), e)
