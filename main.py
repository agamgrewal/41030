from lxrt.modeling import LXRTEncoder
from lxrt.tokenization import BertTokenizer

print("✅ Imports are working!")

# Create a dummy tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("Tokenizer loaded:", tokenizer.__class__.__name__)

# Create a dummy encoder
model = LXRTEncoder(model='bert-base-uncased')
print("Model loaded:", model.__class__.__name__)
