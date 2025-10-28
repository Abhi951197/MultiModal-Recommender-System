# Run this on a machine that has internet access
# Requires: pip install huggingface_hub sentence-transformers transformers

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import os

# Replace these IDs if your project uses different model ids
SBERT_ID = "sentence-transformers/all-MiniLM-L6-v2"
BLIP_ID = "Salesforce/blip-image-captioning-base"

out_dir = "models_offline"
os.makedirs(out_dir, exist_ok=True)

print("Downloading SBERT...")
# Option A: snapshot_download saves the HF repo files to a directory
sbert_dir = os.path.join(out_dir, "sbert-all-MiniLM-L6-v2")
snapshot_download(repo_id=SBERT_ID, cache_dir=sbert_dir, repo_type="model", allow_patterns="*")

# Some SentenceTransformer objects store data in a particular structure; we also save via API
print("Saving SentenceTransformer object (optional)...")
sbert_model = SentenceTransformer(SBERT_ID)
sbert_model.save(os.path.join(out_dir, "sbert_local_saved"))

print("Downloading BLIP (image captioning)...")
blip_dir = os.path.join(out_dir, "blip-image-captioning-base")
snapshot_download(repo_id=BLIP_ID, cache_dir=blip_dir, repo_type="model", allow_patterns="*")

print("Done. Folders created under:", out_dir)