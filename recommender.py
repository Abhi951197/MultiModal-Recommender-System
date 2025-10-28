"""
Core Recommender Logic for Multimodal Recommender System
Handles image captioning, text processing, and similarity-based recommendations
"""

import os
import numpy as np
import pickle
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from sklearn.metrics.pairwise import cosine_similarity
import json

class MultimodalRecommender:
    """Multimodal recommender combining text and image inputs"""
    
    def __init__(self, embeddings_path, sbert_model='all-MiniLM-L6-v2', models_offline_dir: str | None = None):
        """
        Initialize the recommender system
        
        Args:
            embeddings_path: Path to precomputed embeddings file
            sbert_model: Sentence-BERT model name
        """
        print("Loading recommender system...")
        
        # Load precomputed embeddings and dataset
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        self.items = data['items']
        self.embeddings = data['embeddings']
        
        print(f"Loaded {len(self.items)} items")
        
        # Determine local models directory (if present)
        self.models_offline_dir = models_offline_dir or os.environ.get('MODELS_OFFLINE_DIR', 'models_offline')

        # Helper: choose SBERT source (local path if available, else model id)
        def _select_sbert_source(requested_model: str):
            # If env override provided, prefer it
            env_path = os.environ.get('SBERT_LOCAL_PATH')
            if env_path and os.path.isdir(env_path):
                print(f"Using SBERT from SBERT_LOCAL_PATH={env_path}")
                return env_path

            # Check common local locations under models_offline_dir
            candidates = [
                os.path.join(self.models_offline_dir, 'sbert_local_saved'),
                os.path.join(self.models_offline_dir, f'sbert-{requested_model}'),
                os.path.join(self.models_offline_dir, f'sbert_{requested_model}'),
                os.path.join(self.models_offline_dir, requested_model),
            ]
            for c in candidates:
                if os.path.isdir(c):
                    print(f"Using SBERT from local folder: {c}")
                    return c

            # Fallback to the requested model id (will download if not present locally)
            print(f"Using SBERT model id: {requested_model} (online) or cached path")
            return requested_model

        sbert_source = _select_sbert_source(sbert_model)

        # Initialize Sentence-BERT for query encoding (local path or HF id)
        try:
            self.sbert_model = SentenceTransformer(sbert_source)
        except Exception as e:
            print(f"Warning: failed to load SBERT from {sbert_source}: {e}\nFalling back to model id '{sbert_model}'.")
            self.sbert_model = SentenceTransformer(sbert_model)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sbert_model.to(self.device)
        
        # Initialize BLIP for image captioning (lazy loading)
        self.blip_processor = None
        self.blip_model = None
        # Determine BLIP local path candidate
        self._blip_local_path = None
        env_blip = os.environ.get('BLIP_LOCAL_PATH')
        if env_blip and os.path.isdir(env_blip):
            self._blip_local_path = env_blip
        else:
            candidate = os.path.join(self.models_offline_dir, 'blip-image-captioning-base')
            if os.path.isdir(candidate):
                # Attempt to find a proper subfolder that contains the required files
                self._blip_local_path = candidate

        # End of __init__

    def _find_valid_model_folder(self, root_dir: str, required_files=None):
        """Search nested folders for a valid HF model folder containing required files.

        Returns the first matching folder path, or None if not found.
        """
        if not root_dir or not os.path.isdir(root_dir):
            return None
        if required_files is None:
            required_files = ['preprocessor_config.json', 'config.json']
        for dirpath, dirnames, filenames in os.walk(root_dir):
            lower_files = [f.lower() for f in filenames]
            # Check for any required file or model weights
            if any(rf.lower() in lower_files for rf in required_files):
                # also ensure there's a model weight file
                if any(x in lower_files for x in ['pytorch_model.bin', 'pytorch_model.safetensors', 'tf_model.h5']):
                    return dirpath
                # sometimes weights are in parent; accept folder if config present
                return dirpath
        return None
        
        print(f"Recommender initialized on device: {self.device}")
    
    def load_blip_model(self):
        """Lazy load BLIP model for image captioning"""
        if self.blip_processor is None:
            print("Loading BLIP model for image captioning...")
            model_name = "Salesforce/blip-image-captioning-base"
            # If local path exists, load from there with local_files_only
            if self._blip_local_path:
                try:
                    print(f"Attempting to load BLIP from local path: {self._blip_local_path}")
                    # If the candidate path doesn't directly contain the required files, search nested folders
                    valid_folder = self._find_valid_model_folder(self._blip_local_path)
                    load_path = valid_folder or self._blip_local_path
                    if valid_folder and valid_folder != self._blip_local_path:
                        print(f"Found nested valid BLIP model folder: {valid_folder}")

                    self.blip_processor = BlipProcessor.from_pretrained(load_path, local_files_only=True)
                    self.blip_model = BlipForConditionalGeneration.from_pretrained(load_path, local_files_only=True)
                    self.blip_model.to(self.device)
                    print("BLIP model loaded from local path successfully")
                    return
                except Exception as e:
                    print(f"Failed to load BLIP from local path {self._blip_local_path}: {e}\nFalling back to HF model id '{model_name}'.")

            # Fallback to model id (may attempt to download if not cached)
            try:
                self.blip_processor = BlipProcessor.from_pretrained(model_name)
                self.blip_model = BlipForConditionalGeneration.from_pretrained(model_name)
                self.blip_model.to(self.device)
                print("BLIP model loaded successfully from HF name (or cache)")
            except Exception as e:
                print(f"Error loading BLIP model '{model_name}': {e}")
                raise
    
    def generate_image_caption(self, image):
        """
        Generate caption from image using BLIP
        
        Args:
            image: PIL Image object
            
        Returns:
            Generated caption string
        """
        self.load_blip_model()
        
        # Process image
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        
        # Generate caption
        with torch.no_grad():
            output = self.blip_model.generate(**inputs, max_length=50, num_beams=5)
        
        caption = self.blip_processor.decode(output[0], skip_special_tokens=True)
        
        return caption
    
    def encode_query(self, text):
        """
        Encode text query into embedding
        
        Args:
            text: Query text string
            
        Returns:
            Normalized embedding vector
        """
        embedding = self.sbert_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding[0]
    
    def get_recommendations(self, text_query=None, image=None, top_k=5, content_type=None):
        """
        Get recommendations based on text and/or image input
        
        Args:
            text_query: Text description from user (optional)
            image: PIL Image object (optional)
            top_k: Number of recommendations to return
            content_type: Filter by type ('movie', 'music', 'book', or None for all)
            
        Returns:
            List of recommended items with similarity scores
        """
        # Build combined query
        query_parts = []
        
        if text_query:
            query_parts.append(text_query)
        
        if image is not None:
            caption = self.generate_image_caption(image)
            query_parts.append(f"Image shows: {caption}")
        
        if not query_parts:
            raise ValueError("At least one of text_query or image must be provided")
        
        # Combine all query parts
        combined_query = " ".join(query_parts)
        
        # Encode query
        query_embedding = self.encode_query(combined_query)
        
        # Calculate similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        )[0]
        
        # Filter by content type if specified
        if content_type:
            valid_indices = [i for i, item in enumerate(self.items) 
                           if item['type'] == content_type]
            filtered_similarities = [(i, similarities[i]) for i in valid_indices]
        else:
            filtered_similarities = list(enumerate(similarities))
        
        # Sort by similarity
        filtered_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k recommendations
        recommendations = []
        for idx, score in filtered_similarities[:top_k]:
            item = self.items[idx].copy()
            item['similarity_score'] = float(score)
            
            # Parse metadata
            if 'metadata' in item and isinstance(item['metadata'], str):
                try:
                    item['metadata'] = json.loads(item['metadata'])
                except:
                    item['metadata'] = {}
            
            recommendations.append(item)
        
        return recommendations, combined_query
    
    def get_recommendations_by_domain(self, text_query=None, image=None, top_k_per_domain=2):
        """
        Get recommendations across all domains
        
        Args:
            text_query: Text description from user (optional)
            image: PIL Image object (optional)
            top_k_per_domain: Number of recommendations per domain
            
        Returns:
            Dictionary with recommendations by domain
        """
        results = {}
        
        for domain in ['movie', 'music', 'book']:
            recs, query = self.get_recommendations(
                text_query=text_query,
                image=image,
                top_k=top_k_per_domain,
                content_type=domain
            )
            results[domain] = recs
        
        return results, query
    
    def get_statistics(self):
        """Get dataset statistics"""
        stats = {
            'total': len(self.items),
            'movies': sum(1 for item in self.items if item['type'] == 'movie'),
            'music': sum(1 for item in self.items if item['type'] == 'music'),
            'books': sum(1 for item in self.items if item['type'] == 'book')
        }
        return stats

# Example usage
if __name__ == "__main__":
    # Initialize recommender
    recommender = MultimodalRecommender("models/unified_embeddings.pkl")
    
    # Example 1: Text-only query
    print("\n" + "="*60)
    print("Example 1: Text-only recommendation")
    print("="*60)
    query = "dark thriller with mystery and suspense"
    recs, combined_query = recommender.get_recommendations(text_query=query, top_k=3)
    print(f"\nQuery: {query}")
    print(f"\nTop 3 recommendations:")
    for i, rec in enumerate(recs, 1):
        print(f"\n{i}. [{rec['type'].upper()}] {rec['title']}")
        print(f"   Score: {rec['similarity_score']:.4f}")
        print(f"   Description: {rec['description'][:100]}...")
    
    # Example 2: Cross-domain recommendations
    print("\n" + "="*60)
    print("Example 2: Cross-domain recommendations")
    print("="*60)
    query = "uplifting and inspirational content"
    results, combined_query = recommender.get_recommendations_by_domain(
        text_query=query,
        top_k_per_domain=2
    )
    print(f"\nQuery: {query}\n")
    for domain, recs in results.items():
        print(f"\n{domain.upper()}:")
        for rec in recs:
            print(f"  • {rec['title']} (score: {rec['similarity_score']:.4f})")
    
    # Display statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    stats = recommender.get_statistics()
    for key, value in stats.items():
        print(f"{key.capitalize()}: {value}")