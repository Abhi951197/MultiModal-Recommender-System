"""
Streamlit UI for Multimodal Recommender System
Interactive web interface for text and image-based recommendations
"""

import streamlit as st
from PIL import Image
import pandas as pd
import json
import math
from recommender import MultimodalRecommender
from nlp_utils import parse_query_constraints, pos_tag_extract_keywords


# Page configuration
st.set_page_config(
    page_title="Multimodal Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .movie-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .music-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .book-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }
    .score-badge {
        background: rgba(255,255,255,0.3);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommender():
    """Load recommender system (cached)"""
    return MultimodalRecommender("models/unified_embeddings.pkl")

def display_recommendation(item, rank):
    """Display a single recommendation card"""
    type_icon = {
        'movie': '🎬',
        'music': '🎵',
        'book': '📚'
    }
    
    type_class = {
        'movie': 'movie-card',
        'music': 'music-card',
        'book': 'book-card'
    }
    
    icon = type_icon.get(item['type'], '📌')
    card_class = type_class.get(item['type'], 'recommendation-card')
    
    # Parse metadata
    metadata = item.get('metadata', {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except:
            metadata = {}
    
    # Build metadata display
    meta_parts = []
    if item['type'] == 'movie':
        genre = metadata.get('genre', 'Unknown')
        rating = metadata.get('rating', 0)
        meta_parts.append(f"Genre: {genre}")
        if rating > 0:
            meta_parts.append(f"Rating: ⭐ {rating:.1f}/10")
    elif item['type'] == 'music':
        artist = metadata.get('artist', 'Unknown')
        genre = metadata.get('genre', 'Unknown')
        meta_parts.append(f"Artist: {artist}")
        meta_parts.append(f"Genre: {genre}")
    elif item['type'] == 'book':
        author = metadata.get('author', 'Unknown')
        rating = metadata.get('rating', 0)
        meta_parts.append(f"Author: {author}")
        if rating > 0:
            meta_parts.append(f"Rating: ⭐ {rating:.1f}/5")
    
    meta_str = " | ".join(meta_parts)
    
    st.markdown(f"""
    <div class="recommendation-card {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div style="flex: 1;">
                <h3 style="margin: 0; font-size: 1.5rem;">{icon} {item['title']}</h3>
                <p style="margin: 0.5rem 0; opacity: 0.9; font-size: 0.9rem;">{meta_str}</p>
            </div>
            <span class="score-badge">Match: {item['similarity_score']*100:.1f}%</span>
        </div>
        <p style="margin-top: 1rem; line-height: 1.6;">{item['description'][:200]}{'...' if len(item['description']) > 200 else ''}</p>
    </div>
    """, unsafe_allow_html=True)

def detect_content_types(text_query: str | None, image_caption: str | None = None):
    """
    Infer which content types the user is asking for from the text query and optional image caption.

    Returns:
      - list of content type strings (subset of ['movie','music','book']) when confidently detected
      - None when unclear / should search all types
    """
    if not text_query and not image_caption:
        return None

    combined = " ".join(filter(None, [text_query or "", image_caption or ""]))
    s = combined.lower()

    # If user explicitly asks for everything
    if any(tok in s for tok in ["all", "everything", "any", "both", "every"]):
        return None

    movie_kw = ["movie", "movies", "film", "films", "cinema", "director", "actor", "actors", "actress", "trailer"]
    music_kw = ["music", "song", "songs", "album", "albums", "track", "tracks", "band", "artist", "lyrics", "playlist"]
    book_kw = ["book", "books", "novel", "novels", "author", "read", "reading", "chapter", "literature", "story"]

    found = set()
    for kw in movie_kw:
        if kw in s:
            found.add('movie')
            break
    for kw in music_kw:
        if kw in s:
            found.add('music')
            break
    for kw in book_kw:
        if kw in s:
            found.add('book')
            break

    # If nothing found, return None to search across all domains
    if not found:
        return None

    # Return a stable list in preferred order
    ordered = [d for d in ['movie', 'music', 'book'] if d in found]
    return ordered

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Multimodal Recommender System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover movies, music, and books through text and images</p>', unsafe_allow_html=True)
    
    # Load recommender
    try:
        recommender = load_recommender()
    except FileNotFoundError:
        st.error("⚠ Embeddings file not found! Please run train_model.py first to generate embeddings.")
        st.stop()
    except Exception as e:
        st.error(f"⚠ Error loading recommender: {e}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        # st.header("⚙ Settings")

      
        # Number of recommendations is inferred from the user's query.
        # If the query doesn't specify a number, we default to 3 recommendations.
        # st.markdown("**Number of recommendations:** inferred from your query (default: 3 if not specified)")
        num_recs = 3
        
        # st.markdown("---")
        
        # Dataset statistics (rendered with smaller font)
        st.markdown("<h4 style='margin:0;'>📊 Dataset Info</h4>", unsafe_allow_html=True)
        stats = recommender.get_statistics()
        # Use compact HTML blocks so we can control font-size
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style='text-align:center;'>
                <div style='font-size:14px;color:#666;margin-bottom:4px;'>Movies</div>
                <div style='font-size:18px;font-weight:bold'>{stats['movies']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style='text-align:center;'>
                <div style='font-size:14px;color:#666;margin-bottom:4px;'>Music</div>
                <div style='font-size:18px;font-weight:bold'>{stats['music']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style='text-align:center;'>
                <div style='font-size:14px;color:#666;margin-bottom:4px;'>Books</div>
                <div style='font-size:18px;font-weight:bold'>{stats['books']}</div>
            </div>
            """, unsafe_allow_html=True)
        # Total items smaller as well
        st.markdown(f"<div style='font-size:13px;color:#444;margin-top:8px;'>Total Items: <strong style='font-size:16px;'>{stats['total']}</strong></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Tips
        - Use descriptive text about mood, genre, or theme
        - Upload images that represent the vibe you want
        - Combine both for better results!
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Text Input")
        text_query = st.text_area(
            "Describe what you're looking for...",
            placeholder="e.g., 'dark thriller with mystery', 'uplifting and energetic music', 'romantic comedy'",
            height=150,
            help="Describe the mood, genre, theme, or feeling you want"
        )
    
    with col2:
        st.subheader("🖼 Image Input")
        uploaded_image = st.file_uploader(
            "Upload an image (optional)",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image that represents the vibe or mood you want"
        )
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image",  use_container_width=True)
    
    # Search button
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔍 Get Recommendations", use_container_width=True):
        if not text_query and not uploaded_image:
            st.warning("⚠ Please provide either text input or upload an image!")
        else:
            with st.spinner("🤔 Analyzing your preferences and finding perfect matches..."):
                try:
                    # Process image if uploaded
                    image_obj = None
                    if uploaded_image:
                        image_obj = Image.open(uploaded_image).convert('RGB')
                    # Try to generate an image caption (used for domain detection only)
                    image_caption = None
                    if image_obj:
                        try:
                            image_caption = recommender.generate_image_caption(image_obj)
                        except Exception:
                            # BLIP loading can fail in some environments; ignore and continue
                            image_caption = None

                    # Detect content types from query + image caption
                    detected_types = detect_content_types(text_query, image_caption)

                    # Parse constraints (number/genre/sort) from combined text+caption
                    combined_for_parse = " ".join(filter(None, [text_query or "", image_caption or ""]))
                    constraints = parse_query_constraints(combined_for_parse)
                    requested_num = constraints.get('number')

                    # If user requested a number in the query, prefer it over the slider (cap to avoid runaway values)
                    if requested_num is not None:
                        try:
                            requested_num = int(requested_num)
                            if requested_num < 1:
                                requested_num = None
                            else:
                                requested_num = min(requested_num, 50)
                                num_recs = requested_num
                        except Exception:
                            requested_num = None

                    # Use POS tagging to extract up to `effective_k` keywords from the combined query
                    effective_k = requested_num if requested_num is not None else num_recs
                    try:
                        keywords = pos_tag_extract_keywords(combined_for_parse, k=effective_k)
                    except Exception:
                        keywords = []

                    # Build a search text by combining original text input with the extracted keywords
                    keyword_text = " ".join(keywords) if keywords else ""
                    search_text = " ".join(filter(None, [text_query or "", keyword_text])).strip()

                    # Get recommendations based on detection
                    if detected_types is None:
                        # Unclear: search across all domains (use previous distribution logic)
                        if num_recs <= 6:
                            recs_per_domain = max(1, num_recs // 3)
                            results, combined_query = recommender.get_recommendations_by_domain(
                                text_query=search_text if search_text else None,
                                image=image_obj,
                                top_k_per_domain=recs_per_domain
                            )
                            recommendations = []
                            for domain in ['movie', 'music', 'book']:
                                recommendations.extend(results[domain])
                        else:
                            recommendations, combined_query = recommender.get_recommendations(
                                text_query=search_text if search_text else None,
                                image=image_obj,
                                top_k=num_recs,
                                content_type=None
                            )
                    else:
                        # One or more specific domains detected
                        combined_query = None
                        per_domain_k = math.ceil(num_recs / len(detected_types))
                        all_recs = []
                        for domain in detected_types:
                            recs, q = recommender.get_recommendations(
                                text_query=search_text if search_text else None,
                                image=image_obj,
                                top_k=per_domain_k,
                                content_type=domain
                            )
                            # record combined_query from first call
                            if combined_query is None:
                                combined_query = q
                            all_recs.extend(recs)

                        # Deduplicate by title and sort by similarity, then take top num_recs
                        seen = set()
                        unique = []
                        for r in sorted(all_recs, key=lambda x: x['similarity_score'], reverse=True):
                            title = r.get('title') or r.get('id')
                            if title in seen:
                                continue
                            seen.add(title)
                            unique.append(r)
                        recommendations = unique[:num_recs]
                    
                    # Display results
                    st.success(f"✨ Found {len(recommendations)} perfect matches for you!")

                    # Show interpreted query and detected domains
                    with st.expander("🧠 How we interpreted your request", expanded=False):
                        st.info(f"*Combined Query:* {combined_query}")
                        detected_display = detected_types if detected_types is not None else ['movie', 'music', 'book']
                        st.write(f"*Detected content types:* {', '.join(detected_display)}")
                        try:
                            st.write(f"**Extracted keywords (k={effective_k}):** {', '.join(keywords) if keywords else '—'}")
                        except Exception:
                            # keywords might not be defined in some error branches
                            pass
                    
                    st.markdown("---")
                    st.subheader("🎯 Your Recommendations")
                    
                    # Display recommendations
                    for idx, item in enumerate(recommendations, 1):
                        display_recommendation(item, idx)
                    
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()