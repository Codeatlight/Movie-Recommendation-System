import streamlit as st
from PIL import Image
import json
import re
import os
from Classifier import KNearestNeighbours
from bs4 import BeautifulSoup
import requests, io
import PIL.Image
from urllib.request import urlopen
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

# Page configuration
st.set_page_config(
    page_title="CinemaScope - Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    with open('./Data/movie_data.json', 'r+', encoding='utf-8') as f:
        data = json.load(f)
    with open('./Data/movie_titles.json', 'r+', encoding='utf-8') as f:
        movie_titles = json.load(f)
    return data, movie_titles

data, movie_titles = load_data()
hdr = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

# OMDB API configuration (optional - can be set via environment variable or Streamlit secrets)
OMDB_API_KEY = None
try:
    # Try to get from Streamlit secrets first
    if hasattr(st, 'secrets') and 'omdb_api_key' in st.secrets:
        OMDB_API_KEY = st.secrets['omdb_api_key']
    # Try environment variable
    import os
    if not OMDB_API_KEY:
        OMDB_API_KEY = os.getenv('OMDB_API_KEY')
except:
    pass

def extract_imdb_id(imdb_link):
    """Extract IMDB ID from IMDB URL"""
    try:
        # Pattern: tt followed by 7-8 digits
        match = re.search(r'tt\d{7,8}', imdb_link)
        if match:
            return match.group(0)
    except:
        pass
    return None

def fetch_from_omdb(imdb_id=None, movie_title=None):
    """Fetch movie data from OMDB API"""
    if not OMDB_API_KEY:
        return None
    
    try:
        base_url = "http://www.omdbapi.com/"
        params = {"apikey": OMDB_API_KEY}
        
        # Prefer IMDB ID over title (more accurate)
        if imdb_id:
            params["i"] = imdb_id
            response = requests.get(base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("Response") == "True":
                    return data
        
        # Fallback to title search if IMDB ID didn't work
        if movie_title:
            # Clean title - remove year if present in parentheses
            clean_title = re.sub(r'\s*\(\d{4}\)\s*$', '', movie_title).strip()
            params = {"apikey": OMDB_API_KEY, "t": clean_title}
            response = requests.get(base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("Response") == "True":
                    return data
    except Exception as e:
        pass
    return None

# Advanced Custom CSS with Dark Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Playfair+Display:wght@700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main App Background - Dark Cinematic Theme */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, rgba(15, 12, 41, 0.95) 0%, rgba(48, 43, 99, 0.95) 100%);
        padding: 3rem 2rem;
        border-radius: 25px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
        text-align: center;
        border: 2px solid rgba(255, 215, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 215, 0, 0.1), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .header-title {
        font-family: 'Playfair Display', serif;
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 50%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.5);
        letter-spacing: 2px;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.3rem;
        font-weight: 400;
        margin-top: 1rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    }
    
    /* Movie Cards - Dark with Gold Accents */
    .movie-card {
        background: linear-gradient(135deg, rgba(20, 20, 40, 0.95) 0%, rgba(30, 30, 60, 0.95) 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        border-left: 5px solid #ffd700;
        border: 2px solid rgba(255, 215, 0, 0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .movie-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #ffd700, #ff6b6b, #ffd700);
        transform: scaleX(0);
        transition: transform 0.4s ease;
    }
    
    .movie-card:hover {
        transform: translateY(-8px) translateX(5px);
        box-shadow: 0 20px 60px rgba(255, 215, 0, 0.3), 0 0 0 1px rgba(255, 215, 0, 0.5);
        border-color: rgba(255, 215, 0, 0.6);
    }
    
    .movie-card:hover::before {
        transform: scaleX(1);
    }
    
    .movie-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffd700;
        margin-bottom: 1rem;
        text-shadow: 0 2px 10px rgba(255, 215, 0, 0.5);
    }
    
    .movie-info {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1rem;
        line-height: 1.8;
        margin: 0.8rem 0;
    }
    
    .rating-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 0.6rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 1rem;
        box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffd700;
        margin: 2.5rem 0 1.5rem 0;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
        text-align: center;
        letter-spacing: 1px;
    }
    
    /* Input Styling - Dark Theme */
    .stSelectbox > div > div {
        background-color: rgba(20, 20, 40, 0.9) !important;
        border: 2px solid rgba(255, 215, 0, 0.3) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    
    .stSelectbox label {
        color: #ffd700 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    .stSlider > div > div {
        background-color: rgba(20, 20, 40, 0.9) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        border: 2px solid rgba(255, 215, 0, 0.3) !important;
    }
    
    .stSlider label {
        color: #ffd700 !important;
        font-weight: 600 !important;
    }
    
    .stNumberInput > div > div > input {
        background-color: rgba(20, 20, 40, 0.9) !important;
        border: 2px solid rgba(255, 215, 0, 0.3) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    
    .stNumberInput label {
        color: #ffd700 !important;
        font-weight: 600 !important;
    }
    
    .stMultiSelect > div > div {
        background-color: rgba(20, 20, 40, 0.9) !important;
        border: 2px solid rgba(255, 215, 0, 0.3) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    
    .stMultiSelect label {
        color: #ffd700 !important;
        font-weight: 600 !important;
    }
    
    .stRadio > div {
        background-color: rgba(20, 20, 40, 0.9) !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        border: 2px solid rgba(255, 215, 0, 0.3) !important;
    }
    
    .stRadio label {
        color: #ffd700 !important;
        font-weight: 600 !important;
    }
    
    /* Buttons - Gold Gradient */
    .stButton > button {
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 50%, #ff6b6b 100%) !important;
        color: #0f0c29 !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 0.9rem 2.5rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 20px rgba(255, 215, 0, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(255, 215, 0, 0.6) !important;
        background: linear-gradient(135deg, #ffed4e 0%, #ffd700 50%, #ffed4e 100%) !important;
    }
    
    /* Info Boxes */
    .info-box {
        background: rgba(20, 20, 40, 0.9) !important;
        padding: 1.5rem !important;
        border-radius: 15px !important;
        border-left: 5px solid #ffd700 !important;
        margin: 1.5rem 0 !important;
        border: 2px solid rgba(255, 215, 0, 0.3) !important;
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    .info-box h4 {
        color: #ffd700 !important;
        font-weight: 700 !important;
    }
    
    /* Genre Tags */
    .genre-tag {
        display: inline-block;
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
        color: #0f0c29;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 0.3rem;
        font-weight: 600;
        box-shadow: 0 3px 10px rgba(255, 215, 0, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Stats Container */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 3rem 0;
        flex-wrap: wrap;
        gap: 1.5rem;
    }
    
    .stat-box {
        background: linear-gradient(135deg, rgba(20, 20, 40, 0.95) 0%, rgba(30, 30, 60, 0.95) 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        min-width: 180px;
        flex: 1;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        border: 2px solid rgba(255, 215, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .stat-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(255, 215, 0, 0.3);
        border-color: rgba(255, 215, 0, 0.6);
    }
    
    .stat-number {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1rem;
        margin-top: 0.8rem;
        font-weight: 500;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 100%);
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        background: transparent;
    }
    
    /* Poster Image Styling */
    .poster-container {
        text-align: center;
        margin: 1rem 0;
    }
    
    .poster-image {
        border-radius: 15px;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6), 0 0 0 3px rgba(255, 215, 0, 0.3);
        transition: transform 0.4s ease;
    }
    
    .poster-image:hover {
        transform: scale(1.08) rotate(2deg);
        box-shadow: 0 20px 50px rgba(255, 215, 0, 0.5), 0 0 0 3px rgba(255, 215, 0, 0.6);
    }
    
    /* Warning/Info Messages */
    .stAlert {
        background-color: rgba(20, 20, 40, 0.95) !important;
        border: 2px solid rgba(255, 215, 0, 0.5) !important;
        border-radius: 12px !important;
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Text Colors */
    h1, h2, h3, h4, h5, h6 {
        color: #ffd700 !important;
    }
    
    /* Link Styling */
    a {
        color: #ffd700 !important;
        text-decoration: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    a:hover {
        color: #ffed4e !important;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.5) !important;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(20, 20, 40, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #ffd700, #ff6b6b);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #ffed4e, #ffd700);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=1800, show_spinner=False)  # Cache for 30 minutes
def movie_poster_fetcher(imdb_link, movie_title=None):
    """Fetch and display movie poster from IMDB with OMDB API fallback"""
    # First, try OMDB API (more reliable)
    imdb_id = extract_imdb_id(imdb_link)
    if imdb_id or movie_title:
        omdb_data = fetch_from_omdb(imdb_id=imdb_id, movie_title=movie_title)
        if omdb_data and omdb_data.get("Poster") and omdb_data["Poster"] != "N/A":
            try:
                poster_url = omdb_data["Poster"]
                u = urlopen(poster_url)
                raw_data = u.read()
                image = PIL.Image.open(io.BytesIO(raw_data))
                image = image.resize((220, 330))
                return image
            except:
                pass
    
    # Fallback to IMDB scraping
    try:
        url_data = requests.get(imdb_link, headers=hdr, timeout=15).text
        s_data = BeautifulSoup(url_data, 'html.parser')
        
        # Method 1: Try JSON-LD structured data (most reliable)
        json_ld = s_data.find("script", type="application/ld+json")
        if json_ld:
            try:
                data = json.loads(json_ld.string)
                if isinstance(data, dict) and 'image' in data:
                    poster_url = data['image']
                    if isinstance(poster_url, str) and poster_url.startswith('http'):
                        u = urlopen(poster_url)
                        raw_data = u.read()
                        image = PIL.Image.open(io.BytesIO(raw_data))
                        image = image.resize((220, 330))
                        return image
            except:
                pass
        
        # Method 2: Try Open Graph image
        og_image = s_data.find("meta", property="og:image")
        if og_image and 'content' in og_image.attrs:
            poster_url = og_image['content']
            if poster_url.startswith('http'):
                try:
                    u = urlopen(poster_url)
                    raw_data = u.read()
                    image = PIL.Image.open(io.BytesIO(raw_data))
                    image = image.resize((220, 330))
                    return image
                except:
                    pass
        
        # Method 3: Try the new IMDB structure with various selectors
        selectors = [
            ("div", {"class": "ipc-media--poster-27x40"}),
            ("div", {"class": "ipc-media--poster"}),
            ("div", {"class": "poster"}),
            ("img", {"class": "ipc-image"}),
            ("img", {"data-testid": "hero-poster"}),
        ]
        
        for tag, attrs in selectors:
            element = s_data.find(tag, attrs)
            if element:
                img_tag = element.find("img") if element.name != "img" else element
                if img_tag:
                    # Try multiple attributes
                    for attr in ['src', 'data-src', 'srcset', 'data-image-url']:
                        if attr in img_tag.attrs:
                            poster_url = img_tag[attr]
                            if attr == 'srcset':
                                poster_url = poster_url.split(',')[0].split()[0]
                            
                            if poster_url and poster_url.startswith('http'):
                                try:
                                    # Clean up URL
                                    if '._V1_' in poster_url:
                                        # Try to get higher resolution
                                        poster_url = poster_url.split('._V1_')[0] + '._V1_SX300.jpg'
                                    u = urlopen(poster_url)
                                    raw_data = u.read()
                                    image = PIL.Image.open(io.BytesIO(raw_data))
                                    image = image.resize((220, 330))
                                    return image
                                except:
                                    continue
        
        # Method 4: Search all images for poster-like URLs
        all_imgs = s_data.find_all("img")
        for img in all_imgs:
            src = img.get('src') or img.get('data-src') or img.get('data-image-url')
            if src and ('poster' in src.lower() or 'images' in src.lower()):
                if src.startswith('http') or src.startswith('//'):
                    if not src.startswith('http'):
                        src = 'https:' + src
                    try:
                        u = urlopen(src)
                        raw_data = u.read()
                        image = PIL.Image.open(io.BytesIO(raw_data))
                        image = image.resize((220, 330))
                        return image
                    except:
                        continue
        
        return None
    except Exception as e:
        # Try one more time with a simpler approach - extract from page source
        try:
            # Look for image URLs in the raw HTML
            if 'images' in url_data.lower() or 'poster' in url_data.lower():
                # Find image URLs
                img_pattern = r'https?://[^"\s]+\.(?:jpg|jpeg|png|webp)[^"\s]*'
                matches = re.findall(img_pattern, url_data)
                for img_url in matches[:5]:  # Try first 5 matches
                    if 'poster' in img_url.lower() or 'images' in img_url.lower():
                        try:
                            u = urlopen(img_url)
                            raw_data = u.read()
                            image = PIL.Image.open(io.BytesIO(raw_data))
                            image = image.resize((220, 330))
                            return image
                        except:
                            continue
        except:
            pass
        return None

@st.cache_data(ttl=1800, show_spinner=False)  # Cache for 30 minutes  
def get_movie_info(imdb_link, movie_title=None):
    """Extract movie information from IMDB with OMDB API fallback"""
    # First, try OMDB API (more reliable)
    imdb_id = extract_imdb_id(imdb_link)
    if imdb_id or movie_title:
        omdb_data = fetch_from_omdb(imdb_id=imdb_id, movie_title=movie_title)
        if omdb_data:
            title = omdb_data.get("Title", "")
            plot = omdb_data.get("Plot", "")
            actors = omdb_data.get("Actors", "")
            
            # If we got data from OMDB, return it
            if plot and plot != "N/A":
                return title, actors, plot, ""
            elif title:
                # At least we have a title
                return title, actors, "", ""
    
    # Fallback to IMDB scraping
    try:
        url_data = requests.get(imdb_link, headers=hdr, timeout=15).text
        s_data = BeautifulSoup(url_data, 'html.parser')
        
        # Initialize return values
        title = ""
        cast = ""
        story = ""
        
        # Method 1: Try JSON-LD structured data (most reliable)
        json_ld = s_data.find("script", type="application/ld+json")
        if json_ld:
            try:
                data = json.loads(json_ld.string)
                if isinstance(data, dict):
                    if 'name' in data:
                        title = data['name']
                    if 'description' in data:
                        story = data['description']
                    if 'actor' in data:
                        actors = data['actor']
                        if isinstance(actors, list):
                            cast_names = [actor.get('name', '') for actor in actors[:5] if isinstance(actor, dict)]
                            cast = ", ".join([name for name in cast_names if name])
            except:
                pass
        
        # Method 2: Try meta description (very reliable for basic info)
        imdb_content = s_data.find("meta", {"name": "description"})
        if imdb_content and 'content' in imdb_content.attrs:
            movie_descr = imdb_content.attrs['content']
            if movie_descr and len(movie_descr) > 10:
                # Parse the description - usually format: "Title. Cast info. Story info."
                parts = movie_descr.split(".")
                if not title and len(parts) >= 1:
                    title = parts[0].strip()
                if not cast and len(parts) >= 2:
                    cast = parts[1].strip()
                if not story and len(parts) >= 3:
                    story = ".".join(parts[2:]).strip()
        
        # Method 3: Extract plot summary from page with multiple selectors
        if not story:
            plot_selectors = [
                ("div", {"data-testid": "plot"}),
                ("span", {"data-testid": "plot-xl"}),
                ("span", {"data-testid": "plot-l"}),
                ("div", {"class": "plot_summary"}),
                ("div", {"class": "summary_text"}),
                ("p", {"data-testid": "plot"}),
            ]
            
            for tag, attrs in plot_selectors:
                plot_div = s_data.find(tag, attrs)
                if plot_div:
                    story = plot_div.get_text(strip=True)
                    story = " ".join(story.split())
                    if story and len(story) > 20:  # Valid plot
                        break
        
        # Method 4: Extract cast information with multiple methods
        if not cast:
            # Try structured data first
            cast_section = s_data.find("div", {"data-testid": "title-cast"})
            if not cast_section:
                cast_section = s_data.find("table", class_="cast_list")
            if not cast_section:
                cast_section = s_data.find("div", class_="cast_list")
            
            if cast_section:
                cast_links = cast_section.find_all("a", href=True)
                cast_names = []
                for link in cast_links[:8]:  # Get more names
                    name = link.get_text(strip=True)
                    if name and name not in ["See full cast", "See full cast & crew", "See more"]:
                        # Check if it's a valid name (not a link text)
                        if len(name) > 2 and not name.startswith("http"):
                            cast_names.append(name)
                if cast_names:
                    cast = ", ".join(cast_names[:5])  # Limit to 5
        
        # Method 5: Extract title with multiple selectors
        if not title:
            title_selectors = [
                ("h1", {"data-testid": "hero-title-block__title"}),
                ("h1", {"class": "title_wrapper"}),
                ("title", {}),
                ("meta", {"property": "og:title"}),
            ]
            
            for tag, attrs in title_selectors:
                title_elem = s_data.find(tag, attrs)
                if title_elem:
                    if tag == "meta" and 'content' in title_elem.attrs:
                        title = title_elem['content']
                    else:
                        title = title_elem.get_text(strip=True)
                    if title:
                        # Clean up title
                        title = title.split(" - IMDb")[0].split(" | ")[0].split(" (")[0].strip()
                        break
        
        # Clean and return (without prefixes, we'll add them in display)
        title = title if title else ""
        cast = cast if cast else ""
        story = story if story else ""
        
        # Final fallback: Try to extract from raw HTML if nothing found
        if not story and not cast:
            try:
                # Look for common patterns in the HTML
                page_text = s_data.get_text()
                
                # Try to find plot in common locations
                plot_keywords = ['plot', 'summary', 'synopsis', 'story']
                for keyword in plot_keywords:
                    # Look for text near these keywords
                    idx = page_text.lower().find(keyword)
                    if idx > 0:
                        # Extract surrounding text
                        snippet = page_text[max(0, idx-50):idx+200]
                        if len(snippet) > 30:
                            story = snippet.strip()
                            break
            except:
                pass
        
        return title, cast, story, ""
        
    except Exception as e:
        # Last resort: return empty but don't fail
        return "", "", "", ""

def KNN_Movie_Recommender(test_point, k):
    """Generate movie recommendations using KNN algorithm"""
    target = [0 for item in movie_titles]
    model = KNearestNeighbours(data, target, test_point, k=k)
    model.fit()
    table = []
    for i in model.indices:
        table.append([movie_titles[i][0], movie_titles[i][2], data[i][-1]])
    return table

def clean_text(text):
    """Clean and prepare text for display"""
    if not text:
        return ""
    # Remove extra whitespace and newlines
    text = " ".join(str(text).split())
    # Remove common prefixes
    text = text.replace("Cast:", "").replace("Story:", "").strip()
    return text

def display_movie_card(movie, link, ratings, index, show_poster=False):
    """Display a beautifully formatted movie card"""
    # Fetch movie information and poster (pass movie title for OMDB fallback)
    title_info, cast_info, story_info, total_rat = get_movie_info(link, movie_title=movie)
    poster = movie_poster_fetcher(link, movie_title=movie) if show_poster else None
    
    # Clean and prepare content - be more lenient with what we accept
    title_text = clean_text(title_info) if title_info and len(title_info.strip()) > 0 else ""
    cast_text = clean_text(cast_info) if cast_info and len(cast_info.strip()) > 0 else ""
    story_text = clean_text(story_info) if story_info and len(story_info.strip()) > 0 else ""
    
    # Build content sections with proper labels
    content_sections = []
    
    # Add plot/story first (most important)
    if story_text and len(story_text) > 10:
        # Truncate if too long
        if len(story_text) > 300:
            story_text = story_text[:300] + "..."
        content_sections.append(("Plot", story_text))
    
    # Add cast information
    if cast_text and len(cast_text) > 3:
        content_sections.append(("Cast", cast_text))
    
    # Add title/about info if we have it
    if title_text and len(title_text) > 3 and title_text.lower() != movie.lower():
        content_sections.append(("About", title_text))
    
    # If no information was found, try to provide at least something
    if not content_sections:
        # Try to extract year or basic info from the link
        content_sections.append(("Info", f"IMDB Rating: {ratings:.1f}/10. Click the link below for full details."))
    
    # Create HTML for content - properly escape and format
    content_html = ""
    for label, text in content_sections:
        if text:
            # Escape HTML special characters
            text_escaped = (str(text)
                          .replace("&", "&amp;")
                          .replace("<", "&lt;")
                          .replace(">", "&gt;")
                          .replace('"', "&quot;")
                          .replace("'", "&#39;"))
            # Build the HTML div
            content_html += f'<div class="movie-info"><strong style="color: #ffd700;">{label}:</strong> {text_escaped}</div>'
    
    # Fix: Handle columns properly based on whether we have a poster
    if show_poster and poster:
        col1, col2 = st.columns([1, 2.5])
        with col1:
            st.image(poster, width='stretch')
        with col2:
            st.markdown(f"""
            <div class="movie-card">
                <div class="movie-title">#{index}. {movie}</div>
                {content_html}
                <div class="rating-badge">⭐ IMDB Rating: {ratings:.1f}</div>
                <a href="{link}" target="_blank" style="color: #ffd700; text-decoration: none; font-weight: 600; margin-top: 1rem; display: inline-block;">View on IMDB →</a>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Show placeholder if poster was requested but not found
        if show_poster:
            col1, col2 = st.columns([1, 2.5])
            with col1:
                st.markdown("""
                <div style="background: rgba(20, 20, 40, 0.7); padding: 3rem 1rem; border-radius: 15px; text-align: center; border: 2px dashed rgba(255, 215, 0, 0.3);">
                    <p style="color: rgba(255, 215, 0, 0.6); font-size: 3rem;">🎬</p>
                    <p style="color: rgba(255, 255, 255, 0.6); font-size: 0.9rem;">Poster not available</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">#{index}. {movie}</div>
                    {content_html}
                    <div class="rating-badge">⭐ IMDB Rating: {ratings:.1f}</div>
                    <a href="{link}" target="_blank" style="color: #ffd700; text-decoration: none; font-weight: 600; margin-top: 1rem; display: inline-block;">View on IMDB →</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="movie-card">
                <div class="movie-title">#{index}. {movie}</div>
                {content_html}
                <div class="rating-badge">⭐ IMDB Rating: {ratings:.1f}</div>
                <a href="{link}" target="_blank" style="color: #ffd700; text-decoration: none; font-weight: 600; margin-top: 1rem; display: inline-block;">View on IMDB →</a>
            </div>
            """, unsafe_allow_html=True)

def main():
    # Sidebar with enhanced design
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0; border-bottom: 2px solid rgba(255, 215, 0, 0.3);">
            <h1 style="color: #ffd700; font-size: 2.5rem; margin-bottom: 0.5rem; font-family: 'Playfair Display', serif; text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);">🎬</h1>
            <h2 style="color: #ffd700; font-size: 1.8rem; margin-bottom: 0.5rem; font-family: 'Playfair Display', serif;">CinemaScope</h2>
            <p style="color: rgba(255,255,255,0.8); font-size: 1rem;">Intelligent Movie Recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Dataset Information")
        st.markdown(f"""
        <div style="background: rgba(20, 20, 40, 0.7); padding: 1.5rem; border-radius: 15px; border: 2px solid rgba(255, 215, 0, 0.3);">
            <p style="color: #ffd700; font-weight: 700; font-size: 1.3rem; margin-bottom: 0.5rem;">{len(movie_titles):,}</p>
            <p style="color: rgba(255,255,255,0.9); margin-bottom: 1rem;">Total Movies Available</p>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">📚 Data Source: IMDB 5000 Movie Dataset</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🎯 Key Features")
        st.markdown("""
        <div style="background: rgba(20, 20, 40, 0.7); padding: 1.5rem; border-radius: 15px; border: 2px solid rgba(255, 215, 0, 0.3);">
            <p style="color: rgba(255,255,255,0.9); line-height: 2;">
                🎬 Movie-based recommendations<br>
                🎭 Genre-based recommendations<br>
                🖼️ High-quality movie posters<br>
                ⭐ Real-time IMDB ratings<br>
                📝 Detailed movie information<br>
                🤖 AI-powered KNN algorithm
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # OMDB API Key info (optional)
        if OMDB_API_KEY:
            st.markdown("""
            <div style="background: rgba(0, 150, 0, 0.2); padding: 1rem; border-radius: 10px; border: 2px solid rgba(0, 255, 0, 0.3); margin-top: 1rem;">
                <p style="color: rgba(0, 255, 0, 0.9); font-size: 0.9rem; margin: 0;">
                    ✅ OMDB API: Active<br>
                    <span style="color: rgba(255,255,255,0.7); font-size: 0.8rem;">Enhanced poster & info fetching enabled</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(100, 100, 0, 0.2); padding: 1rem; border-radius: 10px; border: 2px solid rgba(255, 215, 0, 0.3); margin-top: 1rem;">
                <p style="color: rgba(255, 215, 0, 0.9); font-size: 0.85rem; margin: 0;">
                    💡 Tip: Set OMDB_API_KEY environment variable<br>
                    <span style="color: rgba(255,255,255,0.7); font-size: 0.75rem;">for more reliable poster & info fetching</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Main content with cinematic header
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🎬 CinemaScope</div>
        <div class="header-subtitle">Discover Your Next Favorite Movie with AI-Powered Recommendations</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Stats section
    st.markdown("""
    <div class="stats-container">
        <div class="stat-box">
            <div class="stat-number">{}</div>
            <div class="stat-label">Movies Available</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">26</div>
            <div class="stat-label">Genres</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">KNN</div>
            <div class="stat-label">AI Algorithm</div>
        </div>
    </div>
    """.format(len(movie_titles)), unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">🎯 Choose Your Recommendation Type</div>', unsafe_allow_html=True)
    
    genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
              'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
              'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
    movies = [title[0] for title in movie_titles]
    
    # Recommendation type selection
    category = ['--Select--', 'Movie based', 'Genre based']
    cat_op = st.selectbox('**Select Recommendation Type**', category, key='cat_select')
    
    if cat_op == category[0]:
        st.warning('⚠️ Please select a recommendation type to continue!')
        st.markdown("""
        <div class="info-box">
            <h4>💡 How It Works:</h4>
            <ul style="color: rgba(255, 255, 255, 0.9); line-height: 2;">
                <li><strong style="color: #ffd700;">Movie-based:</strong> Get recommendations similar to a movie you love</li>
                <li><strong style="color: #ffd700;">Genre-based:</strong> Discover movies based on your favorite genres</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif cat_op == category[1]:  # Movie-based recommendations
        st.markdown("### 🎬 Select a Movie")
        select_movie = st.selectbox(
            'Choose a movie you enjoyed (recommendations will be based on this selection)',
            ['--Select--'] + movies,
            key='movie_select'
        )
        
        if select_movie != '--Select--':
            dec = st.radio("**Display Options**", ('Show Posters', 'Text Only'), key='poster_radio1')
            show_poster = (dec == 'Show Posters')
            
            if show_poster:
                st.info("ℹ️ Fetching movie posters may take a moment. Please be patient.")
            
            no_of_reco = st.slider(
                '**Number of recommendations:**',
                min_value=5,
                max_value=20,
                step=1,
                value=10,
                key='num_reco1'
            )
            
            if st.button('🔍 Get Recommendations', key='get_reco1'):
                with st.spinner('🎬 Analyzing movies and generating recommendations...'):
                    genres_data = data[movies.index(select_movie)]
                    test_points = genres_data
                    table = KNN_Movie_Recommender(test_points, no_of_reco + 1)
                    table.pop(0)
                    
                    st.markdown(f'<div class="section-title">✨ Recommended Movies Similar to "{select_movie}"</div>', unsafe_allow_html=True)
                    
                    # Add progress bar
                    progress_bar = st.progress(0)
                    total_movies = len(table)
                    
                    for idx, (movie, link, ratings) in enumerate(table, 1):
                        progress_bar.progress((idx) / total_movies)
                        display_movie_card(movie, link, ratings, idx, show_poster)
                    
                    progress_bar.empty()
    
    elif cat_op == category[2]:  # Genre-based recommendations
        st.markdown("### 🎭 Select Your Favorite Genres")
        sel_gen = st.multiselect(
            'Choose one or more genres:',
            genres,
            key='genre_select'
        )
        
        if sel_gen:
            st.markdown(f"""
            <div class="info-box">
                <strong style="color: #ffd700;">Selected Genres:</strong><br>
                {' '.join([f'<span class="genre-tag">{g}</span>' for g in sel_gen])}
            </div>
            """, unsafe_allow_html=True)
            
            dec = st.radio("**Display Options**", ('Show Posters', 'Text Only'), key='poster_radio2')
            show_poster = (dec == 'Show Posters')
            
            if show_poster:
                st.info("ℹ️ Fetching movie posters may take a moment. Please be patient.")
            
            col1, col2 = st.columns(2)
            with col1:
                imdb_score = st.slider(
                    '**Minimum IMDb Score:**',
                    1, 10, 8,
                    key='imdb_slider'
                )
            with col2:
                no_of_reco = st.number_input(
                    '**Number of recommendations:**',
                    min_value=5,
                    max_value=20,
                    step=1,
                    value=10,
                    key='num_reco2'
                )
            
            if st.button('🔍 Get Recommendations', key='get_reco2'):
                with st.spinner('🎬 Finding the perfect movies for you...'):
                    test_point = [1 if genre in sel_gen else 0 for genre in genres]
                    test_point.append(imdb_score)
                    table = KNN_Movie_Recommender(test_point, no_of_reco)
                    
                    st.markdown(f'<div class="section-title">✨ Movies Matching Your Preferences</div>', unsafe_allow_html=True)
                    
                    # Add progress bar
                    progress_bar = st.progress(0)
                    total_movies = len(table)
                    
                    for idx, (movie, link, ratings) in enumerate(table, 1):
                        progress_bar.progress((idx) / total_movies)
                        display_movie_card(movie, link, ratings, idx, show_poster)
                    
                    progress_bar.empty()
        else:
            st.info("👆 Please select at least one genre to get recommendations.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
