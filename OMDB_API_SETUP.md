# OMDB API Setup (Optional)

The movie recommendation app now uses OMDB API as a fallback for fetching movie posters and descriptions when IMDB scraping fails. This significantly improves reliability.

## How to Get an OMDB API Key (Free)

1. Visit: http://www.omdbapi.com/apikey.aspx
2. Choose the **FREE** tier (1,000 requests per day)
3. Enter your email address
4. Check your email and click the activation link
5. Copy your API key

## How to Set the API Key

### Option 1: Environment Variable (Recommended)
Set the `OMDB_API_KEY` environment variable:
- **Windows (PowerShell):** `$env:OMDB_API_KEY="your_api_key_here"`
- **Windows (CMD):** `set OMDB_API_KEY=your_api_key_here`
- **Linux/Mac:** `export OMDB_API_KEY="your_api_key_here"`

### Option 2: Streamlit Secrets
Create a `.streamlit/secrets.toml` file in your project directory:
```toml
omdb_api_key = "your_api_key_here"
```

## How It Works

1. **With API Key:** The app tries OMDB API first (most reliable), then falls back to IMDB scraping if needed.
2. **Without API Key:** The app uses IMDB scraping only (current behavior).

The app will work fine without an API key, but having one provides:
- ✅ More reliable poster fetching
- ✅ Better movie descriptions
- ✅ Faster response times
- ✅ More consistent data

## Note

The OMDB API is completely optional. The app will continue to work using IMDB scraping if no API key is provided.




