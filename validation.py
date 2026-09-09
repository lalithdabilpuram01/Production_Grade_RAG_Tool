# validation.py (async version)

from groq import AsyncGroq
import asyncio

from groq_models import GRADER, resolve_model


async def validate_groq_api_key_async(api_key):
    """Async validation for better performance. 
    Checks if the provided Groq API key is valid by attempting a lightweight API call.
    """
    # Check if the user provided an empty string
    if not api_key:
        return False, "API key is empty"

    # Remove any accidental leading/trailing spaces or quote characters
    cleaned_key = api_key.strip().strip("'\"")

    # Perform a basic length check before making network requests
    if len(cleaned_key) < 20:
        return False, "API key is too short"

    try:
        # Initialize the asynchronous Groq client with the user's key
        client = AsyncGroq(api_key=cleaned_key)

        # Attempt a tiny, low-cost API call to verify the key works. The model
        # is resolved against the live catalogue so a retired id cannot make a
        # valid key look invalid.
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model=resolve_model(GRADER),
            max_tokens=5
        )

        # If the call succeeds, the key is valid
        return True, "API key is valid"

    except Exception as e:
        # Catch any errors (like authentication failures)
        error_msg = str(e)
        # A 401 error specifically means unauthorized/invalid key
        if "401" in error_msg:
            return False, "Invalid API key"
        else:
            # Handle other potential errors like network issues or rate limits
            return False, f"API error: {error_msg}"


def validate_groq_api_key(api_key):
    """Sync wrapper for async validation.
    Allows standard synchronous code (like our Streamlit app) to call the async validation function.
    """
    return asyncio.run(validate_groq_api_key_async(api_key))