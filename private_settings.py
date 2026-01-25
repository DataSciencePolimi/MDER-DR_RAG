"""Private settings for the project."""

# pylint: disable=line-too-long

PRIVATE_SETINGS = {
    "LLM_LOCAL": True,
    "LLM_KEY": {
        "openai": "", 
        "ollama": "a_key",
        "anthropic": "",
        "deepseek": "",
    },
    "LLM_BASE_URL": "http://localhost:11434",  # ollama
}
 
