import yaml

with open("secrets.yaml", "r") as file:
    _secrets: dict[str, str] = yaml.safe_load(file)
    OPENROUTER_API_KEY: str = _secrets.get("openrouter_api_key", "")
