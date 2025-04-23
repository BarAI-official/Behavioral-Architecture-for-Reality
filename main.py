"""
BarAI Configuration Module

Centralizes configuration loading, validation, and dynamic updates for all BarAI components,
supporting environment variables, .env files, YAML, and remote config sources.
"""
import os
import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Load .env if present
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

logger = logging.getLogger('barai.config')

class ConfigError(Exception):
    pass

class ConfigLoader:
    """
    Unified loader supporting multiple sources: environment, .env, YAML, JSON, and remote.
    """
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.config: Dict[str, Any] = {}

    def load_env(self, prefix: Optional[str] = None) -> None:
        """Load variables from environment variables (optionally filtered by prefix)."""
        for key, val in os.environ.items():
            if not prefix or key.startswith(prefix):
                self.config[key.lower()] = val
        logger.info('Environment variables loaded: %d entries', len(self.config))

    def load_yaml(self, filename: str) -> None:
        path = self.base_dir / filename
        if path.exists():
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            self.config.update(data or {})
            logger.info('YAML config loaded from %s', filename)

    def load_json(self, filename: str) -> None:
        path = self.base_dir / filename
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
            self.config.update(data or {})
            logger.info('JSON config loaded from %s', filename)

    def fetch_remote(self, url: str) -> None:
        """Optional: fetch remote config via HTTP and merge."""
        try:
            import requests
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            self.config.update(data)
            logger.info('Remote config fetched from %s', url)
        except Exception as e:
            logger.warning('Failed remote fetch: %s', e)

    def get(self, key: str, default: Any = None, cast: Any = None, required: bool = False) -> Any:
        raw = self.config.get(key.lower(), default)
        if required and raw is None:
            raise ConfigError(f"Missing required config: {key}")
        if raw is not None and cast:
            try:
                raw = cast(raw)
            except Exception as e:
                raise ConfigError(f"Invalid cast for {key}: {e}")
        return raw

class BarAIConfig:
    """Typed configuration for BarAI."""
    def __init__(self, loader: ConfigLoader):
        # Load in order: YAML, JSON, env
        loader.load_yaml('config.yaml')
        loader.load_json('config.json')
        loader.load_env(prefix='BARAI_')
        # Optional remote config
        remote_url = loader.get('remote_config_url', default=None)
        if remote_url:
            loader.fetch_remote(remote_url)

        # Core settings
        self.nlp_model_path: Path = Path(loader.get('nlp_model_path', required=True))
        self.sentiment_model: str = loader.get('sentiment_model', default='nlptown/bert-base-multilingual-uncased-sentiment')
        self.audio_model_path: Optional[Path] = Path(loader.get('audio_model_path')) if loader.get('audio_model_path') else None

        self.provider_url: str = loader.get('blockchain_provider_url', required=True)
        self.contract_address: str = loader.get('contract_address', required=True)
        self.contract_abi_path: Path = Path(loader.get('contract_abi_path', required=True))

        # Logging settings
        self.log_level: str = loader.get('log_level', default='INFO')
        self.log_json: bool = loader.get('log_json', default='False', cast=lambda x: x.lower()=='true')

        # Feature flags
        self.enable_microexpressions: bool = loader.get('enable_microexpressions', default='False', cast=lambda x: x.lower()=='true')
        self.enable_remote_config: bool = bool(remote_url)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'nlp_model_path': str(self.nlp_model_path),
            'sentiment_model': self.sentiment_model,
            'audio_model_path': str(self.audio_model_path) if self.audio_model_path else None,
            'provider_url': self.provider_url,
            'contract_address': self.contract_address,
            'contract_abi_path': str(self.contract_abi_path),
            'log_level': self.log_level,
            'log_json': self.log_json,
            'enable_microexpressions': self.enable_microexpressions,
            'enable_remote_config': self.enable_remote_config
        }

# Entry point

def load_config() -> BarAIConfig:
    base = Path(__file__).parent
    loader = ConfigLoader(base)
    return BarAIConfig(loader)

if __name__ == '__main__':
    cfg = load_config()
    print(json.dumps(cfg.to_dict(), indent=2))
