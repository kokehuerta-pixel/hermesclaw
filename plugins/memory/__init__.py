import logging
import importlib
import importlib.util
import os
from typing import TYPE_CHECKING, Any
from pathlib import Path

if TYPE_CHECKING:
    from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

class _ProviderCollector:
    """Helper to collect a provider registered via register(ctx)."""
    def __init__(self):
        self.provider = None

    def register_memory_provider(self, provider: "MemoryProvider"):
        self.provider = provider

def load_memory_provider(name: str, config: dict[str, Any] = None) -> "MemoryProvider":
    """Dynamic loader for memory provider plugins. Supports bundled and user plugins."""
    mod = None

    # 1. Try repo-bundled providers first (plugins.memory.<name>)
    try:
        mod = importlib.import_module(f"plugins.memory.{name}")
    except ImportError:
        # 2. Try user-installed plugins (~/.hermes/plugins/<name>/)
        from hermes_constants import get_hermes_home
        plugin_dir = get_hermes_home() / "plugins" / name
        init_file = plugin_dir / "__init__.py"
        
        if init_file.exists():
            spec = importlib.util.spec_from_file_location(f"user_plugins.memory.{name}", str(init_file))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
    
    if not mod:
        logger.error("Memory provider %s not found in bundled or user plugins", name)
        raise ImportError(f"No memory provider found for '{name}'")

    # Try register(ctx) pattern first (how our plugins are written)
    if hasattr(mod, "register"):
        collector = _ProviderCollector()
        try:
            mod.register(collector)
            if collector.provider:
                return collector.provider
        except Exception as e:
            logger.debug("register() failed for %s: %s", name, e)

    # Fallback: look for a class named <Name>MemoryProvider
    # Handle snake_case to CamelCase conversion for class names
    camel_name = "".join(word.capitalize() for word in name.split("_"))
    class_name = f"{camel_name}MemoryProvider"
    
    if hasattr(mod, class_name):
        provider_cls = getattr(mod, class_name)
        provider = provider_cls()
        return provider

    raise ValueError(f"No valid memory provider found in plugins.memory.{name}")

def discover_memory_providers() -> list[tuple[str, str, bool]]:
    """Discover available memory providers in bundled and user plugins."""
    from hermes_constants import get_hermes_home
    
    providers = []
    seen_names = set()

    def _get_info(d: Path) -> tuple[str, str]:
        """Helper to get name and description from plugin.yaml or folder name."""
        name = d.name
        desc = f"Memory provider from {d.name}"
        
        manifest_file = d / "plugin.yaml"
        if not manifest_file.exists():
            manifest_file = d / "plugin.yml"
            
        if manifest_file.exists():
            try:
                import yaml
                with open(manifest_file, "r") as f:
                    manifest = yaml.safe_load(f) or {}
                    name = manifest.get("name", name)
                    desc = manifest.get("description", desc)
            except Exception:
                pass
        return name, desc

    # 1. Bundled providers
    bundled_dir = Path(__file__).parent
    if bundled_dir.is_dir():
        for d in bundled_dir.iterdir():
            if d.is_dir() and not d.name.startswith("__"):
                name, desc = _get_info(d)
                providers.append((name, desc, True))
                seen_names.add(name)

    # 2. User providers
    user_plugins_dir = get_hermes_home() / "plugins"
    if user_plugins_dir.is_dir():
        for d in user_plugins_dir.iterdir():
            if d.is_dir() and not d.name.startswith("__"):
                # We check for __init__.py to ensure it's a loadable plugin
                if (d / "__init__.py").exists():
                    name, desc = _get_info(d)
                    if name not in seen_names:
                        providers.append((name, desc, True))
                        seen_names.add(name)

    return providers
