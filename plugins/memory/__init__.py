import logging
import importlib
import importlib.util
import os
from typing import TYPE_CHECKING, Any

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
    """Dynamic loader for memory provider plugins.

    # 1. Try repo-bundled providers first (plugins.memory.<name>)
    try:
        mod = importlib.import_module(f"plugins.memory.{name}")
    except ImportError:
        # 2. Try user-installed plugins (~/.hermes/plugins/<name>/)
        from hermes_constants import get_hermes_home
        plugin_dir = get_hermes_home() / "plugins" / name
        init_file = plugin_dir / "__init__.py"
        
        if not init_file.exists():
            logger.error("Memory provider %s not found in bundled or user plugins", name)
            raise ImportError(f"No memory provider found for '{name}'")
            
        spec = importlib.util.spec_from_file_location(f"user_plugins.memory.{name}", str(init_file))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            raise ImportError(f"Could not load memory provider from {init_file}")

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
    class_name = f"{name.capitalize()}MemoryProvider"
    if hasattr(mod, class_name):
        provider_cls = getattr(mod, class_name)
        provider = provider_cls()
        return provider

    raise ValueError(f"No valid memory provider found in plugins.memory.{name}")
