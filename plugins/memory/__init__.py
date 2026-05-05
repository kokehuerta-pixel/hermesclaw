import logging
import importlib
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

    Resolution order:
      1. plugins.memory.<name>
      2. ~/.hermes/plugins/<name> (TODO)
    """
    try:
        mod = importlib.import_module(f"plugins.memory.{name}")
    except ImportError as e:
        logger.error("Failed to import memory provider %s: %s", name, e)
        raise

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
