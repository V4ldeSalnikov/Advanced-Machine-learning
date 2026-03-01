"""
Simple registry so that models can be added / retrieved by name.

Usage
-----
    from project.models.registry import ModelRegistry

    # Register a new model class
    @ModelRegistry.register("latent_ddpm")
    class LatentDDPMModel(GenerativeModel):
        ...

    # Or register manually
    ModelRegistry.register("vae")(VAEModel)

    # Retrieve & instantiate
    model = ModelRegistry.create("latent_ddpm", device="cuda")

    # List all registered names
    print(ModelRegistry.list())
"""

from __future__ import annotations

from typing import Dict, Type

from .base import GenerativeModel


class ModelRegistry:
    """Global name→class mapping for generative models."""

    _registry: Dict[str, Type[GenerativeModel]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator / function to register a model class under *name*."""
        def decorator(model_cls: Type[GenerativeModel]):
            cls._registry[name] = model_cls
            model_cls.name = name
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> GenerativeModel:
        """Instantiate a registered model by *name*, forwarding **kwargs."""
        if name not in cls._registry:
            raise KeyError(
                f"Model '{name}' not registered. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def list(cls):
        return list(cls._registry.keys())

    @classmethod
    def get_class(cls, name: str) -> Type[GenerativeModel]:
        return cls._registry[name]
