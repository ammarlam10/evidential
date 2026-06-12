"""
Import model modules so their @register_model decorators fire when
`src.models` is imported.
"""

from src.models import factory  # noqa: F401 - must come first
from src.models import unet_evidential  # noqa: F401

from src.models.factory import build_model, list_models  # noqa: F401
