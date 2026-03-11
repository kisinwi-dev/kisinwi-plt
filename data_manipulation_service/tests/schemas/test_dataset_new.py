# Здесь тестируются данные которые получаются от

import pytest
from datetime import datetime
from pydantic import ValidationError

from app.api.schemas.dataset import DatasetMetadata, Version, Source

