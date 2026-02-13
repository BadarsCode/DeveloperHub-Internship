from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class SimpleDocument:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
