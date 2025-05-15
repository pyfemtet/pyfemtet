import os
import yaml
from typing import TypedDict

__all__ = ['secret']


class Secret(TypedDict):
    NOTIFICATION_TO: list
    NOTIFICATION_FROM: str
    SMTP_SERVER: str
    SMTP_PORT: int


secret_path = os.path.join(os.path.dirname(__file__), 'secret.yaml')

with open(secret_path) as f:
    secret: Secret = yaml.safe_load(f)
