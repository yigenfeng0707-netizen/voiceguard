"""Shared test setup for VoiceGuard test suite.

Ensures scripts/ is importable without heavy ML deps (funasr/openvino_genai).
This file is auto-loaded by pytest; the sys.path manipulation runs at collection time.
"""

import os
import sys

# Add scripts/ to path for all test modules
_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
_SCRIPTS = os.path.normpath(_SCRIPTS)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_DIR = os.path.join(PROJECT_ROOT, "rules")
