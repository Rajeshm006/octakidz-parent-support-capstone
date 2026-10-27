"""Recompute the published semantic scorecards without provider requests."""
import json
from pathlib import Path
from quality_metrics import summarise

root=Path(__file__).resolve().parent/'evidence/quality'
metrics=summarise(json.loads((root/'results.json').read_text()),json.loads((root/'reviews.json').read_text()))
print(json.dumps({'core_acceptance':metrics['core_acceptance'],'consistency':metrics['consistency'],
    'limitations':'Known authored cases reviewed by the assistant; no independent/held-out accuracy or grade claim. This command makes no API requests.'},indent=2))
