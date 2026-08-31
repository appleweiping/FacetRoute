"""End-to-end offline demo; run after ``pip install -e .``."""

from pathlib import Path

from facetroute import FeedbackLog, LinUCBRouter, OfflineSimulator
from facetroute.config import load_models, load_preferences, load_requests, load_rules

root = Path(__file__).resolve().parent
state_dir = root / "_state"
state_dir.mkdir(exist_ok=True)
models = load_models(root / "models.json")
preferences = load_preferences(root / "preferences.json")
rules = load_rules(root / "rules.json")
requests = load_requests(root / "queries.jsonl")

router = LinUCBRouter(models, preferences, rules)
feedback_path = state_dir / "demo-feedback.jsonl"
simulator = OfflineSimulator(router, models, seed=17, feedback_log=FeedbackLog(feedback_path))
observations, report = simulator.run(requests, learn=True)

for observation in observations:
    decision = observation.decision
    print(f"{decision.request_id}: {decision.selected_model} ({decision.score:.3f})")
print(report.to_dict())

router.save_state(state_dir / "demo-bandit-state.json")
