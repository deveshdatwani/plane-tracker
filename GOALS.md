Objective: Maximize HOTA (Higher Order Tracking Accuracy) across all 6 sequences.
Metric: Run python eval_harness.py to get current performance.
Workflow: > 1. Propose a tracking architecture (e.g., OC-SORT + YOLOv11).
2. Write the implementation in tracker.py.
3. Run evaluation.
4. If HOTA < 0.85, read the logs/error_cases.log and iterate.
5. STRICT RULE: Use Leave-One-Out Cross-Validation. Never train on the sequence you are testing.
