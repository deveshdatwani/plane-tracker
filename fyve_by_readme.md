# Fyve By — Senior Computer Vision Engineer Take-Home

This assignment is intentionally open-ended. You are expected and encouraged to use any AI tools to help you — just include your prompts/logs in `llm_prompts.md`. We are looking for engineers who are productive with AI tools and can think creatively about problems.

The ground truth annotations were generated with a custom model. You are not expected to match our results exactly — we want to see how you approach and solve these problems.

Feel free to restructure the code however you see fit. The only constraint is that `run.py`'s I/O contract and output JSON schema must remain unchanged.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

You are free to install and use any additional libraries (e.g. `ultralytics`, `torch`, `detectron2`, etc.). Just make sure to update `requirements.txt` with anything you add.

## Test your setup

```bash
python src/view_annotations.py simple_plane_add
```

This will overlay ground truth annotations on the video. If you see bounding boxes on aircraft, your environment is working.

## Running

```bash
# With visualization
python run.py --video data/simple_plane_add.mp4 --annotations annotations/simple_plane_add.json

# Headless, save annotated video
python run.py --video data/simple_plane_add.mp4 \
              --annotations annotations/simple_plane_add.json \
              --no-display \
              --save-video outputs/result.mp4 \
              --output outputs/results.json
```

## What to implement

| File | Task |
|------|------|
| `src/detector.py` | Part 1 — Detection |
| `src/tracker.py` | Part 2 — Data association + state estimation |
| `src/hangar.py` | Part 3 — Hangar entry detection |
| `src/metrics.py` | Part 4 — Three evaluation metrics |

## Submission

Please submit:
- Your completed source code
- `outputs/results.json` from running on each clip
- `WRITEUP.md` answering all questions
- `llm_prompts.md` — a log of prompts you submitted to any LLM

## Data

Download the video clips and annotations from [Google Drive](https://drive.google.com/drive/folders/1arJE5uhSBPxrlSpw5CNtFjI-UHQNqket?usp=sharing) and place them in the project root:

```
data/
    simple_plane_add.mp4
    simple_plane_add2.mp4
    dynamic_to_static.mp4
    night_plane_overlap.mp4
    multi_plane.mp4
    no_planes.mp4

annotations/
    simple_plane_add.json
    simple_plane_add2.json
    dynamic_to_static.json
    night_plane_overlap.json
    multi_plane.json
    no_planes.json
```

Annotation format:
```json
{
  "frames": {
    "0": [
      {"track_id": 1, "bbox": [x1, y1, x2, y2], "class": "aircraft"}
    ]
  }
}
```

## Time expectation

We expect this to take **3-4 hours**. We value pragmatic solutions over polished ones — do not over-invest. If you run out of time, describe what you would have done next in your write-up.

**Please submit within 48 hours of receiving this challenge.**

## Follow-up

After submission, we may schedule a **20-30 minute debrief** where you will walk through your implementation, explain your design decisions, and discuss tradeoffs. Be prepared to modify your code live during this session.
