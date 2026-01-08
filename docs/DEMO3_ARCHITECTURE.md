# Demo 3: Multi-Video Analysis Platform - Architecture Specification

**Author:** Claude (Partner-Level Microsoft SDE)
**Date:** 2026-01-08
**Status:** Design Phase
**Stakeholder:** Warren (Engineering Manager)

---

## Executive Summary

Demo 3 transforms the single-video Demo 2 into a comprehensive **Multi-Video Analysis Platform** that enables toggling between all 26 Cultivate Learning videos, comparing ML predictions against expert ground truth annotations, and triggering on-demand transcription via Azure VM.

### Key Metrics
- **26 videos** from secure data folder (5.1GB total)
- **98 processed transcripts** with word-level timestamps
- **119 expert annotations** (ground truth from CSV)
- **Real-time ML classification** via batch API

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DEMO 3 FRONTEND                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │   Video     │  │  Question    │  │  Ground     │  │  Analytics   │  │
│  │  Selector   │  │ Classifier   │  │   Truth     │  │  Dashboard   │  │
│  │  (26 vids)  │  │   (ML API)   │  │ Comparison  │  │  (Metrics)   │  │
│  └──────┬──────┘  └──────┬───────┘  └──────┬──────┘  └──────┬───────┘  │
│         │                │                 │                 │          │
│         └────────────────┴────────┬────────┴─────────────────┘          │
│                                   │                                      │
└───────────────────────────────────┼──────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        ┌───────────▼───────────┐     ┌────────────▼────────────┐
        │    LOCAL ML API       │     │     AZURE RESOURCES     │
        │   (FastAPI:5001)      │     │                         │
        │                       │     │  ┌──────────────────┐   │
        │  /api/v1/classify     │     │  │  Azure Blob      │   │
        │  /api/v2/classify     │     │  │  (Video Storage) │   │
        │  /api/v1/batch        │     │  └──────────────────┘   │
        │  /api/v2/batch        │     │                         │
        └───────────────────────┘     │  ┌──────────────────┐   │
                                      │  │  Azure VM        │   │
                                      │  │  (Whisper GPU)   │   │
                                      │  │  52.183.73.2     │   │
                                      │  └──────────────────┘   │
                                      └─────────────────────────┘
```

---

## Data Architecture

### 1. Video Catalog Structure

```json
{
  "videos": [
    {
      "id": "161765962",
      "filename": "Launch Highland Park 004.mp4",
      "display_name": "Highland Park Snack Time",
      "age_group": "PK",
      "duration_seconds": 40,
      "file_size_mb": 39,
      "location": {
        "local": "/home/warrenjo/src/tmp2/secure data/Launch Highland Park 004.mp4",
        "azure_blob": "https://cultivatemlvideos.blob.core.windows.net/videos/161765962.mp4"
      },
      "transcripts_available": true,
      "transcript_count": 5,
      "questions": [
        {
          "id": "q1",
          "timestamp": "0:02",
          "ground_truth": {
            "type": "OEQ",
            "description": "OEQ with a pause but child shrugs"
          },
          "ml_prediction": null
        }
      ]
    }
  ]
}
```

### 2. Ground Truth Mapping (CSV → JSON)

The CSV contains expert annotations with this schema:

| Field | Type | Example |
|-------|------|---------|
| Video Title | String | "Launch Highland Park 004.mp4" |
| Asset # | Integer | 161765962 |
| Age group | Enum | PK, TODDLER, INFANT |
| Timestamp | String | "0:00-0:45" |
| Question N | String | "0:02" |
| QN description | String | "OEQ with a pause but child shrugs" |
| Description | Text | Full video description |

**Parsing Logic:**
1. Extract video filename → map to asset ID
2. Parse question timestamps (Q1-Q8)
3. Extract classification (OEQ/CEQ) and behavioral notes
4. Link to existing transcript JSONs

### 3. Transcript Integration

Existing transcripts in `data/transcripts/transcripts/` contain:
- `original_video`: Maps to video filename
- `quality_annotation`: Contains ground truth
- `transcript_segments`: Word-level Whisper output
- `processing_metadata`: Model info, confidence

---

## Component Specifications

### Component 1: Video Selector

**Purpose:** Enable toggling between all 26 Cultivate Learning videos

**UI Elements:**
- Dropdown selector with video thumbnails
- Filter buttons by age group (All | PK | Toddler | Infant)
- Video metadata card (duration, question count, processing status)
- Quick stats badges (OEQ count, CEQ count, accuracy %)

**Implementation:**
```javascript
// Video selector component
const VideoSelector = {
  videos: [],           // Loaded from video_catalog.json
  currentVideo: null,   // Currently selected video
  filters: {
    ageGroup: 'all',    // PK, TODDLER, INFANT, all
    hasTranscript: true // Only show videos with transcripts
  },

  async loadCatalog() {
    const response = await fetch('./video_catalog.json');
    this.videos = await response.json();
  },

  selectVideo(videoId) {
    this.currentVideo = this.videos.find(v => v.id === videoId);
    this.dispatchEvent('videoSelected', this.currentVideo);
  }
};
```

### Component 2: Question Classification Panel

**Purpose:** Display ML classifications with real-time re-classification

**Features:**
- Batch re-classification on video load (like Demo 2)
- Model toggle (Classic vs Ensemble)
- Per-question confidence display
- Rate button for human feedback

**API Integration:**
```javascript
async function classifyVideoQuestions(video) {
  const questions = video.questions.map(q => q.transcript_text);

  const response = await fetch(`${API_BASE}/api/v2/classify/batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      questions: questions,
      model: currentModel // 'classic' or 'ensemble'
    })
  });

  return await response.json();
}
```

### Component 3: Ground Truth Comparison

**Purpose:** Side-by-side comparison of ML vs Expert annotations

**Metrics Calculated:**
- **Accuracy:** (TP + TN) / Total
- **Precision (OEQ):** TP / (TP + FP)
- **Recall (OEQ):** TP / (TP + FN)
- **F1 Score:** 2 * (Precision * Recall) / (Precision + Recall)
- **Cohen's Kappa:** Inter-rater agreement

**UI Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  QUESTION COMPARISON                                        │
├─────────────────────────────────────────────────────────────┤
│  Q1 @ 0:02  "What are you going to do next?"               │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │ GROUND TRUTH   │  │ ML PREDICTION  │  │   MATCH?     │  │
│  │ OEQ            │  │ OEQ (87%)      │  │ ✅ CORRECT   │  │
│  │ "pause, child  │  │ "OEQ indicators│  │              │  │
│  │  shrugs"       │  │  detected"     │  │              │  │
│  └────────────────┘  └────────────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Q2 @ 0:04  "Are you going to eat it?"                     │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │ GROUND TRUTH   │  │ ML PREDICTION  │  │   MATCH?     │  │
│  │ CEQ (yes/no)   │  │ CEQ (92%)      │  │ ✅ CORRECT   │  │
│  └────────────────┘  └────────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Component 4: Analytics Dashboard

**Purpose:** Cross-video performance metrics and insights

**Tabs:**
1. **Overview:** Total questions, OEQ/CEQ distribution, model accuracy
2. **By Video:** Per-video accuracy breakdown
3. **By Age Group:** Performance across PK/Toddler/Infant
4. **Model Comparison:** Classic vs Ensemble head-to-head
5. **Error Analysis:** Common misclassification patterns

**Visualizations:**
- Confusion matrix (2x2: OEQ vs CEQ)
- Accuracy bar chart by video
- Confidence distribution histogram
- ROC curve for model comparison

### Component 5: Azure VM Transcription Trigger

**Purpose:** On-demand transcription for videos without transcripts

**Workflow:**
1. User selects video without transcript
2. UI shows "Transcription Required" state
3. User clicks "Transcribe with Azure VM"
4. SSH command triggers Whisper processing
5. Progress polling until complete
6. Transcript loaded into UI

**Implementation:**
```javascript
async function triggerAzureTranscription(videoPath) {
  const response = await fetch('/api/transcription/trigger', {
    method: 'POST',
    body: JSON.stringify({
      video_path: videoPath,
      azure_vm: '52.183.73.2',
      whisper_model: 'medium'
    })
  });

  return response.json(); // { job_id, estimated_time }
}

async function pollTranscriptionStatus(jobId) {
  const response = await fetch(`/api/transcription/status/${jobId}`);
  return response.json(); // { status: 'processing' | 'complete', progress: 0.75 }
}
```

---

## File Structure

```
unified-demos/
├── demo3/
│   ├── index.html              # Main Demo 3 page
│   ├── app.js                  # Core application logic
│   ├── styles.css              # Demo 3 specific styles
│   ├── components/
│   │   ├── video-selector.js   # Video selection component
│   │   ├── question-panel.js   # Question classification panel
│   │   ├── ground-truth.js     # Ground truth comparison
│   │   ├── analytics.js        # Analytics dashboard
│   │   └── transcription.js    # Azure VM transcription trigger
│   ├── data/
│   │   ├── video_catalog.json  # All 26 videos with metadata
│   │   └── ground_truth.json   # Parsed CSV annotations
│   └── utils/
│       ├── metrics.js          # Accuracy/precision/recall calculations
│       └── csv-parser.js       # CSV ground truth parser
```

---

## Data Pipeline

### Step 1: Parse CSV Ground Truth

```python
# scripts/parse_ground_truth.py
import csv
import json
import re

def parse_ground_truth_csv(csv_path):
    """Parse Cultivate Learning expert annotations CSV"""
    videos = {}

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_title = row['Video Title']
            asset_id = row['Asset #']

            if not asset_id or not video_title:
                continue

            video = {
                'id': asset_id,
                'filename': video_title,
                'age_group': row['Age group'].upper(),
                'timestamp_range': row['Timestamp'],
                'description': row['Description'],
                'questions': []
            }

            # Parse Q1-Q8
            for i in range(1, 9):
                q_col = f'Question {i} '
                desc_col = f'Q{i} description'

                timestamp = row.get(q_col, '').strip()
                description = row.get(desc_col, '').strip()

                if timestamp and timestamp != 'na':
                    q_type = 'OEQ' if 'OEQ' in description else 'CEQ'
                    video['questions'].append({
                        'id': f'q{i}',
                        'timestamp': timestamp,
                        'type': q_type,
                        'description': description
                    })

            videos[asset_id] = video

    return videos
```

### Step 2: Build Video Catalog

```python
# scripts/build_video_catalog.py
import os
import json
from pathlib import Path

def build_video_catalog(secure_data_path, transcripts_path, ground_truth):
    """Build comprehensive video catalog for Demo 3"""
    catalog = {'videos': [], 'metadata': {}}

    # Scan secure data folder for videos
    video_extensions = {'.mp4', '.MP4', '.mov', '.MOV'}

    for video_file in Path(secure_data_path).iterdir():
        if video_file.suffix in video_extensions:
            # Find matching ground truth
            gt = find_ground_truth_for_video(video_file.name, ground_truth)

            # Find matching transcripts
            transcripts = find_transcripts_for_asset(gt['id'], transcripts_path) if gt else []

            video_entry = {
                'id': gt['id'] if gt else video_file.stem,
                'filename': video_file.name,
                'display_name': generate_display_name(video_file.name),
                'age_group': gt['age_group'] if gt else 'UNKNOWN',
                'file_size_mb': video_file.stat().st_size / (1024 * 1024),
                'local_path': str(video_file),
                'has_transcripts': len(transcripts) > 0,
                'transcript_count': len(transcripts),
                'questions': gt['questions'] if gt else [],
                'description': gt['description'] if gt else ''
            }

            catalog['videos'].append(video_entry)

    catalog['metadata'] = {
        'total_videos': len(catalog['videos']),
        'with_transcripts': sum(1 for v in catalog['videos'] if v['has_transcripts']),
        'total_questions': sum(len(v['questions']) for v in catalog['videos']),
        'generated_at': datetime.now().isoformat()
    }

    return catalog
```

### Step 3: Link Transcripts to Questions

```python
# scripts/link_transcripts.py
def link_transcripts_to_questions(catalog, transcripts_path):
    """Link existing Whisper transcripts to video questions"""

    for video in catalog['videos']:
        asset_id = video['id']

        for question in video['questions']:
            q_id = question['id']
            transcript_file = f"{asset_id}_{q_id}_transcript.json"
            transcript_path = Path(transcripts_path) / transcript_file

            if transcript_path.exists():
                with open(transcript_path) as f:
                    transcript = json.load(f)

                # Extract transcript text
                question['transcript_text'] = ' '.join(
                    seg['text'] for seg in transcript.get('transcript_segments', [])
                )
                question['transcript_confidence'] = transcript.get(
                    'processing_metadata', {}
                ).get('transcription_confidence', 0)
                question['has_transcript'] = True
            else:
                question['has_transcript'] = False

    return catalog
```

---

## API Endpoints (New)

### Transcription Trigger API

```python
# src/api/endpoints/transcription_trigger.py
from fastapi import APIRouter, BackgroundTasks
import asyncio
import subprocess

router = APIRouter(prefix="/api/transcription", tags=["transcription"])

@router.post("/trigger")
async def trigger_transcription(
    video_path: str,
    azure_vm: str = "52.183.73.2",
    whisper_model: str = "medium",
    background_tasks: BackgroundTasks
):
    """Trigger Whisper transcription on Azure VM"""
    job_id = generate_job_id()

    background_tasks.add_task(
        run_azure_transcription,
        job_id, video_path, azure_vm, whisper_model
    )

    return {
        "job_id": job_id,
        "status": "started",
        "estimated_time_seconds": estimate_transcription_time(video_path)
    }

@router.get("/status/{job_id}")
async def get_transcription_status(job_id: str):
    """Check transcription job status"""
    status = get_job_status(job_id)
    return status
```

---

## UI Mockup

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DEMO 3 - Multi-Video Analysis Platform          [Classic ▼] [🌙 Dark Mode] │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ SELECT VIDEO                                [All ▼] [PK] [Toddler]  │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │ │Highland │ │Structure│ │ Book    │ │Building │ │ Launch  │ ...   │   │
│  │ │  Park   │ │Activity │ │Flowers  │ │ Blocks  │ │Cascade  │       │   │
│  │ │  PK     │ │   PK    │ │   PK    │ │ TODDLER │ │   PK    │       │   │
│  │ │ 5 Qs    │ │  5 Qs   │ │  4 Qs   │ │  4 Qs   │ │  3 Qs   │       │   │
│  │ │ ✅ 98%  │ │ ✅ 92%  │ │ ⚠️ 75%  │ │ ✅ 100% │ │ ✅ 89%  │       │   │
│  │ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌───────────────────────────────┬─────────────────────────────────────┐   │
│  │ VIDEO PLAYER                  │ QUESTION CLASSIFICATION             │   │
│  │ ┌───────────────────────────┐ │ ┌─────────────────────────────────┐ │   │
│  │ │                           │ │ │ Q1 @ 0:02 ✅                    │ │   │
│  │ │      [VIDEO PLAYER]       │ │ │ "What are you going to do?"     │ │   │
│  │ │                           │ │ │ GT: OEQ | ML: OEQ (87%)         │ │   │
│  │ │      Highland Park        │ │ ├─────────────────────────────────┤ │   │
│  │ │      00:15 / 00:40        │ │ │ Q2 @ 0:04 ✅                    │ │   │
│  │ │                           │ │ │ "Are you going to eat it?"      │ │   │
│  │ └───────────────────────────┘ │ │ GT: CEQ | ML: CEQ (92%)         │ │   │
│  │                               │ ├─────────────────────────────────┤ │   │
│  │ Duration: 40s | PK | 5 Qs     │ │ Q3 @ 0:12 ❌                    │ │   │
│  │ Description: Children eating  │ │ "Do you want more?"             │ │   │
│  │ snack, educator asks Qs...    │ │ GT: CEQ | ML: OEQ (54%) ⚠️      │ │   │
│  └───────────────────────────────┴─────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ANALYTICS                                                           │   │
│  │ [Overview] [By Video] [By Age Group] [Model Comparison] [Errors]   │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  CURRENT VIDEO: Highland Park           │  DATASET TOTALS           │   │
│  │  ───────────────────────────           │  ──────────────────────   │   │
│  │  Total Questions: 5                     │  Total Videos: 26         │   │
│  │  OEQ: 2 | CEQ: 3                       │  Total Questions: 119     │   │
│  │  Accuracy: 80% (4/5)                   │  Overall Accuracy: 87%    │   │
│  │  Precision: 100% | Recall: 100%        │  OEQ F1: 0.84             │   │
│  │                                         │  CEQ F1: 0.89             │   │
│  │  ┌─────────────────┐                   │                           │   │
│  │  │ CONFUSION MATRIX│                   │  [Export Results CSV]     │   │
│  │  │     OEQ   CEQ   │                   │  [Retrain Model]          │   │
│  │  │ OEQ  2     0    │                   │                           │   │
│  │  │ CEQ  1     2    │                   │                           │   │
│  │  └─────────────────┘                   │                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Data Preparation (Day 1)
- [ ] Parse CSV ground truth into JSON
- [ ] Build video catalog linking videos, transcripts, annotations
- [ ] Validate transcript-to-question alignment
- [ ] Create `video_catalog.json` for frontend

### Phase 2: Core UI (Day 1-2)
- [ ] Create demo3/index.html with tab structure
- [ ] Implement video selector component
- [ ] Port question classification from Demo 2
- [ ] Add ground truth comparison panel

### Phase 3: Analytics Dashboard (Day 2)
- [ ] Implement accuracy/precision/recall calculations
- [ ] Create confusion matrix visualization
- [ ] Add per-video and per-age-group breakdowns
- [ ] Build model comparison view

### Phase 4: Azure VM Integration (Day 3)
- [ ] Create transcription trigger API endpoint
- [ ] Implement SSH-based job execution
- [ ] Add progress polling and status display
- [ ] Handle transcript import after completion

### Phase 5: Polish & Testing (Day 3)
- [ ] Responsive design for mobile
- [ ] Error handling and loading states
- [ ] Performance optimization (lazy loading)
- [ ] Comprehensive scenario testing

---

## Technical Decisions

### 1. Video Hosting Strategy
**Decision:** Local file serving for development, Azure Blob for production
**Rationale:**
- 5.1GB of videos too large to include in repo
- Azure Blob already set up for Demo 2 (warren_teaching_web.mp4)
- Local dev uses symlink or direct path

### 2. Ground Truth Source of Truth
**Decision:** Parse CSV once, generate JSON, use JSON in app
**Rationale:**
- CSV is human-editable master
- JSON is faster to load/query
- Rebuild JSON on CSV changes

### 3. ML Classification Approach
**Decision:** Batch API re-classification on video select
**Rationale:**
- Matches Demo 2 pattern (proven)
- Enables model switching
- Fresh classifications always

### 4. Transcript Storage
**Decision:** Use existing 98 transcripts, lazy-transcribe new
**Rationale:**
- Don't re-process already done work
- Azure VM transcription for gaps
- Store results alongside existing

---

## Success Criteria

1. **Functional:** Toggle between all 26 videos with <1s load time
2. **Accurate:** ML vs Ground Truth comparison shows real metrics
3. **Actionable:** Identify videos/questions where model struggles
4. **Extensible:** Easy to add new videos as they arrive
5. **Production-Ready:** Error handling, loading states, responsive UI

---

## Dependencies

- **Existing:** Demo 2 codebase, FastAPI backend, ML models
- **Data:** CSV ground truth, 98 transcripts, 26 video files
- **Infrastructure:** Azure VM (52.183.73.2), Azure Blob Storage
- **Libraries:** Whisper, PyTorch, librosa (already installed)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Azure VM unavailable | Fallback to existing transcripts only |
| Video files too large | Compress to web-optimized MP4 (Demo 2 pattern) |
| Ground truth ambiguous | Use ML confidence threshold for uncertain cases |
| Model accuracy poor | Surface errors for manual review/retraining |

---

## Next Steps

1. **Immediate:** Create `scripts/parse_ground_truth.py` and generate JSON
2. **Today:** Build Demo 3 skeleton with video selector
3. **This Week:** Complete all phases, deploy to staging
4. **Follow-up:** Upload remaining videos to Azure Blob

---

*Document generated by Claude (Partner-Level Microsoft SDE)*
*Review pending: Warren (Engineering Manager)*
