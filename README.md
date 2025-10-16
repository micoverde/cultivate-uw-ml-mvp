# Cultivate Learning ML MVP

## Overview
ML-powered feedback system for early educators on child interactions, developed in partnership with Cultivate Learning at University of Washington.

## Mission
Supporting early educators through evidence-based ML feedback to ensure all children have access to high-quality early education experiences.

## Project Goals
- Analyze educator-child interactions from text and video data
- Provide actionable feedback based on peer-reviewed research
- Focus on open-ended questioning patterns and conversational depth
- Create demo-ready system for stakeholder feedback

## Technology Stack
- **ML Backend**: Python, PyTorch (primary), TensorFlow (legacy support)
- **Data Processing**: OpenCV (video), Whisper (audio), HuggingFace Transformers (NLP)
- **Frontend**: React/Next.js for demo interface
- **Cloud Infrastructure**: Azure Container Apps with auto-scaling
- **Research Foundation**: Evidence-based early childhood education practices

## 🧠 ML Model Architecture

### Core Models & Approaches

#### 1. Ensemble ML Architecture (Production)
- **Multi-Model Fusion**: Combines classical ML, deep learning, and hybrid approaches
- **Performance Target**: >90% accuracy on question classification and wait time detection
- **Models**: Random Forest + SVM + Gradient Boosting + Neural Networks
- **Domain**: Educational coaching with real-time feedback capabilities

#### 2. Multi-Modal Processing Pipeline
- **Audio Processing**: Wav2Vec2 (Facebook) + Conformer layers for prosodic analysis
- **Text Processing**: DistilBERT + Educational domain fine-tuning
- **Video Processing**: OpenCV + MediaPipe for gesture/interaction analysis
- **Fusion Architecture**: Cross-modal attention with memory mechanisms

#### 3. Three-Tier Deployment Strategy
```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Edge Tier     │  │  Mobile Tier    │  │   Cloud Tier    │
│   <10ms         │  │   <50ms         │  │   <200ms        │
│ Classical ML    │  │ Lightweight NN  │  │ Full Deep Learning │
│ Random Forest   │  │ MobileBERT      │  │ Multi-Modal Fusion │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Key ML Capabilities

#### Educational Domain Tasks
- **Question Type Classification**: Open-ended vs Closed-ended vs Follow-up vs Rhetorical
- **Wait Time Analysis**: Appropriate pausing vs Insufficient wait vs Interruption patterns
- **CLASS Framework Scoring**: Pedagogical quality assessment (1-5 scale)
- **Conversation Context**: Memory-augmented networks for session-long analysis

#### Advanced Features
- **Adaptive ML/DL Toggle**: Cost-optimized routing between lightweight demos and full ML
- **Fine-Tuning Support**: Gradient descent optimization with educational domain knowledge
- **Synthetic Data Pipeline**: Safe experimentation with reinforcement learning
- **Real-Time Feedback**: AR glasses compatible with <100ms inference latency

### Model Training & Data

#### Training Data Sources
- **119 Expert Annotations**: Professional educator quality assessments
- **26 Educational Videos**: Multi-age group interactions (Toddler, Preschool, Kindergarten)
- **Synthetic Data Generation**: Augmentation for limited dataset scenarios
- **Transfer Learning**: Pre-trained models adapted for educational domain

#### Training Strategies
- **Multi-Task Learning**: Joint optimization across all educational tasks
- **Transfer Learning**: Wav2Vec2 (audio) + DistilBERT (text) foundation models
- **Data Augmentation**: Audio/text augmentation for robustness
- **Human-in-the-Loop RL**: Safe exploration for adaptive coaching strategies

## 🌿 Development Workflow
This project follows a structured git workflow: `feature/fix → dev → main → production`. See [Git Workflow Guide](docs/GIT_WORKFLOW.md) for complete details.

> **Claude Code OAuth Test**: Final verification that Claude Code reviews are working with corrected `github_token` parameter. Workflows fixed to use underscore instead of hyphen.

## Getting Started

### For ML Developers

#### Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd cultivate-uw-ml-mvp

# Install Python dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install HuggingFace transformers
pip install transformers datasets accelerate
```

#### Key Development Commands
```bash
# Start development server
npm run serve  # (runs in background automatically)

# Run ML model training
python src/ml/train_ensemble.py

# Run inference tests
python src/ml/test_inference.py

# Build and test
npm run build && npm run test
```

#### Model Development Workflow
1. **Data Preparation**: Load 119 expert annotations + 26 videos from `data/` directory
2. **Feature Engineering**: Extract audio/text/video features using pipeline in `src/ml/features/`
3. **Model Training**: Use ensemble training scripts in `src/ml/training/`
4. **Evaluation**: Test on validation set with metrics in `src/ml/evaluation/`
5. **Deployment**: Export to ONNX for three-tier deployment

#### Important Notes
- **Never commit real Cultivate Learning data** (privacy policy - use synthetic data only)
- **Always test with real ML models** (no simulation/faking per CLAUDE.md)
- **Follow scenario-based testing** (comprehensive validation before deployment)
- **Branch naming**: Use `fix-{issue-id}-description` format with GitHub issue IDs

### For Other Developers
See individual module documentation for frontend, backend, and infrastructure setup instructions.

## Sprint 1: E2E Demo
Current focus on building end-to-end demonstration ready for feedback from Cultivate Learning stakeholders.

## 📊 Model Performance & Benchmarks

### Current Performance Targets
- **Question Classification**: >85% accuracy (OEQ vs CEQ vs Follow-up vs Rhetorical)
- **Wait Time Detection**: >80% accuracy (Appropriate vs Insufficient vs Interruption)
- **CLASS Framework Correlation**: >75% correlation with expert assessments
- **Inference Latency**: <100ms for real-time AR feedback

### Evaluation Metrics
- **Educational Impact**: 30% increase in open-ended question usage
- **Wait Time Improvement**: 2+ second average increase in appropriate pausing
- **Educator Satisfaction**: >4.0/5.0 rating from user studies
- **Technical Performance**: <10ms edge inference, <200ms cloud inference

## 🔧 Development Resources

### Key Documentation
- **ML Architecture Specs**: See GitHub Issues #29, #76, #104, #187
- **Azure Infrastructure**: Issue #79 (Security & deployment)
- **Testing Strategy**: Issue #80 (Quality gates & validation)
- **Git Workflow**: `docs/GIT_WORKFLOW.md`

### Related SPEC Issues
- [#187: Ensemble ML Model Architecture](../../issues/187) - Advanced multi-model fusion
- [#152: Adaptive ML/DL Toggle with Auto-Scaling](../../issues/152) - Cost-optimized routing
- [#120: Fine-tuning with Gradient Descent](../../issues/120) - Educational domain optimization
- [#118: Synthetic Data and RL](../../issues/118) - Safe experimentation framework

### Development Standards
- **Code Quality**: Lint and typecheck before commits
- **Testing**: Scenario-based validation (comprehensive pre-deployment testing)
- **Security**: No secrets/credentials in code, Azure security gates
- **Privacy**: No real Cultivate Learning data in repositories

## License
MIT - Supporting open access to early education research and tools.