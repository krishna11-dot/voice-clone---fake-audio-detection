
# Voice Cloning & Fake Audio Detection System

**Production-Ready Triple-Layer Detection | NeuTTS Air | 99.3% Accuracy**

---
#System Evolution: I Switched TTS Engines

### Previous Version (Chatterbox TTS)

## 🎯 Quick Links

- **[Live Demo](#)** - Interactive Colab notebook
- **[Full Results](#)** - Complete visualizations  
- **[Source Code](#)** - GitHub repository

---

## 📊 Performance Summary

### Detection Results (October 2025)

| Model | F1-Score | Precision | Recall | AUC |
|-------|----------|-----------|--------|-----|
| **AASIST** | **99.28%** | 98.57% | **100%** | 1.000 |
| **CNN** | **97.87%** | 97.18% | 98.57% | 0.999 |
| **Watermark** | **98.0%** detection rate | - | - | - |

**Dataset:** 1,400 samples (700 real + 700 fake)

**Key Achievement:** AASIST has **100% recall** - catches every single fake sample with zero false negatives.

---

## 🏆 What Makes This Special

### Triple-Layer Security
1. **CNN Detector** - Fast baseline (97.87% accuracy)
2. **AASIST Detector** - Advanced attention-based (99.28% accuracy)  
3. **Watermark Detector** - Perth watermark verification (98% detection)

### Production Features
- Progressive scaling (5 → 700 samples)
- Automated pipeline (no manual intervention)
- Memory efficient (peak 80% usage)
- Comprehensive metrics tracking

---

## 🔧 System Overview

### Voice Cloning
- **Model:** NeuTTS Air from Hugging Face
- **Quality:** Professional-grade voice synthesis
- **Watermark:** Automatic Perth watermark embedding
- **Samples Generated:** 700 fake audio files

### Fake Detection
- **Training Data:** 700 real + 700 fake (perfectly balanced)
- **Models:** CNN + AASIST (trained separately)
- **Verification:** Watermark detection as 3rd layer
- **Result:** 99.28% accuracy with 100% recall

---

## 📈 Detailed Metrics

### AASIST Performance
```
F1-Score:    99.28%
Precision:   98.57%
Recall:      100% ← Perfect! Never misses a fake
Accuracy:    99.29%
AUC:         1.000 ← Perfect separation
```

### CNN Performance
```
F1-Score:    97.87%
Precision:   97.18%
Recall:      98.57%
Accuracy:    97.86%
AUC:         0.999
```

### Watermark Detection
```
Detection Rate:     98.0% (49/50 samples detected)
Average Confidence: 0.818
False Positives:    5% (acceptable for security)
```

### Production Metrics
```
Real-Time Factor:   0.58 (can be optimized to >1.0)
Processing Time:    78 minutes for full pipeline
Memory Peak:        80% (safe threshold <85%)
Hardware:           CUDA GPU (A100)
```

---

## 🎨 Visualizations

The system generates 12 comprehensive charts:

**Dataset & Training:**
1. Progressive scaling chart
2. Dataset composition (50/50 split)
3. CNN performance metrics
4. AASIST performance metrics

**Detection & Comparison:**
5. Hardware comparison (CPU vs GPU)
6. Triple-layer effectiveness
7. CNN vs AASIST comparison
8. Watermark detection rate

**Production & Deployment:**
9. Real-time factor over scaling
10. Time distribution breakdown
11. Memory usage profile
12. Production readiness score (9.0/10)

---

## 💻 Installation

### Requirements
```bash
pip install torch torchaudio librosa
pip install scikit-learn matplotlib seaborn
pip install whisper tqdm neuttsair
```

### For Google Colab
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🚀 Usage

### Quick Test (2 minutes)
```python
from vcfad_system import run_production_quick_test

result = run_production_quick_test()
# Tests: 1 voice clone + detection + watermark verification
```

### Progressive Scaling Test (5 minutes)
```python
from vcfad_system import run_progressive_scaling_test

result = run_progressive_scaling_test(target_samples=100)
# Tests stability: 5 → 10 → 20 → 50 → 100 samples
```

### Full Pipeline (30-40 minutes)
```python
from vcfad_system import run_complete_production_pipeline

results = run_complete_production_pipeline()
# Complete system: 700 samples + training + evaluation
```

---

## 📊 What Do These Numbers Mean?

### F1-Score: 99.28%
**Simple:** Out of 100 audio samples, system correctly identifies 99.

**Technical:** Harmonic mean of precision and recall.

**Industry Standard:** >90% is excellent, >95% is outstanding.

### Recall: 100%
**Simple:** Catches every single fake audio sample. Zero false negatives.

**Why Important:** In security, missing fakes is dangerous. This system never misses.

**Trade-off:** May occasionally flag real audio as fake (~2%).

### Precision: 98.57%
**Simple:** When system says "fake", it's correct 98.57% of time.

**False Positives:** ~1.4% of real audio incorrectly flagged.

**Balance:** High enough for production while maintaining 100% recall.

### AUC: 1.000
**Simple:** Perfect ability to separate real from fake audio.

**Scale:** 0.5 = random guessing, 1.0 = perfect classifier.

**Result:** AASIST achieves perfect separation.

### Real-Time Factor: 0.58
**Simple:** Takes 1.72 seconds to generate 1 second of audio.

**Goal:** RTF > 1.0 for real-time applications.

**Status:** Good for batch processing, can be optimized for real-time.

---

## 🎯 Use Cases

### ✅ Ready For:
- **Content Moderation** - Verify audio submissions in bulk
- **Forensic Analysis** - High-confidence fake detection
- **Security Auditing** - Watermark verification
- **Research** - Benchmark for detection systems
- **API Services** - With load balancing

### ⚠️ Needs Optimization For:
- **Real-Time Streaming** - Need RTF > 1.0
- **Mobile Apps** - Need model quantization
- **Edge Devices** - Consider CNN-only mode

---

## 🔬 Technical Details

### Architecture
```
Phase 1: Voice Cloning
├─ NeuTTS Air TTS
├─ 700 samples generated progressively
└─ Perth watermark embedded automatically

Phase 2: Data Collection
├─ 700 real samples (CommonVoice)
└─ Dataset balanced 50/50

Phase 3: Model Training
├─ CNN trained (1.7 seconds)
├─ AASIST trained (35.5 seconds)
└─ Models trained separately (memory efficient)

Phase 4: Watermark Verification
├─ Test 50 samples
└─ 98% detection rate confirmed

Phase 5: Evaluation
├─ 12 visualization charts
└─ Production metrics calculated
```

### Triple-Layer Decision Logic
```python
# Each layer votes with weighted confidence
votes = {
    'cnn':       35% weight × 0.98 confidence
    'aasist':    35% weight × 0.99 confidence  
    'watermark': 30% weight × 0.82 confidence
}

# Final decision by weighted voting
if (fake_score > real_score):
    return "FAKE"
```

### Technology Stack
- **Voice Synthesis:** NeuTTS Air (Hugging Face)
- **CNN:** 3-layer custom architecture
- **AASIST:** Multi-head attention + graph convolution
- **Watermark:** Perth spectral analysis
- **Framework:** PyTorch 2.0+ with CUDA

---

## 📉 Comparison with Previous Version

| Metric | Old (Chatterbox) | New (NeuTTS Air) | Change |
|--------|------------------|------------------|--------|
| **F1-Score** | 91.0% | 99.28% | +8.28% ✅ |
| **Dataset** | 25 samples | 1,400 samples | 56x larger ✅ |
| **Recall** | Unknown | 100% | Perfect ✅ |
| **Watermark** | None | 98% detection | New feature ✅ |
| **Models** | 1 (RandomForest) | 3 (CNN+AASIST+Watermark) | Triple-layer ✅ |

---

## ⚠️ Known Limitations

### 1. Real-Time Factor (0.58)
- **Issue:** Slightly slower than real-time
- **Impact:** Not suitable for live streaming yet
- **Solution:** Model quantization, GPU optimization

### 2. False Positives (1.4%)
- **Issue:** Some real audio flagged as fake
- **Impact:** ~14 out of 1,000 real samples
- **Solution:** Adjust detection threshold based on use case

### 3. Memory Requirements
- **Issue:** Needs 16GB+ RAM
- **Impact:** Cannot run on low-spec machines
- **Solution:** Implement streaming processing

### 4. Processing Time (78 minutes)
- **Issue:** Full pipeline takes over an hour
- **Impact:** Not instant results
- **Solution:** Parallelize voice cloning

---

## 🔮 Roadmap

### Short-term
- [ ] Optimize to RTF > 1.0
- [ ] Reduce false positive rate to <1%
- [ ] Add Docker container
- [ ] Create REST API

### Long-term
- [ ] Add more TTS engines (Coqui, VALL-E)
- [ ] Mobile deployment (iOS/Android)
- [ ] Multilingual support
- [ ] Real-time streaming detection

---

## 🎓 Research Impact

### Novel Contributions
1. **Triple-layer detection** - First system combining CNN + AASIST + Watermark
2. **100% recall** - Zero false negatives in production setting
3. **Progressive scaling** - Validated approach for stable large-scale generation
4. **Production metrics** - RTF, efficiency tracking for deployment readiness

### Publications
*Paper in preparation*

---

## 📚 Citation
```bibtex
@software{vcfad_neutts_2025,
  title   = {VCFAD: Triple-Layer Fake Audio Detection with NeuTTS Air},
  author  = {Krishna Balachandran Nair},
  year    = {2025},
  url     = {https://github.com/krishna11-dot/voice-clone---fake-audio-detection},
  note    = {F1-Score: 99.28\%, Recall: 100\%}
}
```

---

## 🙏 Acknowledgments

- **NeuTTS Air** by Neuphonic
- **AASIST** architecture researchers
- **TIMIT Dataset** for voice cloning
- **CommonVoice** by Mozilla
- **PyTorch** team

---

## 📄 License

MIT License - Free for academic and commercial use

---

## 📧 Contact

**Krishna Balachandran Nair**

- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- Email: your.email@example.com

---

## ⭐ Support

If this helped your research or project, please:
- ⭐ Star this repository
- 🔄 Share with others
- 📝 Cite in your papers

---

**Status:** Production Ready | Last Updated: January 2025
