# Interview Guide

**Focus**: What is input? What is output? How are you combining? Why are you combining?

---

## The Core Interview Framework

 **clarity about the data pipeline**:

> "The moment you say I am using two models, I would want to know:
> - What is each individual model predicting?
> - What is the input feature?
> - What is the target feature?
> - How are you combining?
> - Why are you combining?"

---

## Your System (Input/Output Clarity)

### Complete Pipeline Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    INPUT: Suspicious Audio File              │
│                         (3 seconds, WAV)                      │
└─────────────┬────────────────────────────────────────────────┘
              │
              ├─────────────────┬────────────────┬──────────────┐
              ▼                 ▼                ▼              │
       [CNN Detector]    [AASIST Detector] [Watermark Detector]│
              │                 │                │              │
    ┌─────────┼─────────┐       │                │              │
    │ INPUT FEATURES    │       │                │              │
    │ ─────────────────│        │                │              │
    │ 30-D vector:      │       │                │              │
    │ • 13 MFCC means   │       │                │              │
    │ • 13 MFCC stds    │       │                │              │
    │ • 4 spectral      │       │                │              │
    └───────────────────┘       │                │              │
              │                 │                │              │
    ┌─────────┼─────────────┐   │                │              │
    │ MODEL ARCHITECTURE   │   │                │              │
    │ ────────────────────│    │                │              │
    │ Conv1D(1→64→128→256)│    │                │              │
    │ + BatchNorm + ReLU  │    │                │              │
    │ FC(768→512→128→2)   │    │                │              │
    └──────────────────────┘   │                │              │
              │                 │                │              │
    ┌─────────▼──────────┐      │                │              │
    │ OUTPUT             │      │                │              │
    │ ───────────────────│      │                │              │
    │ [0.08, 0.92]       │      │                │              │
    │  real   fake       │      │                │              │
    │                    │      │                │              │
    │ Prediction: FAKE   │      │                │              │
    │ Confidence: 92%    │      │                │              │
    └────────────────────┘      │                │              │
                                │                │              │
                       ┌────────┼────────┐       │              │
                       │ INPUT FEATURES  │       │              │
                       │ ────────────────│       │              │
                       │ 64×128 mel-spec │       │              │
                       │ (time×frequency)│       │              │
                       └─────────────────┘       │              │
                                │                │              │
                       ┌────────▼────────────┐   │              │
                       │ MODEL ARCHITECTURE  │   │              │
                       │ ───────────────────│    │              │
                       │ Conv2D (spectral)  │    │              │
                       │ Attention (global) │    │              │
                       │ Conv1D (temporal)  │    │              │
                       │ Attention pooling  │    │              │
                       │ Classifier         │    │              │
                       └────────────────────┘    │              │
                                │                │              │
                       ┌────────▼─────────┐      │              │
                       │ OUTPUT           │      │              │
                       │ ────────────────│       │              │
                       │ [0.05, 0.95]    │       │              │
                       │  real   fake    │       │              │
                       │                 │       │              │
                       │ Prediction: FAKE│       │              │
                       │ Confidence: 95% │       │              │
                       └─────────────────┘       │              │
                                                 │              │
                                        ┌────────┼────────┐     │
                                        │ INPUT FEATURES  │     │
                                        │ ────────────────│     │
                                        │ Audio @ 24kHz   │     │
                                        │ 8-12kHz band    │     │
                                        └─────────────────┘     │
                                                 │              │
                                        ┌────────▼────────────┐ │
                                        │ ALGORITHM           │ │
                                        │ ───────────────────│  │
                                        │ 1. STFT (n_fft=2048)│ │
                                        │ 2. Extract 8-12kHz  │ │
                                        │ 3. Compute features:│ │
                                        │    - Energy ratio   │ │
                                        │    - Flatness       │ │
                                        │    - Variance       │ │
                                        │    - Periodicity    │ │
                                        │    - Alignment      │ │
                                        │ 4. Weighted sum     │ │
                                        └─────────────────────┘ │
                                                 │              │
                                        ┌────────▼─────────┐    │
                                        │ OUTPUT           │    │
                                        │ ────────────────│     │
                                        │ has_watermark: T│     │
                                        │ confidence: 0.82│     │
                                        │                 │     │
                                        │ Prediction: FAKE│     │
                                        │ (Perth detected)│     │
                                        └─────────────────┘     │
                                                                │
              ┌─────────────────────────────────────────────────┘
              ▼
    ┌──────────────────────────────────────────────────────┐
    │             WEIGHTED VOTING (Combining)              │
    │ ───────────────────────────────────────────────────│
    │                                                       │
    │ CNN vote:       0.92 (FAKE) × 0.35 = 0.322          │
    │ AASIST vote:    0.95 (FAKE) × 0.35 = 0.333          │
    │ Watermark vote: 0.82 (FAKE) × 0.30 = 0.246          │
    │                                        ─────          │
    │ FAKE score:  0.322 + 0.333 + 0.246 = 0.901 (90.1%)  │
    │ REAL score:  0                                        │
    │                                                       │
    │ Decision rule: FAKE if score > 0.5                   │
    │ Result: FAKE (unanimous agreement)                   │
    └───────────────────────┬───────────────────────────────┘
                            ▼
               ┌─────────────────────────┐
               │   FINAL OUTPUT          │
               │ ─────────────────────── │
               │ is_fake: True           │
               │ confidence: 90.1%       │
               │ agreement: UNANIMOUS    │
               │ winner: AASIST (95%)    │
               └─────────────────────────┘
```

---

## Key Interview Questions & Answers

### Q1: "What is each model predicting?"

**Answer (Be crystal clear)**:

> "All three models predict the SAME thing: binary classification (real vs fake).
>
> **Input → Output for each**:
>
> **CNN**:
> - Input: 30-D feature vector (MFCC statistics + spectral features)
> - Output: [p_real, p_fake] probabilities
> - Example: [0.08, 0.92] → 92% confidence FAKE
>
> **AASIST**:
> - Input: 64×128 mel-spectrogram (time-frequency image)
> - Output: [p_real, p_fake] probabilities
> - Example: [0.05, 0.95] → 95% confidence FAKE
>
> **Watermark**:
> - Input: Raw audio at 24kHz
> - Output: has_watermark boolean + confidence score
> - Example: has_watermark=True, confidence=0.82 → 82% confidence FAKE
>
> All three answer 'Is this audio fake?' but use DIFFERENT input features."

### Q2: "What are the input features for each model?"

**Answer (Detail the feature engineering)**:

> "Each model uses different input representations:
>
> **CNN - 30-dimensional vector** (hand-crafted features):
> ```
> Feature engineering:
> 1. Load audio at 16kHz
> 2. Compute 13 MFCCs → extract mean and std → 26 features
> 3. Compute spectral centroid → 1 feature
> 4. Compute spectral rolloff → 1 feature
> 5. Compute spectral bandwidth → 1 feature
> 6. Compute zero crossing rate → 1 feature
> Total: 30 features per audio sample
> ```
> WHY these features: MFCCs capture perceptual characteristics, spectral features capture TTS artifacts (unnatural brightness, energy distribution)
>
> **AASIST - 64×128 mel-spectrogram** (learned features):
> ```
> Feature engineering:
> 1. Load audio at 16kHz
> 2. Compute STFT (n_fft=512, hop=256)
> 3. Convert to mel scale (64 bins)
> 4. Log scale: log(1 + mel_spec)
> 5. Normalize: (spec - mean) / std
> 6. Pad/truncate to 128 time steps
> Result: 64×128 image (frequency × time)
> ```
> WHY this representation: Captures both time and frequency information, attention mechanism learns which patterns matter
>
> **Watermark - 5 spectral features** (heuristic):
> ```
> Feature engineering:
> 1. Load audio at 24kHz (higher rate for watermark)
> 2. Compute STFT (n_fft=2048, high resolution)
> 3. Extract 8-12kHz band (where Perth watermark lives)
> 4. Compute 5 features:
>    - Energy ratio (8-12kHz vs total)
>    - Spectral flatness (is it tonal or noisy?)
>    - Temporal variance (does it vary over time?)
>    - Periodicity (repeating patterns?)
>    - Frequency alignment (specific frequencies?)
> 5. Weighted sum → confidence score
> ```
> WHY these features: Perth watermark embedded in high-frequency band, these features detect its signature"

### Q3: "How are you combining the models?"

**Answer (Explain the combination mechanism)**:

> "I use weighted voting - each model contributes to the final decision proportionally.
>
> **Combination formula**:
> ```python
> # Each model outputs probability of FAKE
> cnn_prob = 0.92
> aasist_prob = 0.95
> watermark_prob = 0.82
>
> # Weighted sum
> fake_score = (cnn_prob × 0.35) + (aasist_prob × 0.35) + (watermark_prob × 0.30)
>             = (0.92 × 0.35) + (0.95 × 0.35) + (0.82 × 0.30)
>             = 0.322 + 0.333 + 0.246
>             = 0.901 (90.1% confidence FAKE)
>
> # Decision
> if fake_score > 0.5:
>     prediction = FAKE
> else:
>     prediction = REAL
> ```
>
> **Weights**:
> - CNN: 35% (trained ML model, general-purpose)
> - AASIST: 35% (trained ML model, general-purpose)
> - Watermark: 30% (heuristic, TTS-specific)
>
> **Agreement tracking**:
> - Unanimous (all 3 agree): Highest confidence
> - Majority (2 agree): Medium confidence
> - Split (disagreement): Flag for review"

### Q4: "WHY are you combining them? Why not just use the best model (AASIST)?"

**Answer (Justify the ensemble decision)**:

> "Three reasons: Complementary strengths, redundancy, and speed/accuracy trade-off.
>
> **1. Complementary strengths** - Each model sees different patterns:
> ```
> CNN sees: Statistical patterns in MFCC features
> - Example: Unnatural low variance in pitch (too consistent)
> - Limitation: Local patterns only (15-feature receptive field)
>
> AASIST sees: Long-range patterns in spectrograms
> - Example: Pitch at 0.5s too similar to pitch at 2.3s
> - Limitation: Computationally expensive (150ms inference)
>
> Watermark sees: TTS-specific signatures
> - Example: Perth watermark in 8-12kHz band
> - Limitation: Only works on NeuTTS Air
> ```
>
> **2. Redundancy** - If one fails, others catch it:
> ```
> Scenario: Different TTS engine (not NeuTTS Air)
> - Watermark detector: FAILS (no Perth watermark)
> - CNN detector: WORKS (general TTS artifacts)
> - AASIST detector: WORKS (attention catches patterns)
> Result: 2/3 vote FAKE → Still detects!
> ```
>
> **3. Speed/accuracy trade-off**:
> ```
> CNN alone:  Fast (50ms), Good accuracy (95.74%)
> AASIST alone: Slow (150ms), Better accuracy (98.14%)
> Ensemble: 230ms total, 98.14% accuracy + redundancy
> ```
>
> **Why not just AASIST alone?**
> - Loses redundancy (single point of failure)
> - Loses speed option (can't use CNN for fast screening)
> - Loses TTS-specific detection (watermark catches NeuTTS Air perfectly)"

### Q5: "How did you train the models? Jointly or separately?"

**Answer (Explain training strategy)**:

> "I train CNN and AASIST **separately**, not end-to-end.
>
> **WHY separate training**:
> ```
> Joint training (end-to-end):
> ├─ Memory: CNN (1GB) + AASIST (8GB) + gradients (9GB) = 18GB
> ├─ My GPU: 12GB
> └─ Result: Out of memory error
>
> Separate training:
> ├─ Train CNN: 3.1s, peak 2GB ✓
> ├─ Train AASIST: 22.77s, peak 6GB ✓
> └─ Total: 26s, peak 6GB (fits in GPU!)
> ```
>
> **Training process**:
> ```
> Step 1: Generate 700 fake samples (NeuTTS Air, 67.7 min)
> Step 2: Collect 700 real samples (CommonVoice)
> Step 3: Extract CNN features (30-D vectors, 7.48 min)
> Step 4: Extract AASIST features (64×128 specs, 7.48 min)
> Step 5: Train CNN separately (10 epochs, 3.1s)
> Step 6: Train AASIST separately (10 epochs, 22.77s)
> Step 7: Combine with weighted voting (no training)
> ```
>
> **Input/Output at each step**:
> ```
> CNN Training:
>   Input: 1,120 × 30-D vectors (training set)
>   Target: [0, 1] labels (real=0, fake=1)
>   Output: Trained CNN with 95.74% F1
>
> AASIST Training:
>   Input: 1,120 × 64×128 spectrograms
>   Target: [0, 1] labels (real=0, fake=1)
>   Output: Trained AASIST with 98.14% F1
>
> Ensemble (no training):
>   Input: CNN predictions, AASIST predictions, Watermark predictions
>   Combination: Weighted average (0.35, 0.35, 0.30)
>   Output: Final prediction with 98.14% F1
> ```"

### Q6: "What is your target feature? What are you predicting?"

**Answer (Be specific about the label)**:

> "Binary classification: Real (0) vs Fake (1)
>
> **Label generation**:
> ```
> Real audio (700 samples):
> ├─ Source: CommonVoice dataset
> ├─ Label: 0 (positive class)
> └─ Meaning: Authentic human speech
>
> Fake audio (700 samples):
> ├─ Source: NeuTTS Air voice cloning
> ├─ Generation: encode_reference() + infer() pipeline
> ├─ Label: 1 (negative class)
> └─ Meaning: Synthetic TTS output
> ```
>
> **Output format**:
> ```python
> # Model outputs probabilities
> output = [p_real, p_fake]
>
> # Example
> cnn_output = [0.08, 0.92]
> # Means: 8% probability REAL, 92% probability FAKE
>
> # Final prediction
> prediction = 1 if p_fake > 0.5 else 0
> ```
>
> **Evaluation**:
> ```
> Metric: F1-Score (balance of precision and recall)
> CNN F1: 95.74%
> AASIST F1: 98.14%
> Ensemble F1: 98.14%
> ```"

---

## Common Follow-Up Questions

### Q7: "Why 35-35-30 weights? Did you tune these?"

**Answer (Be honest)**:

> "Design decision based on model characteristics, not exhaustive tuning.
>
> **Reasoning**:
> - **CNN and AASIST**: Both trained ML models with similar F1 (~95-98%)
>   → Give them equal weight (35% each)
>
> - **Watermark**: Heuristic-based, 100% detection on NeuTTS Air but TTS-specific
>   → Slightly lower weight (30%) because it only works on one TTS engine
>
> **If I were to tune systematically**:
> ```
> Grid search on validation set:
> For weights in [(0.33, 0.33, 0.34), (0.35, 0.35, 0.30), (0.40, 0.40, 0.20)]:
>     Evaluate F1-score
>     Pick best
>
> I didn't do this because:
> - Validation F1 with 35-35-30: 98.14% (already excellent)
> - Tuning would likely give <0.5% improvement
> - Not worth the complexity for marginal gain
> ```"

### Q8: "What happens if models disagree?"

**Answer (Show you thought about edge cases)**:

> "The weighted voting handles disagreement naturally. Let me show examples:
>
> **Example 1: Unanimous agreement**
> ```
> CNN: FAKE (92%)
> AASIST: FAKE (95%)
> Watermark: FAKE (82%)
>
> Combined: 0.92×0.35 + 0.95×0.35 + 0.82×0.30 = 0.901 (90.1%)
> Decision: FAKE (high confidence)
> Agreement: UNANIMOUS
> ```
>
> **Example 2: Majority vote**
> ```
> CNN: FAKE (78%)
> AASIST: FAKE (85%)
> Watermark: REAL (only 30% confidence FAKE)
>
> Combined: 0.78×0.35 + 0.85×0.35 + 0.30×0.30 = 0.660 (66%)
> Decision: FAKE (moderate confidence)
> Agreement: MAJORITY (2 out of 3)
> ```
>
> **Example 3: Split decision**
> ```
> CNN: REAL (only 40% FAKE)
> AASIST: FAKE (72%)
> Watermark: REAL (only 20% FAKE)
>
> Combined: 0.40×0.35 + 0.72×0.35 + 0.20×0.30 = 0.452 (45.2%)
> Decision: REAL (low confidence)
> Agreement: SPLIT
>
> In production, I'd flag this for manual review
> (Confidence < 60% = uncertain)
> ```"

### Q9: "How do you extract the MFCC features? Walk me through it."

**Answer (Show you understand the details)**:

> "MFCC extraction step-by-step:
>
> **Process** (using librosa):
> ```python
> import librosa
>
> # Step 1: Load audio
> audio, sr = librosa.load(file_path, sr=16000)
> # Input: WAV file
> # Output: numpy array of samples at 16kHz
>
> # Step 2: Compute MFCCs
> mfccs = librosa.feature.mfcc(
>     y=audio,
>     sr=16000,
>     n_mfcc=13,      # Number of coefficients
>     n_fft=512,      # FFT window size
>     hop_length=256  # Hop between windows
> )
> # Output shape: (13, T) where T = number of time frames
>
> # Step 3: Compute statistics
> mfcc_mean = np.mean(mfccs, axis=1)  # Average over time → 13 features
> mfcc_std = np.std(mfccs, axis=1)    # Std dev over time → 13 features
> # Total: 26 features
>
> # Step 4: Spectral features
> centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=16000))
> rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=16000))
> bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=16000))
> zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
> # Total: 4 features
>
> # Step 5: Concatenate
> features = np.concatenate([mfcc_mean, mfcc_std, [centroid, rolloff, bandwidth, zcr]])
> # Final output: 30-dimensional vector
> ```
>
> **WHY mean AND std**:
> - Mean captures: Average spectral shape (vowel quality, formants)
> - Std captures: Temporal variability (natural pitch variation)
> - Fake audio has LOWER std (TTS is too consistent)"

---

## Interview Success Tips

### 1. Always Frame with Input/Output
❌ **Bad**: "I use CNN and AASIST"
✅ **Good**: "I use CNN which takes 30-D features as input and outputs fake probability, and AASIST which takes 64×128 spectrograms as input and outputs fake probability. I combine them with weighted voting."

### 2. Explain WHY, Not Just WHAT
❌ **Bad**: "I use 64×128 spectrograms"
✅ **Good**: "I use 64×128 spectrograms because 128×256 (standard) only fits batch size 2, which breaks batch normalization. 64×128 enables batch size 8, giving proper batch norm and 98.14% F1 vs 89% with broken batch norm."

### 3. Be Specific About Combining
❌ **Bad**: "I combine them"
✅ **Good**: "I combine using weighted voting: multiply each model's fake probability by its weight (0.35, 0.35, 0.30), sum them, and threshold at 0.5. This gives redundancy - if watermark fails on different TTS, CNN and AASIST still detect."

### 4. Know Your Numbers
- Dataset: 1,400 (700+700), 80/20 split
- CNN: 30-D input, 95.74% F1, 3.1s training, 50ms inference
- AASIST: 64×128 input, 98.14% F1, 22.77s training, 150ms inference
- Weights: 35-35-30
- Voice cloning: 67.7 minutes for 700 samples

### 5. Admit What You Don't Know
If asked about something you didn't measure or test:
✅ **Good**: "I didn't test that. Based on my understanding, I'd expect [X], but I'd need to run that experiment to know for sure."

---

## Summary: 

For ANY system, you should be able to answer:
1. **What is the input?** (Features, format, preprocessing)
2. **What is the output?** (Prediction format, probabilities, decision)
3. **How are you combining?** (Mathematical formula, not vague "ensemble")
4. **Why are you combining?** (What does each model contribute? Why not just one?)
5. **How did you train?** (Separate/joint, optimization, hyperparameters)

**You can now answer all of these clearly for your project!** 🎯
