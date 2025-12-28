# System Architecture

## Table of Contents
1. [High-Level Overview](#high-level-overview)
2. [Data Flow Pipeline](#data-flow-pipeline)
3. [Voice Cloning Pipeline](#voice-cloning-pipeline)
4. [Feature Extraction Architecture](#feature-extraction-architecture)
5. [CNN Architecture Deep Dive](#cnn-architecture-deep-dive)
6. [AASIST Architecture Deep Dive](#aasist-architecture-deep-dive)
7. [Watermark Detection Architecture](#watermark-detection-architecture)
8. [Triple-Layer Voting System](#triple-layer-voting-system)
9. [Training Pipeline](#training-pipeline)
10. [Inference Pipeline](#inference-pipeline)

---

## High-Level Overview

### System Purpose
Three distinct but complementary systems working together:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    VCFAD SYSTEM ARCHITECTURE                          │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   VOICE CLONE   │      │  FAKE DETECTION │      │   WATERMARK     │
│   (NeuTTS Air)  │ ───> │   (CNN+AASIST)  │ <──> │   VERIFICATION  │
│                 │      │                 │      │   (Perth)       │
└─────────────────┘      └─────────────────┘      └─────────────────┘
       │                           │                        │
       │                           │                        │
       ▼                           ▼                        ▼
  700 Fake                    98.14% F1                100% Detection
  Samples                    Accuracy                  on NeuTTS Air
```

**Three Phases:**
1. **Generation**: Create fake audio with NeuTTS Air (includes automatic watermark)
2. **Training**: Train CNN and AASIST on real vs fake audio
3. **Detection**: Triple-layer system validates audio authenticity

---

## Understanding from Dataset Perspective (START HERE!)

### The Datasets Explained Simply

**You have TWO datasets with DIFFERENT purposes**:

```
┌──────────────────────────────────────────────────────────────────┐
│                    YOUR TWO DATASETS                              │
└──────────────────────────────────────────────────────────────────┘

DATASET 1: TIMIT (For Generating FAKE Audio)
Purpose: Voices to clone → Create fake samples
Contents: 630 speakers with audio + transcripts

DATASET 2: CommonVoice (For Collecting REAL Audio)
Purpose: Real human speech → Positive samples for training
Contents: Thousands of real recordings
```

---

### TIMIT Dataset - Detailed Structure

```
TIMIT Dataset Structure:
│
├─ 630 Speakers Total
│  ├─ Male: 438 speakers
│  └─ Female: 192 speakers
│
├─ 8 Dialect Regions (DR1-DR8)
│  ├─ DR1: New England
│  ├─ DR2: Northern
│  ├─ DR3: North Midland
│  ├─ DR4: South Midland
│  ├─ DR5: Southern
│  ├─ DR6: New York City
│  ├─ DR7: Western
│  └─ DR8: Army Brat (moved around)
│
└─ Each Speaker Has:
   ├─ 10 audio files (.WAV)
   └─ 10 transcript files (.TXT)

Example File Structure:
TIMIT/TRAIN/DR1/FCJF0/
  ├─ SA1.WAV  ← Audio recording
  ├─ SA1.TXT  ← "0 46797 She had your dark suit in greasy wash water all year"
  ├─ SA2.WAV  ← Audio recording
  ├─ SA2.TXT  ← Transcript
  ├─ SI1263.WAV
  ├─ SI1263.TXT
  └─ ... (10 files total)

Speaker ID Breakdown:
FCJF0
│││├─ Gender: F = Female, M = Male
││└── Initial: CJF (speaker's initials)
│└─── Unknown middle marker
└──── Number: 0 (speaker instance)
```

**KEY POINT**: TIMIT has BOTH audio (.WAV) AND text (.TXT) - you don't need Whisper to get transcripts!

---

### CommonVoice Dataset - Structure

```
CommonVoice Dataset:
│
├─ Thousands of .mp3 files
├─ Real human speech (crowd-sourced)
├─ Various recording qualities:
│  ├─ Studio quality
│  ├─ Phone recordings
│  ├─ Background noise
│  └─ Different accents
│
└─ Your code samples:
   ├─ High-end GPU: 30,000 files
   ├─ Mid-range GPU: 15,000 files
   └─ Conservative: 10,000 files

   You use: 700 files (for balanced dataset)
```

**Purpose**: Provide REAL audio samples to train detectors on what authentic human speech looks like

---

### Voice Cloning: Step-by-Step from Dataset Perspective

**CRITICAL CLARIFICATION**: Voice gender CAN CHANGE (cross-gender voice cloning)!
- Source text from male speaker → Can generate female voice output
- Source text from female speaker → Can generate male voice output
- Gender determined by TARGET speaker (reference audio), not source text speaker

```
┌──────────────────────────────────────────────────────────────────┐
│          VOICE CLONING FROM TIMIT (Simple Explanation)            │
└──────────────────────────────────────────────────────────────────┘

EXAMPLE: Cloning Speaker FCJF0 (Female, Dialect Region 1)

Step 1: SELECT TIMIT SPEAKER
├─ Your code picks: FCJF0 (Female from New England)
├─ Location: TIMIT/TRAIN/DR1/FCJF0/
└─ Why FCJF0? She's in TIMIT training set

Step 2: LOAD REFERENCE AUDIO + TRANSCRIPT
├─ Audio file: SA1.WAV
│  Contains: FCJF0 saying "She had your dark suit in greasy wash water all year"
│  Duration: ~3 seconds
│  Sample rate: 16kHz
│
└─ Transcript file: SA1.TXT
   Content: "0 46797 She had your dark suit in greasy wash water all year"
            ↑   ↑   ↑
         start end  actual transcript

   Your code reads this and extracts: "She had your dark suit..."

Step 3: ENCODE REFERENCE (NeuTTS Air)
├─ Load NeuTTS Air TTS model
├─ Call: ref_codes = encode_reference("SA1.WAV")
├─ What happens:
│  ├─ NeuTTS Air analyzes FCJF0's voice
│  ├─ Extracts characteristics:
│  │  ├─ Pitch: Female voice (200-300 Hz)
│  │  ├─ Tone: Smooth, clear
│  │  ├─ Rate: Medium speed
│  │  ├─ Accent: New England dialect
│  │  └─ Timbre: Unique voice quality
│  └─ Compresses into ref_codes (tensor)
│
└─ Result: ref_codes = Voice "fingerprint" of FCJF0

Step 4: SELECT NEW TEXT TO SYNTHESIZE
├─ Your code picks NEW sentence (not in TIMIT):
│  "The quick brown fox jumps over the lazy dog"
│
└─ CRITICAL: FCJF0 never said this sentence!
   You're making her voice say something new

Step 5: GENERATE CLONED AUDIO (NeuTTS Air)
├─ Call: cloned_wav = infer(
│           source_text = "The quick brown fox...",  ← NEW text
│           ref_codes = <FCJF0's voice>,             ← From Step 3
│           ref_text = "She had your dark suit..."  ← From SA1.TXT
│        )
│
├─ What NeuTTS Air does internally:
│  ├─ Takes source_text: "The quick brown fox..."
│  ├─ Uses ref_codes: Apply FCJF0's voice characteristics
│  ├─ Uses ref_text: Helps alignment and prosody
│  ├─ Generates mel-spectrogram in FCJF0's voice
│  ├─ Embeds Perth watermark (AUTOMATIC!)
│  └─ Converts to waveform at 24kHz
│
└─ Result: Audio of FCJF0 saying "The quick brown fox..." (FAKE!)

   Gender determined by REFERENCE: FCJF0 is female → Female output ✓
   (If reference was male speaker, output would be male voice)

Step 6: SAVE FAKE AUDIO
├─ Save as: fake_sample_001.wav
├─ Label: negative (fake)
├─ Has Perth watermark: Yes (embedded automatically)
└─ Ready for training detectors

REPEAT 700 TIMES:
└─ Use different TIMIT speakers (male AND female)
   └─ Use different source texts
      └─ Generate 700 diverse fake samples
```

---

### Where Each Tool Is Used (Simple)

```
┌──────────────────────────────────────────────────────────────────┐
│                TOOL USAGE MAP (What, Where, Why)                  │
└──────────────────────────────────────────────────────────────────┘

TOOL 1: NeuTTS Air TTS
├─ WHAT: Text-to-speech engine with voice cloning
├─ WHERE: Voice cloning phase (generating 700 fake samples)
├─ INPUT: source_text + ref_codes + ref_text
├─ OUTPUT: Cloned audio at 24kHz with Perth watermark
├─ WHY: Creates realistic fake audio for training
└─ GENDER: Determined by REFERENCE audio (ref_codes contains gender)

TOOL 2: Whisper
├─ WHAT: Speech-to-text model (transcription)
├─ WHERE: ONLY for quality evaluation (NOT for reference transcripts!)
├─ WHEN: After generating cloned audio
├─ INPUT: cloned_wav (the fake audio you just created)
├─ OUTPUT: Transcribed text
├─ WHY: Check quality - does clone say what you intended?
│  Example:
│    Expected: "The quick brown fox..."
│    Whisper heard: "The quick brown fox..."
│    Match? → Good quality clone ✓
│
└─ NOT USED FOR: Getting TIMIT transcripts (they're already in .TXT files!)

TOOL 3: Librosa
├─ WHAT: Audio processing library
├─ WHERE: Feature extraction for CNN and AASIST
├─ WHEN: After you have 700 fake + 700 real samples
├─ INPUT: Audio files (WAV/MP3)
├─ OUTPUT:
│  ├─ For CNN: 30-D feature vectors (MFCCs + spectral)
│  └─ For AASIST: 64×128 mel-spectrograms
├─ WHY: Convert raw audio to features models can learn from
└─ OPERATIONS:
   ├─ librosa.load() - Load audio at specified sample rate
   ├─ librosa.feature.mfcc() - Extract MFCCs
   └─ librosa.feature.melspectrogram() - Create mel-spectrograms

TOOL 4: PyTorch
├─ WHAT: Deep learning framework
├─ WHERE: Training CNN and AASIST models
├─ WHEN: After feature extraction
├─ INPUT: Features + labels (real=0, fake=1)
├─ OUTPUT: Trained models (cnn_model.pth, aasist_model.pth)
└─ WHY: Learn to distinguish real from fake audio

TOOL 5: SoundFile (sf)
├─ WHAT: Audio file I/O library
├─ WHERE: Saving generated audio files
├─ WHEN: After NeuTTS Air generates cloned audio
├─ INPUT: Audio waveform (numpy array) + sample rate
├─ OUTPUT: .WAV file on disk
└─ WHY: Save fake samples for later use
```

---

### Complete Flow: Dataset → Training → Detection

```
┌──────────────────────────────────────────────────────────────────┐
│              COMPLETE PIPELINE (Simple Explanation)               │
└──────────────────────────────────────────────────────────────────┘

PHASE 1: GENERATE FAKE AUDIO (TIMIT → NeuTTS Air)
═══════════════════════════════════════════════════════════════════

For i = 1 to 700:
│
├─ Pick random TIMIT speaker (e.g., FCJF0, MCPM0, etc.)
│  ├─ Gender distribution: ~50% male, 50% female
│  └─ Dialect distribution: Mix of all 8 regions
│
├─ Load reference audio: speaker/SA1.WAV
├─ Load reference transcript: speaker/SA1.TXT
│
├─ Encode voice: ref_codes = NeuTTS_Air.encode_reference(WAV)
│  └─ Captures speaker's voice characteristics
│
├─ Pick NEW random text (50-150 words)
│  └─ Not from TIMIT - completely new sentences
│
├─ Generate clone: cloned_wav = NeuTTS_Air.infer(
│                     new_text,      ← What to say (NEW!)
│                     ref_codes,     ← How to say it (speaker's voice)
│                     ref_transcript ← Reference text from .TXT
│                  )
│  └─ Perth watermark embedded AUTOMATICALLY
│
├─ (OPTIONAL) Quality check with Whisper:
│  └─ whisper.transcribe(cloned_wav)
│  └─ Does it match what we asked it to say? ✓
│
└─ Save: fake_samples/fake_{i:03d}.wav
   └─ Label: negative (fake)

Result: 700 fake audio files with embedded watermarks
Time: 67.7 minutes (5.8 seconds per sample)

═══════════════════════════════════════════════════════════════════
PHASE 2: COLLECT REAL AUDIO (CommonVoice)
═══════════════════════════════════════════════════════════════════

├─ Load CommonVoice dataset (thousands of real recordings)
├─ Sample 700 random files
│  ├─ Various speakers (not in TIMIT)
│  ├─ Various recording qualities
│  └─ Various accents and speaking styles
│
├─ Resample to 16kHz (standardization)
├─ Trim/pad to 2-3 seconds
│
└─ Save: real_samples/real_{i:03d}.wav
   └─ Label: positive (real)

Result: 700 real audio files
Why CommonVoice? Diverse, real-world audio (not lab recordings)

═══════════════════════════════════════════════════════════════════
PHASE 3: EXTRACT FEATURES (Librosa)
═══════════════════════════════════════════════════════════════════

Dataset now: 1,400 samples (700 real + 700 fake)

For each of 1,400 samples:
│
├─ PATH 1: CNN Features
│  ├─ Load audio at 16kHz (librosa)
│  ├─ Compute 13 MFCCs → mean + std → 26 features
│  ├─ Compute 4 spectral features → 4 features
│  └─ Result: 30-D vector per sample
│     Shape: (1400, 30)
│
└─ PATH 2: AASIST Features
   ├─ Load audio at 16kHz (librosa)
   ├─ Compute mel-spectrogram (64 mels, ~128 time steps)
   ├─ Convert to log scale
   ├─ Normalize (mean=0, std=1)
   └─ Result: 64×128 spectrogram per sample
      Shape: (1400, 64, 128)

Result: Features ready for training
Time: ~7.5 minutes

═══════════════════════════════════════════════════════════════════
PHASE 4: TRAIN DETECTORS (PyTorch)
═══════════════════════════════════════════════════════════════════

STEP 4A: Split Data (80% train, 20% validation)
├─ Train: 1,120 samples (560 real + 560 fake)
└─ Val: 280 samples (140 real + 140 fake)

STEP 4B: Train CNN (SEPARATELY)
├─ Input: 30-D features
├─ Target: Labels (real=0, fake=1)
├─ Batch size: 16
├─ Epochs: 10
├─ Time: 3.1 seconds
└─ Result: cnn_model.pth with 95.74% F1

STEP 4C: Train AASIST (SEPARATELY)
├─ Input: 64×128 spectrograms
├─ Target: Labels (real=0, fake=1)
├─ Batch size: 8
├─ Epochs: 10
├─ Time: 22.77 seconds
└─ Result: aasist_model.pth with 98.14% F1

Why train separately? Memory constraint (together needs 18GB, separately needs 6GB)

═══════════════════════════════════════════════════════════════════
PHASE 5: WATERMARK VERIFICATION
═══════════════════════════════════════════════════════════════════

Test on 50 fake samples:
├─ Extract 8-12kHz frequency band
├─ Compute 5 features (energy, flatness, variance, periodicity, alignment)
├─ Weighted sum → confidence score
└─ Result: 100% detection rate on NeuTTS Air samples

Why 100%? Perth watermark automatically embedded in all fake samples

═══════════════════════════════════════════════════════════════════
PHASE 6: DETECTION (Inference)
═══════════════════════════════════════════════════════════════════

New unknown audio file arrives:
│
├─ Extract CNN features (30-D)
├─ Extract AASIST features (64×128)
├─ Analyze watermark (8-12kHz)
│
├─ CNN prediction: is_fake=True, confidence=0.88
├─ AASIST prediction: is_fake=True, confidence=0.95
├─ Watermark prediction: is_fake=True, confidence=0.82
│
├─ Weighted voting:
│  fake_score = 0.88×0.35 + 0.95×0.35 + 0.82×0.30 = 0.887
│
└─ Final result:
   ├─ is_fake: True
   ├─ confidence: 88.7%
   ├─ agreement: UNANIMOUS (all 3 detected fake)
   └─ winner: AASIST (highest confidence)
```

---

### Key Nuances Summary

**NUANCE 1: Gender determined by REFERENCE speaker, can cross genders**
```
How it works:
- Source text: MWRE0 (Male, DR1) saying "She had your dark suit..."
- Reference audio: FVKB0 (Female, DR2) voice characteristics
- Output: Female voice (FVKB0's voice) saying the text ✓

The code PREFERS different gender/dialect for diversity:
if (target_info['gender'] != source_info['gender'] or
    target_info['dialect'] != source_info['dialect']):
    target_speaker = speaker  # Use this one!

Result: Male text → Female voice (cross-gender cloning)
```

**NUANCE 2: Whisper ONLY for evaluation, NOT for transcription**
```
Where Whisper IS used:
├─ After generating cloned audio
├─ To check: Does clone say what we intended?
└─ Quality assurance (optional)

Where Whisper is NOT used:
├─ Getting TIMIT reference transcripts
│  (Already have .TXT files!)
└─ Feature extraction for models
```

**NUANCE 3: Sequential voice cloning (not parallel)**
```
NeuTTS Air processes ONE sample at a time:
├─ Sample 1 → Generate → Save
├─ Sample 2 → Generate → Save
├─ ... (700 times)
└─ Total: 67.7 minutes

Cannot parallelize: API limitation
Memory cleanup between samples prevents crashes
```

**NUANCE 4: Two datasets, two purposes**
```
TIMIT (630 speakers with transcripts):
└─ Purpose: Generate FAKE audio
   └─ Clone voices → Create fake samples for training

CommonVoice (crowd-sourced real audio):
└─ Purpose: Collect REAL audio
   └─ Provide authentic samples for training

Both needed for balanced dataset (700 real + 700 fake)
```

**NUANCE 5: Perth watermark is AUTOMATIC**
```
When NeuTTS Air generates audio:
├─ Synthesizes speech from text
├─ Automatically embeds Perth watermark
│  (You don't manually add it!)
└─ Saves audio with invisible signature

Detection: 100% on all NeuTTS Air samples
Location: 8-12kHz frequency band (imperceptible to humans)
```

**NUANCE 6: Training is SEPARATE for memory efficiency**
```
Joint training (together):
├─ CNN + AASIST together
├─ Memory needed: 18GB
└─ Result: Out of memory on 12GB GPU ❌

Separate training (your approach):
├─ Train CNN: 2GB peak
├─ Train AASIST: 6GB peak
├─ Memory needed: 6GB max (fits!)
└─ Accuracy loss: <0.1% (negligible)
```

---

## Data Flow Pipeline

### End-to-End Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE DATA FLOW                                 │
└──────────────────────────────────────────────────────────────────────────┘

INPUT: Text Corpus + Real Audio Dataset
  │
  ├──────────────────────────────────────────────────────────────────┐
  │                                                                    │
  ▼                                                                    ▼
┌─────────────────────┐                                    ┌─────────────────────┐
│  VOICE CLONING      │                                    │  REAL AUDIO         │
│  (NeuTTS Air TTS)   │                                    │  (CommonVoice)      │
├─────────────────────┤                                    ├─────────────────────┤
│ • Sequential Gen    │                                    │ • Sample 700 files  │
│ • Progressive Scale │                                    │ • Quality filter    │
│ • Perth Watermark   │                                    │ • 16kHz resample    │
│ • 700 samples       │                                    │ • 2-3 sec clips     │
│ • 67.7 minutes      │                                    │ • Diverse speakers  │
└──────┬──────────────┘                                    └──────┬──────────────┘
       │                                                          │
       │                                                          │
       └──────────────────────┬───────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  DATASET CREATION   │
                    ├─────────────────────┤
                    │ • 700 Real (Label 0)│
                    │ • 700 Fake (Label 1)│
                    │ • Balanced 50/50    │
                    │ • Total: 1400 samples│
                    └──────┬──────────────┘
                           │
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ CNN FEATURES    │ │ AASIST FEATURES │ │ WATERMARK       │
│ (Traditional)   │ │ (Mel-Spec)      │ │ (Spectral)      │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ • 30-D vectors  │ │ • 64×128 images │ │ • 8-12kHz band  │
│ • MFCCs (26)    │ │ • Log mel-scale │ │ • 5 features    │
│ • Spectral (4)  │ │ • Normalized    │ │ • Weighted sum  │
│ • Fast extract  │ │ • Memory opt    │ │ • Frequency     │
│ • 448.45s       │ │ • Batch ready   │ │ • analysis      │
└────┬────────────┘ └────┬────────────┘ └────┬────────────┘
     │                   │                   │
     │                   │                   │
     ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ CNN TRAINING    │ │ AASIST TRAINING │ │ WATERMARK       │
├─────────────────┤ ├─────────────────┤ │ VALIDATION      │
│ • Batch: 16     │ │ • Batch: 4      │ ├─────────────────┤
│ • Epochs: 10    │ │ • Epochs: 10    │ │ • Test 50 samples│
│ • LR: 0.001     │ │ • LR: 0.001     │ │ • Threshold: 0.65│
│ • Time: 3.1s    │ │ • Time: 22.77s  │ │ • Confidence calc│
│ • F1: 95.74%    │ │ • F1: 98.14%    │ │ • Rate: 100%    │
└────┬────────────┘ └────┬────────────┘ └────┬────────────┘
     │                   │                   │
     │                   │                   │
     └───────────────────┼───────────────────┘
                         │
                         ▼
                ┌─────────────────────┐
                │  TRIPLE-LAYER       │
                │  VOTING SYSTEM      │
                ├─────────────────────┤
                │ • CNN: 35% weight   │
                │ • AASIST: 35% weight│
                │ • Watermark: 30% wt │
                │ • Soft voting       │
                │ • Confidence fusion │
                └──────┬──────────────┘
                       │
                       ▼
                ┌─────────────────────┐
                │  FINAL PREDICTION   │
                ├─────────────────────┤
                │ • is_fake: bool     │
                │ • confidence: 0-1   │
                │ • agreement: level  │
                │ • winner: detector  │
                └─────────────────────┘

OUTPUT: Detection results + confidence + explainability
```

---

## Voice Cloning Pipeline

### Progressive Scaling Strategy

**Why progressive?** Stability. Generating all 700 samples at once crashes. Progressive scaling prevents memory overflow.

```
┌──────────────────────────────────────────────────────────────────┐
│             PROGRESSIVE VOICE CLONING PIPELINE                    │
└──────────────────────────────────────────────────────────────────┘

Stage 1: Initialize
  │
  ├─ Load NeuTTS Air model (10GB download)
  ├─ Initialize Perth watermarker
  ├─ Select diverse text samples from corpus
  └─ Set target: 700 samples
  │
  ▼
Stage 2: Progressive Scaling
  │
  ├─ Batch 1:    5 samples   (stability test)      ✓
  ├─ Batch 2:   10 samples   (2x scale)            ✓
  ├─ Batch 3:   20 samples   (2x scale)            ✓
  ├─ Batch 4:   50 samples   (2.5x scale)          ✓
  ├─ Batch 5:  100 samples   (2x scale)            ✓
  ├─ Batch 6:  200 samples   (2x scale)            ✓
  ├─ Batch 7:  400 samples   (2x scale)            ✓
  └─ Batch 8:  700 samples   (1.75x scale)         ✓
  │
  │  Why this progression?
  │  • Exponential growth catches memory issues early
  │  • Each stage validates system stability
  │  • Total time: 67.7 minutes
  │  • Memory cleanup between batches
  │
  ▼
Stage 3: Per-Sample Generation (Sequential, Reference-Based)
  │
  For each sample:
  │
  ├─ Select TIMIT reference speaker (e.g., FCJF0)
  ├─ Load reference audio (.WAV) + transcript (.TXT)
  │  └─ ref_text = read from .TXT file (skip timing info)
  ├─ Encode reference voice characteristics
  │  └─ ref_codes = encode_reference(audio_path)
  ├─ Select NEW text to synthesize (50-150 words)
  ├─ NeuTTS Air reference-based synthesis
  │  └─ cloned_wav = infer(source_text, ref_codes, ref_text)
  │     ├─ Uses ref_codes for voice characteristics
  │     ├─ Synthesizes source_text in that voice
  │     └─ Perth watermark embedding (automatic)
  ├─ Save as 24kHz WAV file
  ├─ Memory cleanup (torch.cuda.empty_cache())
  └─ Progress tracking
  │
  ▼
Stage 4: Validation
  │
  ├─ Check: 700 files created?
  ├─ Check: Average file size ~1-2MB?
  ├─ Check: No silent files?
  ├─ Check: Duration 2-3 seconds?
  └─ Check: Watermark detectable?
  │
  ▼
Stage 5: Output
  │
  └─ 700 fake audio files
     └─ All with embedded Perth watermark
        └─ Ready for feature extraction

┌────────────────────────────────────────────────────────────────┐
│  Key Design Choice: Sequential vs Parallel                     │
├────────────────────────────────────────────────────────────────┤
│  Sequential (Chosen):                                          │
│    ✓ Predictable memory usage (1 sample at a time)            │
│    ✓ No crashes due to memory spikes                          │
│    ✓ 67.7 min total time (acceptable)                         │
│    ✗ Not using full GPU capacity                              │
│                                                                │
│  Parallel (Rejected):                                          │
│    ✓ Could be 10x faster                                      │
│    ✗ Memory spikes cause OOM errors                           │
│    ✗ Complex error recovery                                   │
│    ✗ Watermarking interference (concurrent writes)            │
└────────────────────────────────────────────────────────────────┘
```

---

## Feature Extraction Architecture

### Dual-Path Feature Extraction

**Why two paths?** CNN and AASIST need different feature representations.

```
┌──────────────────────────────────────────────────────────────────┐
│                  FEATURE EXTRACTION PIPELINE                      │
└──────────────────────────────────────────────────────────────────┘

INPUT: Audio file (WAV, 2-3 seconds)
  │
  ├─────────────────────────────┬────────────────────────────┐
  │                             │                            │
  ▼                             ▼                            ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   PATH 1: CNN    │  │  PATH 2: AASIST  │  │  PATH 3: MARK    │
│   (Traditional)  │  │  (Deep Learning) │  │  (Watermark)     │
└──────────────────┘  └──────────────────┘  └──────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  PATH 1: CNN FEATURES (30-dimensional vector)                    │
└──────────────────────────────────────────────────────────────────┘

Audio File (WAV)
  │
  ├─ librosa.load(sr=16000)  ← Resample to 16kHz
  │
  ▼
Signal (16000 samples/sec)
  │
  ├─────────────────────┬─────────────────┬──────────────────┐
  │                     │                 │                  │
  ▼                     ▼                 ▼                  ▼
┌──────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
│  MFCCs   │  │  Spectral    │  │  Spectral   │  │  Zero Cross  │
│          │  │  Centroid    │  │  Rolloff    │  │  Rate        │
├──────────┤  ├──────────────┤  ├─────────────┤  ├──────────────┤
│ n_fft:   │  │ Mean over    │  │ Mean over   │  │ Mean over    │
│   1024   │  │ time         │  │ time        │  │ time         │
│ hop:     │  │              │  │             │  │              │
│   512    │  │              │  │             │  │              │
│ n_mfcc:  │  │              │  │             │  │              │
│   13     │  │              │  │             │  │              │
└────┬─────┘  └──────┬───────┘  └──────┬──────┘  └──────┬───────┘
     │               │                 │                 │
     ▼               ▼                 ▼                 ▼
┌─────────┐    ┌─────────┐      ┌─────────┐      ┌─────────┐
│ 13 mean │    │    1    │      │    1    │      │    1    │
│ 13 std  │    │ feature │      │ feature │      │ feature │
└────┬────┘    └────┬────┘      └────┬────┘      └────┬────┘
     │              │                 │                 │
     └──────────────┼─────────────────┼─────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │ 30-D Vector   │
            │ (26+1+1+1+1)  │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │ StandardScaler│
            │ Normalization │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │ Ready for CNN │
            │ Input: (30,)  │
            └───────────────┘


┌──────────────────────────────────────────────────────────────────┐
│  PATH 2: AASIST FEATURES (64×128 mel-spectrogram)               │
└──────────────────────────────────────────────────────────────────┘

Audio File (WAV)
  │
  ├─ librosa.load(sr=16000)  ← Resample to 16kHz
  │
  ▼
Signal (16000 samples/sec)
  │
  ├─ librosa.feature.melspectrogram()
  │  ├─ n_fft: 512        ← FFT window size
  │  ├─ hop_length: 256   ← Step size
  │  ├─ n_mels: 64        ← Frequency bins
  │  └─ sr: 16000         ← Sample rate
  │
  ▼
Mel-Spectrogram (Power scale)
  │  Shape: (64, variable_time_steps)
  │
  ├─ librosa.power_to_db()  ← Convert to decibel scale
  │
  ▼
Log Mel-Spectrogram
  │
  ├─ Pad or truncate to 128 time steps
  │  ├─ If shorter: Zero-pad on right
  │  └─ If longer: Truncate
  │
  ▼
Fixed-Size Spectrogram (64×128)
  │
  ├─ Per-sample normalization:
  │  ├─ mean = spectrogram.mean()
  │  ├─ std = spectrogram.std()
  │  └─ normalized = (spec - mean) / (std + 1e-8)
  │
  ▼
Normalized Spectrogram
  │  Shape: (1, 64, 128)  ← Add channel dimension
  │
  ▼
┌───────────────────┐
│ Ready for AASIST  │
│ Input: (1,64,128) │
└───────────────────┘


┌──────────────────────────────────────────────────────────────────┐
│  PATH 3: WATERMARK FEATURES (5-dimensional vector)               │
└──────────────────────────────────────────────────────────────────┘

Audio File (WAV)
  │
  ├─ Load at 24kHz (NeuTTS Air native rate)
  │
  ▼
Signal (24000 samples/sec)
  │
  ├─ STFT (Short-Time Fourier Transform)
  │  ├─ Window: 2048 samples
  │  ├─ Hop: 512 samples
  │  └─ Output: Frequency×Time matrix
  │
  ▼
Spectrogram (frequency domain)
  │
  ├─ Extract 8-12 kHz band
  │  └─ Watermark region (imperceptible to humans)
  │
  ▼
High-Frequency Band Analysis
  │
  ├──────────────┬─────────────┬──────────────┬──────────────┐
  │              │             │              │              │
  ▼              ▼             ▼              ▼              ▼
┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Energy  │ │ Spectral │ │ Temporal │ │Periodicity│ │Frequency │
│ Ratio   │ │ Flatness │ │ Variance │ │  Score   │ │Alignment │
├─────────┤ ├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤
│ Energy  │ │ Tonality │ │ Consist- │ │ Auto-    │ │ Peak freq│
│ 8-12kHz │ │ measure  │ │ ency over│ │ correla- │ │ location │
│ vs total│ │          │ │ time     │ │ tion     │ │ match    │
└────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
     │ 0.25      │ 0.15       │ 0.20       │ 0.20       │ 0.20
     │           │            │            │            │
     └───────────┼────────────┼────────────┼────────────┘
                 │
                 ▼
         ┌───────────────┐
         │ Weighted Sum  │
         │ Confidence    │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │ Threshold 0.65│
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │ has_watermark │
         │ (bool + conf) │
         └───────────────┘
```

---

## CNN Architecture Deep Dive

### Layer-by-Layer Data Transformation

```
┌──────────────────────────────────────────────────────────────────┐
│              CNN ARCHITECTURE (Standard Config)                   │
└──────────────────────────────────────────────────────────────────┘

INPUT: (batch_size, 30)
  │   30 features: [MFCC_mean(13), MFCC_std(13), spectral(4)]
  │
  ├─ Reshape: (batch_size, 1, 30)  ← Add channel dimension for Conv1D
  │
  ▼
┌───────────────────────────────────────────────────────────────────┐
│  BLOCK 1: Low-Level Feature Detection                             │
└───────────────────────────────────────────────────────────────────┘
│
├─ Conv1D(in_channels=1, out_channels=64, kernel_size=3, padding=1)
│  │
│  │  What it learns: Basic spectral patterns
│  │  Examples: "MFCC spike", "zero-crossing cluster"
│  │  Receptive field: 3 features at a time
│  │
│  ▼  Shape: (batch, 64, 30)  ← 64 feature maps, same length
│
├─ BatchNorm1d(64)
│  │  Normalizes across batch: mean=0, std=1 per channel
│  │  Why: Stabilizes training, allows higher learning rate
│  ▼  Shape: (batch, 64, 30)
│
├─ ReLU()
│  │  Activation: max(0, x)
│  │  Why: Non-linearity, sparse activation
│  ▼  Shape: (batch, 64, 30)
│
├─ MaxPool1d(kernel_size=2)
│  │  Takes max value in each window of 2
│  │  Why: Downsampling, translation invariance
│  ▼  Shape: (batch, 64, 15)  ← Length halved
│
▼

┌───────────────────────────────────────────────────────────────────┐
│  BLOCK 2: Mid-Level Feature Composition                           │
└───────────────────────────────────────────────────────────────────┘
│
├─ Conv1D(in_channels=64, out_channels=128, kernel_size=3, padding=1)
│  │
│  │  What it learns: Combinations of low-level patterns
│  │  Examples: "MFCC spike + low zero-crossing = voiced segment"
│  │  Receptive field: 7 original features (3×2 + overlap)
│  │
│  ▼  Shape: (batch, 128, 15)  ← 128 feature maps
│
├─ BatchNorm1d(128)
│  ▼  Shape: (batch, 128, 15)
│
├─ ReLU()
│  ▼  Shape: (batch, 128, 15)
│
├─ MaxPool1d(kernel_size=2)
│  ▼  Shape: (batch, 128, 7)  ← Length halved again
│
▼

┌───────────────────────────────────────────────────────────────────┐
│  BLOCK 3: High-Level Semantic Features                            │
└───────────────────────────────────────────────────────────────────┘
│
├─ Conv1D(in_channels=128, out_channels=256, kernel_size=3, padding=1)
│  │
│  │  What it learns: Abstract patterns indicating real/fake
│  │  Examples: "Unnatural prosody signature", "TTS artifact pattern"
│  │  Receptive field: 15 original features
│  │
│  ▼  Shape: (batch, 256, 7)
│
├─ BatchNorm1d(256)
│  ▼  Shape: (batch, 256, 7)
│
├─ ReLU()
│  ▼  Shape: (batch, 256, 7)
│
├─ MaxPool1d(kernel_size=2)
│  ▼  Shape: (batch, 256, 3)  ← Length halved (floor division)
│
▼

┌───────────────────────────────────────────────────────────────────┐
│  FLATTEN & FULLY CONNECTED LAYERS                                 │
└───────────────────────────────────────────────────────────────────┘
│
├─ Flatten()
│  │  Convert 3D tensor to 2D
│  ▼  Shape: (batch, 256×3) = (batch, 768)
│
├─ Linear(768, 512)
│  │  Fully connected layer
│  │  What it learns: Holistic combinations of all features
│  ▼  Shape: (batch, 512)
│
├─ ReLU()
│  ▼  Shape: (batch, 512)
│
├─ Dropout(p=0.5)
│  │  Randomly zeros 50% of activations
│  │  Why: Prevents overfitting, forces redundancy
│  ▼  Shape: (batch, 512)
│
├─ Linear(512, 128)
│  │  Dimensionality reduction
│  ▼  Shape: (batch, 128)
│
├─ ReLU()
│  ▼  Shape: (batch, 128)
│
├─ Dropout(p=0.5)
│  ▼  Shape: (batch, 128)
│
├─ Linear(128, 2)
│  │  Final classification layer
│  │  2 outputs: [real_score, fake_score]
│  ▼  Shape: (batch, 2)
│
▼

OUTPUT: Logits (batch, 2)
  │
  ├─ During Training: CrossEntropyLoss(logits, labels)
  │                   ├─ Applies softmax internally
  │                   └─ Computes negative log likelihood
  │
  └─ During Inference: Softmax(logits)
                      ├─ Converts to probabilities
                      └─ [P(real), P(fake)]

┌───────────────────────────────────────────────────────────────────┐
│  Parameter Count Breakdown (Standard Config)                      │
├───────────────────────────────────────────────────────────────────┤
│  Conv Block 1:  1×64×3 + 64 = 256 params                          │
│  Conv Block 2:  64×128×3 + 128 = 24,704 params                    │
│  Conv Block 3:  128×256×3 + 256 = 98,560 params                   │
│  FC Layer 1:    768×512 + 512 = 393,728 params                    │
│  FC Layer 2:    512×128 + 128 = 65,664 params                     │
│  FC Layer 3:    128×2 + 2 = 258 params                            │
│  ────────────────────────────────────────────────────────          │
│  TOTAL: ~583,000 parameters                                       │
└───────────────────────────────────────────────────────────────────┘
```

**Key Design Insight:**
- Input 30 → Conv layers reduce to 3 → FC layers classify
- Each MaxPool halves spatial dimension, allowing channel doubling
- Receptive field grows: 3 → 7 → 15 features (half the input)
- BatchNorm + Dropout = regularization at different levels

---

## AASIST Architecture Deep Dive

### Multi-Stage Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│           AASIST ARCHITECTURE (Mid-Range Config)                  │
└──────────────────────────────────────────────────────────────────┘

INPUT: (batch, 1, 64, 128)
  │   1 channel, 64 mel bins, 128 time steps
  │
  ▼

┌───────────────────────────────────────────────────────────────────┐
│  STAGE 1: Spectral Convolution (Local Pattern Extraction)         │
└───────────────────────────────────────────────────────────────────┘
│
├─ Conv2D(1 → 32, kernel=(3,3), padding=1)
│  │  Learns: Local time-frequency patterns
│  │  Examples: Formant edges, harmonic structures
│  ▼  Shape: (batch, 32, 64, 128)
│
├─ BatchNorm2d(32) → ReLU()
│  ▼  Shape: (batch, 32, 64, 128)
│
├─ Conv2D(32 → 64, kernel=(3,3), padding=1)
│  │  Learns: Combinations of local patterns
│  ▼  Shape: (batch, 64, 64, 128)
│
├─ BatchNorm2d(64) → ReLU()
│  ▼  Shape: (batch, 64, 64, 128)
│
├─ MaxPool2d(kernel_size=2)
│  │  Downsamples both dimensions
│  ▼  Shape: (batch, 64, 32, 64)  ← Halved H and W
│
├─ Conv2D(64 → 128, kernel=(3,3), padding=1)
│  │  Learns: Higher-level spectral features
│  ▼  Shape: (batch, 128, 32, 64)
│
├─ BatchNorm2d(128) → ReLU()
│  ▼  Shape: (batch, 128, 32, 64)
│
├─ MaxPool2d(kernel_size=2)
│  ▼  Shape: (batch, 128, 16, 32)  ← Halved again
│
▼

┌───────────────────────────────────────────────────────────────────┐
│  STAGE 2: Graph Attention (Global Relationship Modeling)          │
└───────────────────────────────────────────────────────────────────┘
│
├─ Reshape: (batch, 128, 16, 32) → (batch, 16×32, 128)
│  │  Convert to sequence: 512 positions, 128-D features each
│  ▼  Shape: (batch, 512, 128)
│
├─ MultiheadAttention(embed_dim=128, num_heads=4)
│  │
│  │  How it works:
│  │  ┌─────────────────────────────────────────────┐
│  │  │ For each position in spectrogram:          │
│  │  │   Query: "What am I looking for?"          │
│  │  │   Key: "What do I contain?"                │
│  │  │   Value: "What should I contribute?"       │
│  │  │                                            │
│  │  │ Attention weight = softmax(Q·K^T / √d)     │
│  │  │ Output = Σ(attention_weight × Value)       │
│  │  └─────────────────────────────────────────────┘
│  │
│  │  Why 4 heads?
│  │  - Each head learns different relationships
│  │  - Head 1: Pitch contour (F0) patterns
│  │  - Head 2: Formant transitions
│  │  - Head 3: Energy dynamics
│  │  - Head 4: Harmonic correlations
│  │
│  │  TTS Detection: Fake audio has unnatural long-range correlations
│  │  Example: Pitch at time 10 too similar to pitch at time 100
│  │
│  ▼  Shape: (batch, 512, 128)
│
├─ Reshape: (batch, 512, 128) → (batch, 128, 16, 32)
│  │  Convert back to spatial
│  ▼  Shape: (batch, 128, 16, 32)
│
▼

┌───────────────────────────────────────────────────────────────────┐
│  STAGE 3: Temporal Convolution (Time Dynamics)                    │
└───────────────────────────────────────────────────────────────────┘
│
├─ Flatten spatial: (batch, 128, 16, 32) → (batch, 128, 512)
│  │  Collapse frequency dimension, keep time
│  ▼  Shape: (batch, 128, 512)
│
├─ Conv1D(128 → 256, kernel=3, padding=1)
│  │  Learns: Temporal evolution of spectral patterns
│  │  Examples: "Formant transition too abrupt", "Unnatural prosody"
│  ▼  Shape: (batch, 256, 512)
│
├─ BatchNorm1d(256) → ReLU()
│  ▼  Shape: (batch, 256, 512)
│
├─ Conv1D(256 → 128, kernel=3, padding=1)
│  │  Dimensionality reduction
│  ▼  Shape: (batch, 128, 512)
│
├─ BatchNorm1d(128) → ReLU()
│  ▼  Shape: (batch, 128, 512)
│
▼

┌───────────────────────────────────────────────────────────────────┐
│  STAGE 4: Attention Pooling (Temporal Aggregation)                │
└───────────────────────────────────────────────────────────────────┘
│
├─ Transpose: (batch, 128, 512) → (batch, 512, 128)
│  │  Time steps as sequence
│  ▼  Shape: (batch, 512, 128)
│
├─ Linear(128 → 64)
│  ▼  Shape: (batch, 512, 64)
│
├─ Tanh()
│  │  Non-linearity
│  ▼  Shape: (batch, 512, 64)
│
├─ Linear(64 → 1)
│  │  Compute attention weight for each time step
│  ▼  Shape: (batch, 512, 1)
│
├─ Softmax(dim=1)
│  │  Normalize weights across time: Σ weights = 1
│  │  Learns: Which time steps are most informative
│  ▼  Shape: (batch, 512, 1)
│
├─ Weighted sum: Σ(attention_weight × features)
│  │  Aggregate features from all time steps
│  ▼  Shape: (batch, 128)
│
▼

┌───────────────────────────────────────────────────────────────────┐
│  STAGE 5: Classification                                           │
└───────────────────────────────────────────────────────────────────┘
│
├─ Linear(128 → 64)
│  ▼  Shape: (batch, 64)
│
├─ ReLU() → Dropout(0.3)
│  ▼  Shape: (batch, 64)
│
├─ Linear(64 → 32)
│  ▼  Shape: (batch, 32)
│
├─ ReLU() → Dropout(0.3)
│  ▼  Shape: (batch, 32)
│
├─ Linear(32 → 2)
│  ▼  Shape: (batch, 2)
│
▼

OUTPUT: Logits (batch, 2)
  │
  └─ [real_score, fake_score]


┌───────────────────────────────────────────────────────────────────┐
│  Information Flow Summary                                          │
├───────────────────────────────────────────────────────────────────┤
│  Input:              64×128 spectrogram                            │
│  After Conv2D:       16×32 feature map (4x downsampled)            │
│  After Attention:    512 positions with global context             │
│  After Temporal:     Temporal dynamics extracted                   │
│  After Pooling:      Single 128-D vector per sample                │
│  After Classifier:   2 scores (real vs fake)                       │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│  Why This Multi-Stage Design?                                     │
├───────────────────────────────────────────────────────────────────┤
│  1. Conv2D: Extracts local patterns (formants, harmonics)         │
│  2. Attention: Relates distant patterns (pitch consistency)       │
│  3. Conv1D: Models temporal evolution (prosody, rhythm)           │
│  4. Pooling: Focuses on informative time regions                  │
│  5. Classifier: Combines everything into final decision           │
│                                                                    │
│  Each stage captures different aspects of TTS artifacts:          │
│  - Spatial: Unnatural formant structures                          │
│  - Global: Too-consistent pitch/energy                            │
│  - Temporal: Abrupt transitions                                   │
│  - Aggregation: Suspicious patterns at specific times             │
└───────────────────────────────────────────────────────────────────┘
```

---

## Watermark Detection Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│              WATERMARK DETECTION PIPELINE                         │
└──────────────────────────────────────────────────────────────────┘

INPUT: Audio File (24kHz WAV)
  │
  ├─ Load waveform
  │
  ▼
STFT (Short-Time Fourier Transform)
  │
  ├─ Window size: 2048 samples (85ms at 24kHz)
  ├─ Hop length: 512 samples (21ms)
  ├─ Output: Complex spectrogram
  │
  ▼
Power Spectrum: |STFT|²
  │  Shape: (frequency_bins, time_frames)
  │
  ├─ Extract 8-12 kHz band
  │  ├─ Bin calculation: freq_bins = [8000*2048/24000 : 12000*2048/24000]
  │  └─ ~683 to 1024 bins
  │
  ▼
High-Frequency Band (8-12 kHz)
  │  This is where Perth watermark lives
  │
  ├─────────┬─────────┬─────────┬─────────┬─────────┐
  │         │         │         │         │         │
  ▼         ▼         ▼         ▼         ▼         ▼

┌─────────────────────────────────────────────────────────────────────┐
│  FEATURE 1: Energy Ratio (Weight: 0.25)                             │
└─────────────────────────────────────────────────────────────────────┘

Total Energy = Σ |spectrum|²
Energy_8_12k = Σ |spectrum[8-12kHz]|²
Ratio = Energy_8_12k / Total Energy

WHY: NeuTTS Air has elevated high-frequency energy
Real speech: Ratio ~ 0.05-0.10 (5-10%)
Fake speech: Ratio ~ 0.15-0.25 (15-25%)

Normalized Score = (Ratio - 0.10) / 0.15
Clipped to [0, 1]


┌─────────────────────────────────────────────────────────────────────┐
│  FEATURE 2: Spectral Flatness (Weight: 0.15)                        │
└─────────────────────────────────────────────────────────────────────┘

Geometric Mean = exp(mean(log(spectrum)))
Arithmetic Mean = mean(spectrum)
Flatness = Geometric Mean / Arithmetic Mean

WHY: Measures tonality vs noisiness
Flatness = 1: White noise (totally flat spectrum)
Flatness = 0: Pure tone (spike in spectrum)

Real speech: Higher flatness (more noise-like)
Fake speech: Lower flatness (more tonal due to vocoder)

Normalized Score = 1 - Flatness
Clipped to [0, 1]


┌─────────────────────────────────────────────────────────────────────┐
│  FEATURE 3: Temporal Variance (Weight: 0.20)                        │
└─────────────────────────────────────────────────────────────────────┘

Energy_per_frame = Σ |spectrum[8-12kHz, frame_t]|²
Variance = std(Energy_per_frame) / mean(Energy_per_frame)

WHY: Consistency of watermark over time
Real speech: High variance (variable phoneme energy)
Fake speech: Low variance (watermark is constant)

TTS has more uniform energy distribution in 8-12kHz

Normalized Score = 1 - Variance
Clipped to [0, 1]


┌─────────────────────────────────────────────────────────────────────┐
│  FEATURE 4: Periodicity Score (Weight: 0.20)                        │
└─────────────────────────────────────────────────────────────────────┘

Signal = Energy_per_frame from 8-12kHz band
Autocorrelation = correlation(Signal, Signal shifted by lag)

For lags 1 to 100:
    Compute autocorrelation

Periodicity = max(Autocorrelation[lags 5-50])

WHY: Detects repetitive patterns
Real speech: Low autocorrelation (irregular)
Fake speech: High autocorrelation (vocoder periodicity)

Perth watermark has subtle periodic structure

Normalized Score = Periodicity
Clipped to [0, 1]


┌─────────────────────────────────────────────────────────────────────┐
│  FEATURE 5: Frequency Alignment (Weight: 0.20)                      │
└─────────────────────────────────────────────────────────────────────┘

Mean_Spectrum = mean(spectrum[8-12kHz], axis=time)
Peak_Frequency = argmax(Mean_Spectrum)

Optimal_Frequency = 9500 Hz (NeuTTS Air signature)
Tolerance = 2500 Hz

Distance = |Peak_Frequency - Optimal_Frequency|
Alignment = max(0, 1 - Distance / Tolerance)

WHY: NeuTTS Air has characteristic peak at ~9.5kHz
Real speech: Peak is random in 8-12kHz
Fake speech: Peak near 9.5kHz

Normalized Score = Alignment
Clipped to [0, 1]


┌─────────────────────────────────────────────────────────────────────┐
│  FINAL CONFIDENCE CALCULATION                                       │
└─────────────────────────────────────────────────────────────────────┘

Confidence = 0.25 × Energy_Ratio +
             0.15 × Spectral_Flatness +
             0.20 × Temporal_Variance +
             0.20 × Periodicity +
             0.20 × Frequency_Alignment

Result: Score ∈ [0, 1]

if Confidence > 0.65:
    has_watermark = True
else:
    has_watermark = False

return {
    'has_watermark': has_watermark,
    'confidence': Confidence,
    'features': {
        'energy_ratio': ...,
        'spectral_flatness': ...,
        'temporal_variance': ...,
        'periodicity': ...,
        'frequency_alignment': ...
    }
}


┌─────────────────────────────────────────────────────────────────────┐
│  Why 5 Features? Why These Specific Ones?                           │
├─────────────────────────────────────────────────────────────────────┤
│  Solo Performance (Tested individually):                            │
│    Energy Ratio:         78% accuracy  ← Most informative           │
│    Temporal Variance:    72% accuracy                               │
│    Periodicity:          71% accuracy                               │
│    Frequency Alignment:  70% accuracy                               │
│    Spectral Flatness:    65% accuracy  ← Weakest but still useful   │
│                                                                      │
│  Combined Performance: 100% detection on NeuTTS Air samples         │
│                                                                      │
│  Complementary Nature:                                              │
│    - Energy: Overall watermark strength                             │
│    - Flatness: Spectral shape (different aspect)                    │
│    - Variance: Temporal consistency                                 │
│    - Periodicity: Repetitive patterns                               │
│    - Alignment: TTS-specific signature                              │
│                                                                      │
│  Each feature captures different artifact manifestation             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Triple-Layer Voting System

```
┌──────────────────────────────────────────────────────────────────┐
│              TRIPLE-LAYER ENSEMBLE VOTING                         │
└──────────────────────────────────────────────────────────────────┘

INPUT: Audio sample to classify
  │
  ├───────────────────┬─────────────────────┬──────────────────┐
  │                   │                     │                  │
  ▼                   ▼                     ▼                  ▼

┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ LAYER 1: CNN │  │ LAYER 2:     │  │ LAYER 3:     │
│              │  │ AASIST       │  │ WATERMARK    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       │                 │                 │
       ▼                 ▼                 ▼

┌──────────────────────────────────────────────────────────────────┐
│  STEP 1: Individual Predictions                                   │
└──────────────────────────────────────────────────────────────────┘

CNN Output:
├─ Logits: [real_score, fake_score]
├─ Softmax: [P(real), P(fake)]
├─ Prediction: is_fake = (P(fake) > P(real))
└─ Confidence: max(P(real), P(fake))
   Example: [0.12, 0.88] → is_fake=True, confidence=0.88

AASIST Output:
├─ Logits: [real_score, fake_score]
├─ Softmax: [P(real), P(fake)]
├─ Prediction: is_fake = (P(fake) > P(real))
└─ Confidence: max(P(real), P(fake))
   Example: [0.05, 0.95] → is_fake=True, confidence=0.95

Watermark Output:
├─ Features: [energy, flatness, variance, periodicity, alignment]
├─ Weighted sum: confidence_score
├─ Prediction: is_fake = (confidence > 0.65)
└─ Confidence: confidence_score
   Example: 0.82 → is_fake=True, confidence=0.82


┌──────────────────────────────────────────────────────────────────┐
│  STEP 2: Weighted Voting                                          │
└──────────────────────────────────────────────────────────────────┘

votes = {
    'cnn': {
        'is_fake': True,
        'confidence': 0.88,
        'weight': 0.35,
        'vote_strength': 0.88 × 0.35 = 0.308
    },
    'aasist': {
        'is_fake': True,
        'confidence': 0.95,
        'weight': 0.35,
        'vote_strength': 0.95 × 0.35 = 0.333
    },
    'watermark': {
        'is_fake': True,
        'confidence': 0.82,
        'weight': 0.30,
        'vote_strength': 0.82 × 0.30 = 0.246
    }
}

Fake Score Calculation:
fake_score = Σ(vote_strength for detectors where is_fake=True)
fake_score = 0.308 + 0.333 + 0.246 = 0.887

Real Score Calculation:
real_score = Σ(vote_strength for detectors where is_fake=False)
real_score = 0 (all voted fake)

Total Weight:
total_weight = 0.35 + 0.35 + 0.30 = 1.0


┌──────────────────────────────────────────────────────────────────┐
│  STEP 3: Final Decision                                           │
└──────────────────────────────────────────────────────────────────┘

Final Prediction:
final_is_fake = (fake_score > real_score)
final_is_fake = (0.887 > 0) = True

Final Confidence:
final_confidence = max(fake_score, real_score) / total_weight
final_confidence = 0.887 / 1.0 = 0.887


┌──────────────────────────────────────────────────────────────────┐
│  STEP 4: Agreement Analysis                                       │
└──────────────────────────────────────────────────────────────────┘

Count Fake Votes:
fake_count = 3 (CNN=fake, AASIST=fake, Watermark=fake)

Agreement Level:
if fake_count == 0 or fake_count == 3:
    agreement = "UNANIMOUS"  ← All agree
elif fake_count == 2:
    agreement = "MAJORITY"   ← 2 out of 3
else:
    agreement = "SPLIT"      ← Disagreement

This example: UNANIMOUS (all 3 detected fake)


┌──────────────────────────────────────────────────────────────────┐
│  STEP 5: Winner Determination                                     │
└──────────────────────────────────────────────────────────────────┘

Find most confident detector:
confidences = {
    'cnn': 0.88,
    'aasist': 0.95,  ← Highest
    'watermark': 0.82
}

winner = 'aasist'  (confidence 0.95)


┌──────────────────────────────────────────────────────────────────┐
│  STEP 6: Final Output                                             │
└──────────────────────────────────────────────────────────────────┘

return {
    'is_fake': True,
    'confidence': 0.887,
    'agreement': 'UNANIMOUS',
    'winner': 'aasist',
    'layer_votes': {
        'cnn': {'is_fake': True, 'confidence': 0.88},
        'aasist': {'is_fake': True, 'confidence': 0.95},
        'watermark': {'is_fake': True, 'confidence': 0.82}
    }
}


┌──────────────────────────────────────────────────────────────────┐
│  Example Scenarios                                                │
└──────────────────────────────────────────────────────────────────┘

SCENARIO 1: Unanimous Fake Detection
├─ CNN: is_fake=True, conf=0.92
├─ AASIST: is_fake=True, conf=0.98
├─ Watermark: is_fake=True, conf=0.85
└─ Result: is_fake=True, conf=0.916, agreement=UNANIMOUS

SCENARIO 2: Majority Fake (CNN disagrees)
├─ CNN: is_fake=False, conf=0.75  ← Thinks it's real
├─ AASIST: is_fake=True, conf=0.95
├─ Watermark: is_fake=True, conf=0.88
├─ fake_score = 0.95×0.35 + 0.88×0.30 = 0.597
├─ real_score = 0.75×0.35 = 0.263
└─ Result: is_fake=True, conf=0.597, agreement=MAJORITY

SCENARIO 3: Split Decision (Real audio with noise)
├─ CNN: is_fake=True, conf=0.65  ← False positive
├─ AASIST: is_fake=False, conf=0.92
├─ Watermark: is_fake=False, conf=0.45
├─ fake_score = 0.65×0.35 = 0.228
├─ real_score = 0.92×0.35 + 0.45×0.30 = 0.457
└─ Result: is_fake=False, conf=0.457, agreement=MAJORITY

SCENARIO 4: NeuTTS Air Sample (Perfect Detection)
├─ CNN: is_fake=True, conf=0.96
├─ AASIST: is_fake=True, conf=0.99
├─ Watermark: is_fake=True, conf=0.95  ← Watermark perfect on NeuTTS
└─ Result: is_fake=True, conf=0.967, agreement=UNANIMOUS


┌──────────────────────────────────────────────────────────────────┐
│  Why This Voting System?                                          │
├──────────────────────────────────────────────────────────────────┤
│  1. Soft Voting: Uses probabilities, not hard votes              │
│     - Preserves uncertainty information                           │
│     - High-confidence votes matter more                           │
│                                                                   │
│  2. Weighted Combination: Not all detectors equal                │
│     - CNN/AASIST are trained models (35% each)                   │
│     - Watermark is heuristic (30%, slightly less)                │
│                                                                   │
│  3. Agreement Tracking: Indicates reliability                    │
│     - UNANIMOUS: Very confident                                  │
│     - MAJORITY: Fairly confident                                 │
│     - SPLIT: Low confidence, manual review recommended           │
│                                                                   │
│  4. Winner Tracking: Explainability                              │
│     - "AASIST was most confident this is fake"                   │
│     - Helps debug false positives/negatives                      │
│                                                                   │
│  5. Complementary Strengths:                                     │
│     - CNN: Fast, spectral patterns                               │
│     - AASIST: Accurate, attention-based                          │
│     - Watermark: TTS-specific, 100% on NeuTTS Air                │
└──────────────────────────────────────────────────────────────────┘
```

---

## Training Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                              │
└──────────────────────────────────────────────────────────────────┘

Phase 1: Data Preparation
  │
  ├─ Load 1400 samples (700 real + 700 fake)
  ├─ Extract CNN features (30-D)
  ├─ Extract AASIST features (64×128)
  ├─ Create labels (real=0, fake=1)
  │
  ▼
Phase 2: Train/Validation Split (80/20)
  │
  ├─ Shuffle: torch.randperm(1400)
  ├─ Train indices: first 1120
  ├─ Val indices: last 280
  │
  ▼
Phase 3: CNN Training (Separate)
  │
  ├─ Model: OptimizedCNN
  ├─ Device: CUDA (if available)
  ├─ Optimizer: Adam(lr=0.001)
  ├─ Loss: CrossEntropyLoss
  ├─ Batch size: Adaptive (8-64)
  ├─ Epochs: 10
  │
  │  For each epoch:
  │  ├─ Shuffle training data
  │  ├─ For each batch:
  │  │  ├─ Forward pass: logits = model(features)
  │  │  ├─ Compute loss: loss = criterion(logits, labels)
  │  │  ├─ Backward pass: loss.backward()
  │  │  ├─ Update weights: optimizer.step()
  │  │  └─ Zero gradients: optimizer.zero_grad()
  │  │
  │  ├─ Validation:
  │  │  ├─ No gradient: with torch.no_grad()
  │  │  ├─ Compute predictions on val set
  │  │  ├─ Compute F1, precision, recall, AUC
  │  │  └─ Log metrics
  │  │
  │  └─ Memory cleanup: torch.cuda.empty_cache()
  │
  ├─ Save model: cnn_model.pth
  │
  ▼
Phase 4: AASIST Training (Separate)
  │
  ├─ Model: AASISTModel
  ├─ Device: CUDA (if available)
  ├─ Optimizer: Adam(lr=0.001)
  ├─ Loss: CrossEntropyLoss
  ├─ Batch size: Adaptive (2-8, smaller than CNN)
  ├─ Epochs: 10
  │
  │  [Same training loop as CNN]
  │
  ├─ Save model: aasist_model.pth
  │
  ▼
Phase 5: Watermark Validation
  │
  ├─ Test on 50 NeuTTS Air samples
  ├─ Compute detection rate
  ├─ Compute average confidence
  ├─ Compute false positive rate on real audio
  │
  ▼
Phase 6: Ensemble Validation
  │
  ├─ Load both models
  ├─ For each validation sample:
  │  ├─ CNN prediction + confidence
  │  ├─ AASIST prediction + confidence
  │  ├─ Watermark prediction + confidence
  │  └─ Triple-layer voting → final prediction
  │
  ├─ Compute ensemble metrics:
  │  ├─ F1-score
  │  ├─ Precision
  │  ├─ Recall
  │  ├─ Accuracy
  │  └─ AUC
  │
  ▼
Phase 7: Generate Visualizations
  │
  └─ Create 12 charts (performance, comparison, etc.)

Total Training Time: ~70 minutes
  ├─ Voice cloning: 67.7 min
  ├─ Feature extraction: 448.45s (~7.5 min)
  ├─ CNN training: 3.1s
  ├─ AASIST training: 22.77s
  └─ Evaluation: ~10 min
```

---

## Inference Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                   INFERENCE PIPELINE                              │
└──────────────────────────────────────────────────────────────────┘

INPUT: Unknown audio file
  │
  ▼
Step 1: Load Audio
  │
  ├─ Check file exists
  ├─ Load with librosa
  └─ Validate: duration > 0.5s
  │
  ▼
Step 2: Parallel Feature Extraction
  │
  ├──────────────────┬─────────────────┬──────────────────┐
  │                  │                 │                  │
  ▼                  ▼                 ▼                  ▼
┌──────────┐  ┌─────────────┐  ┌──────────────┐
│   CNN    │  │   AASIST    │  │  WATERMARK   │
│ Features │  │  Features   │  │  Analysis    │
├──────────┤  ├─────────────┤  ├──────────────┤
│ 30-D     │  │  64×128     │  │  8-12kHz     │
│ vector   │  │  mel-spec   │  │  5 features  │
└────┬─────┘  └──────┬──────┘  └──────┬───────┘
     │               │                 │
     ▼               ▼                 ▼
Step 3: Load Models
  │
  ├─ cnn_model.pth
  ├─ aasist_model.pth
  └─ Set to eval mode: model.eval()
  │
  ▼
Step 4: Individual Inferences
  │
  ├─ CNN inference:
  │  ├─ features → model → logits
  │  ├─ softmax → [P(real), P(fake)]
  │  └─ confidence = max(P)
  │
  ├─ AASIST inference:
  │  ├─ spectrogram → model → logits
  │  ├─ softmax → [P(real), P(fake)]
  │  └─ confidence = max(P)
  │
  ├─ Watermark detection:
  │  ├─ Compute 5 features
  │  ├─ Weighted sum → confidence
  │  └─ Threshold: confidence > 0.65
  │
  ▼
Step 5: Triple-Layer Voting
  │
  ├─ Combine votes with weights (35-35-30)
  ├─ Compute final confidence
  ├─ Determine agreement level
  └─ Identify winner
  │
  ▼
Step 6: Return Results
  │
  └─ {
      'is_fake': bool,
      'confidence': float,
      'agreement': str,
      'winner': str,
      'layer_votes': dict
    }

Total Inference Time per Sample: ~450ms
  ├─ Feature extraction: 200ms
  ├─ CNN inference: 50ms
  ├─ AASIST inference: 150ms
  ├─ Watermark detection: 30ms
  └─ Voting: 20ms
```

---

## Summary: Key Architectural Insights

### 1. **Separation of Concerns**
- Voice cloning, feature extraction, training, and inference are separate
- Models trained independently, combined only at inference
- Each component can be updated without touching others

### 2. **Hardware Adaptivity**
- Architecture scales automatically based on available VRAM
- Same code runs on CPU, GTX 1060, RTX 3090, A100
- Prevents crashes while maximizing performance

### 3. **Multi-Scale Processing**
- CNN: Hierarchical features (local → global)
- AASIST: Multi-stage (spatial → attention → temporal)
- Watermark: Multi-feature fusion

### 4. **Complementary Detection**
- CNN: Fast screening with traditional features
- AASIST: Deep analysis with attention
- Watermark: TTS-specific fingerprinting

### 5. **Production-Ready Design**
- Memory management throughout
- Error handling at each stage
- Logging and profiling built-in
- Explainable outputs (agreement, winner)

This architecture represents a well-engineered system that balances **accuracy, efficiency, and robustness** through careful design of every component.
