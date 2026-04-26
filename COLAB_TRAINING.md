# 🚀 Google Colab Training Guide

## Quick Start (3 Steps)

### Step 1: Open in Colab
1. Go to https://colab.research.google.com/
2. Click "File" → "Upload notebook"
3. Select `training_grpo_final.ipynb` from this project

**OR** use this direct link:
- Upload the notebook to your Google Drive
- Right-click → "Open with" → "Google Colaboratory"

### Step 2: Enable GPU (CRITICAL!)
1. Click "Runtime" in the top menu
2. Select "Change runtime type"
3. Under "Hardware accelerator", select **"T4 GPU"**
4. Click "Save"

**Verify GPU is enabled:**
- After running the first cell, you should see: `CUDA: True`
- If it says `CUDA: False`, go back and enable GPU

### Step 3: Run Training
1. Click "Runtime" → "Run all" (or press Ctrl+F9)
2. Wait for training to complete (2-3 hours)
3. Download the results

## 📊 What to Expect

### Installation (5 minutes)
```
Installing: unsloth, trl, transformers, accelerate...
✅ All packages installed
```

### Model Loading (5 minutes)
```
Loading Qwen2.5-3B-Instruct with 4-bit quantization...
Adding LoRA adapters (r=16)...
✅ Model loaded with LoRA
```

### Environment Connection (10 seconds)
```
✅ Connected to environment
Episodes: 10
```

### Baseline Collection (2 minutes)
```
Collecting baseline rewards...
Baseline 1/10: -4.500
Baseline 2/10: -4.200
...
Baseline avg: -4.350
```

### GRPO Training (2-3 hours)
```
🚀 Starting GRPO training...

Step 5: reward=0.245, loss=2.341
Step 10: reward=0.312, loss=2.198
Step 15: reward=0.401, loss=2.045
...
Step 200: reward=0.678, loss=1.234

✅ Training complete!
```

### Evaluation (2 minutes)
```
Evaluating trained model...
Trained 1/10: -3.200
Trained 2/10: -2.800
...
Trained avg: -3.100
Improvement: +1.250
```

### Plots Generated
You'll see 3 plots:
1. **Reward Over Time** - Shows reward increasing during training
2. **Loss Over Time** - Shows loss decreasing
3. **Before vs After** - Bar chart comparing baseline vs trained

## 📥 Download Results

After training completes:

1. **Click the folder icon** on the left sidebar
2. **Find these files:**
   - `grpo_training_results.png` - The training plots
   - `hallucination-hunter-grpo-lora/` - The trained model

3. **Download the plot:**
   - Right-click `grpo_training_results.png`
   - Select "Download"
   - Save to your project folder

4. **Optional: Download the model:**
   - Right-click `hallucination-hunter-grpo-lora/`
   - Select "Download" (will download as zip)

## 🔧 Troubleshooting

### "CUDA: False" after enabling GPU
- **Solution**: Runtime → Disconnect and delete runtime → Reconnect → Enable GPU again

### "Out of memory" error
- **Solution**: Reduce batch size in the config:
  ```python
  per_device_train_batch_size=1,  # Already at minimum
  gradient_accumulation_steps=2,  # Reduce from 4 to 2
  ```

### "Connection timeout" to environment
- **Solution**: Check if HuggingFace Space is running:
  - Visit: https://tusharpawar21-hallicunation-hunt.hf.space/health
  - Should show: `{"status": "healthy", ...}`
  - If not, wait a few minutes for Space to wake up

### Colab disconnects during training
- **Problem**: Free Colab has 12-hour limit and can disconnect
- **Solution**: 
  - Use Colab Pro ($10/month) for longer sessions
  - Or: Reduce `max_steps` from 200 to 100 for faster training

### Training is very slow
- **Check**: Make sure GPU is enabled (see Step 2)
- **Expected speed**: ~1-2 minutes per step with GPU
- **Without GPU**: Would take 10+ hours (not recommended)

## ✅ Success Checklist

After training, you should have:
- ✅ Training completed without errors
- ✅ Plot showing reward improvement
- ✅ `grpo_training_results.png` downloaded
- ✅ Baseline and trained rewards recorded

## 📝 Next Steps

1. **Add plot to README:**
   ```bash
   # Copy downloaded plot to project folder
   git add grpo_training_results.png
   git commit -m "Add GRPO training results"
   git push origin main
   ```

2. **Update README with metrics:**
   ```markdown
   ## Training Results
   
   ![GRPO Training Results](grpo_training_results.png)
   
   - Baseline: -4.35 → Trained: -3.10
   - Improvement: +1.25 (28% better)
   - Training: 200 steps with GRPO
   ```

3. **Write blog post** (see TRAINING_INSTRUCTIONS.md)

## 💡 Tips

- **Save your work**: Colab auto-saves, but download the notebook periodically
- **Monitor progress**: Check the output every 30 minutes
- **Don't close the tab**: Colab will disconnect if you close the browser
- **Use Colab Pro**: If you need guaranteed GPU access and longer sessions

## 🎯 Expected Training Time

| Phase | Time |
|-------|------|
| Setup & Installation | 10 min |
| Model Loading | 5 min |
| Baseline Collection | 2 min |
| GRPO Training (200 steps) | 2-3 hours |
| Evaluation | 2 min |
| Plot Generation | 1 min |
| **Total** | **~3 hours** |

## 🚀 Ready to Start?

1. Open https://colab.research.google.com/
2. Upload `training_grpo_final.ipynb`
3. Enable GPU (T4)
4. Click "Run all"
5. Come back in 3 hours!

Good luck! 🎉
