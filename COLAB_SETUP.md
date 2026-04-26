# 🚀 Google Colab Training - Step by Step

## Quick Start (5 minutes to start training)

### Step 1: Upload Notebook to Colab

**Option A: Direct Upload (Easiest)**
1. Go to https://colab.research.google.com/
2. Click "File" → "Upload notebook"
3. Select `training_grpo_final.ipynb` from your computer
4. Wait for upload to complete

**Option B: From GitHub**
1. Go to https://colab.research.google.com/
2. Click "File" → "Open notebook"
3. Click "GitHub" tab
4. Enter: `tusharpawar04/hallicunation-hunter`
5. Select `training_grpo_final.ipynb`

### Step 2: Enable GPU (CRITICAL!)

⚠️ **This is the most important step!**

1. Click "Runtime" in the top menu
2. Select "Change runtime type"
3. Under "Hardware accelerator", select **"T4 GPU"**
4. Click "Save"

**Verify GPU is enabled:**
- After running cell 2, you should see:
  ```
  PyTorch: 2.x.x
  CUDA: True
  GPU: Tesla T4
  ```
- If it says `CUDA: False`, go back and enable GPU!

### Step 3: Run Training

1. Click "Runtime" → "Run all" (or press Ctrl+F9)
2. When prompted, click "Run anyway" (it's safe)
3. Wait for training to complete (2-3 hours)

**What happens:**
- Cell 1: Install dependencies (5 min)
- Cell 2: Check GPU (10 sec)
- Cell 3: Load model (5 min)
- Cell 4: Setup environment (30 sec)
- Cell 5: Configure GRPO (10 sec)
- Cell 6: Create trainer (30 sec)
- Cell 7: **Train model (2-3 hours)** ⏰
- Cell 8: Evaluate (2 min)
- Cell 9: Generate plots (30 sec)
- Cell 10: Save model (1 min)

### Step 4: Download Results

After training completes:

1. **Click the folder icon** (📁) on the left sidebar
2. **Find and download:**
   - `grpo_training_results.png` - The training plots
   - `hallucination-hunter-grpo-lora/` - The trained model (optional)

3. **Right-click** `grpo_training_results.png` → "Download"
4. Save to your project folder

## 📊 What to Expect

### Installation Phase (5 minutes)
```
Installing unsloth...
Installing trl...
Installing transformers...
✅ All packages installed
```

### Model Loading (5 minutes)
```
Loading Qwen2.5-3B-Instruct...
Applying 4-bit quantization...
Adding LoRA adapters (r=16)...
✅ Model loaded with LoRA
Trainable params: 41M / 3B (1.4%)
```

### Environment Connection (10 seconds)
```
Testing connection...
✅ Connected to environment
Episodes: 10
Status: healthy
```

### Baseline Collection (2 minutes)
```
Collecting baseline rewards...
Baseline 1/10: -4.500
Baseline 2/10: -4.200
Baseline 3/10: -4.800
...
Baseline avg: -4.450
```

### Training (2-3 hours)
```
🚀 Starting GRPO training...

Step 5/200: reward=0.245, loss=2.341
Step 10/200: reward=0.312, loss=2.198
Step 15/200: reward=0.401, loss=2.045
...
Step 200/200: reward=0.678, loss=1.234

✅ Training complete!
```

### Evaluation (2 minutes)
```
Evaluating trained model...
Trained 1/10: -3.200
Trained 2/10: -2.800
...
Trained avg: -3.100
Improvement: +1.350
```

### Plots Generated
You'll see 3 plots:
1. **Reward Over Time** - Shows improvement during training
2. **Loss Over Time** - Shows loss decreasing
3. **Before vs After** - Bar chart comparison

## ⚠️ Common Issues

### Issue 1: "CUDA: False"
**Problem**: GPU not enabled
**Solution**: 
1. Runtime → Change runtime type
2. Select "T4 GPU"
3. Save and restart

### Issue 2: "Out of memory"
**Problem**: GPU memory full
**Solution**: Reduce batch size in cell 5:
```python
config = GRPOConfig(
    per_device_train_batch_size=1,  # Already minimum
    gradient_accumulation_steps=2,  # Reduce from 4 to 2
    ...
)
```

### Issue 3: "Connection timeout"
**Problem**: HuggingFace Space not responding
**Solution**: 
1. Visit: https://tusharpawar21-hallicunation-hunt.hf.space/health
2. Wait 2-3 minutes for Space to wake up
3. Restart the cell

### Issue 4: Colab disconnects
**Problem**: Free Colab has 12-hour limit
**Solution**: 
- Keep the tab open
- Check progress every hour
- Or use Colab Pro ($10/month)

### Issue 5: Training is slow
**Problem**: Not using GPU
**Solution**: 
- Check cell 2 output shows `CUDA: True`
- If not, enable GPU (see Issue 1)
- Expected: ~1-2 min per step with GPU
- Without GPU: 10+ hours (not recommended)

## 💡 Tips

1. **Keep tab open**: Colab disconnects if you close the browser
2. **Monitor progress**: Check output every 30 minutes
3. **Save notebook**: File → Save a copy in Drive
4. **Download results early**: Don't wait until the end
5. **Use Colab Pro**: If you need guaranteed GPU access

## ⏱️ Time Breakdown

| Phase | Time | Can Skip? |
|-------|------|-----------|
| Upload notebook | 1 min | No |
| Enable GPU | 1 min | No |
| Install dependencies | 5 min | No |
| Load model | 5 min | No |
| Baseline collection | 2 min | No |
| **GRPO Training** | **2-3 hours** | **No** |
| Evaluation | 2 min | No |
| Plot generation | 30 sec | No |
| Download results | 1 min | No |
| **Total** | **~3 hours** | - |

## ✅ Success Checklist

After training, you should have:
- ✅ Training completed without errors
- ✅ Plot showing reward improvement
- ✅ `grpo_training_results.png` downloaded
- ✅ Baseline and trained rewards recorded
- ✅ Improvement percentage calculated

## 📝 Next Steps After Training

1. **Add plot to README:**
   ```bash
   # Copy downloaded plot to project folder
   git add grpo_training_results.png
   git add README.md  # Update with results
   git commit -m "Add GRPO training results"
   git push origin main
   ```

2. **Update README with your actual metrics:**
   ```markdown
   ## 📊 Training Results
   
   ![GRPO Training Results](grpo_training_results.png)
   
   - Baseline: -4.45 → Trained: -3.10
   - Improvement: +1.35 (30.3% better)
   - Training: 200 steps with GRPO
   ```

3. **Write blog post** with your real results

## 🎯 Ready to Start?

1. ✅ Open https://colab.research.google.com/
2. ✅ Upload `training_grpo_final.ipynb`
3. ✅ Enable GPU (T4)
4. ✅ Click "Run all"
5. ⏰ Come back in 3 hours!

**The notebook is ready - just upload and run!** 🚀
