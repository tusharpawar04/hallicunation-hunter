# 🚀 Training Instructions

## ✅ Environment Status
**HuggingFace Space**: ✅ WORKING
- URL: https://tusharpawar21-hallicunation-hunt.hf.space
- Status: All endpoints working correctly
- Rewards: Being calculated properly

## 📋 Step-by-Step Training Guide

### Step 1: Open Training Notebook in Google Colab (5 minutes)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "File" → "Upload notebook"
3. Upload `training_grpo_final.ipynb` from this project
4. **Enable GPU**:
   - Click "Runtime" → "Change runtime type"
   - Select "GPU" → "T4"
   - Click "Save"

### Step 2: Run Training (2-3 hours)

1. Click "Runtime" → "Run all" (or press Ctrl+F9)
2. The notebook will:
   - Install dependencies (5 minutes)
   - Load Qwen2.5-3B model with Unsloth (5 minutes)
   - Connect to your HuggingFace Space
   - Collect baseline rewards (10 samples)
   - Train with GRPO for 200 steps (2-3 hours)
   - Evaluate trained model (10 samples)
   - Generate plots showing improvement

3. **Monitor progress**:
   - You'll see reward values printed every 5 steps
   - Training loss will decrease over time
   - Final plots will show before/after comparison

### Step 3: Download Results (2 minutes)

1. After training completes, find the generated plot:
   - Look for `grpo_training_results.png` in the Colab files
   - Click the folder icon on the left sidebar
   - Right-click `grpo_training_results.png` → Download

2. Also download the trained model (optional):
   - Folder: `hallucination-hunter-grpo-lora/`
   - Contains LoRA adapters for the trained model

### Step 4: Update README with Results (15 minutes)

1. Add the training plot to your README:
   ```markdown
   ## Training Results
   
   ![GRPO Training Results](grpo_training_results.png)
   
   - **Baseline Reward**: -4.5 (before training)
   - **Trained Reward**: [YOUR RESULT] (after 200 steps)
   - **Improvement**: [DIFFERENCE]
   ```

2. Commit and push:
   ```bash
   git add grpo_training_results.png README.md
   git commit -m "Add GRPO training results"
   git push origin main
   ```

### Step 5: Write HuggingFace Blog Post (1 hour)

1. Go to [HuggingFace](https://huggingface.co/) and log in
2. Click your profile → "New blog post"
3. Write a 700-word story covering:

   **Title**: "Hallucination Hunter: Teaching LLMs to Detect Their Own Hallucinations with GRPO"

   **Structure**:
   - **Introduction** (100 words): The hallucination problem in LLMs
   - **Our Approach** (150 words): Claim-level detection with deterministic rewards
   - **Technical Architecture** (200 words):
     - GRPO reinforcement learning
     - 8-component reward system
     - Anti-gaming mechanisms
     - Curriculum learning (L1→L4)
   - **Training Results** (150 words):
     - Before/after metrics
     - Include the training plot
     - Discuss improvement
   - **Demo & Code** (100 words):
     - Link to HuggingFace Space
     - Link to GitHub repo
     - How to try it
   - **Conclusion** (100 words): Future work and impact

4. Add images:
   - Upload `grpo_training_results.png`
   - Screenshot of your HuggingFace Space demo

5. Publish and copy the blog post URL

### Step 6: Final Updates (10 minutes)

1. Add blog post link to README:
   ```markdown
   ## Blog Post
   
   Read the full story: [Hallucination Hunter on HuggingFace](YOUR_BLOG_URL)
   ```

2. Verify all links work:
   - ✅ HuggingFace Space demo
   - ✅ GitHub repository
   - ✅ Blog post
   - ✅ Training plots visible

3. Final commit:
   ```bash
   git add README.md
   git commit -m "Add blog post link"
   git push origin main
   ```

## 🎯 Expected Results

**Baseline (before training)**:
- Average reward: ~-4.5
- Model struggles with claim detection
- Often produces invalid JSON

**After GRPO training (200 steps)**:
- Average reward: Should improve (less negative)
- Better claim detection accuracy
- More consistent JSON output
- Improved precision/recall

**Training plot will show**:
1. Reward curve increasing over time
2. Loss curve decreasing
3. Before/after comparison bar chart

## ⏱️ Time Breakdown

- **Training setup**: 10 minutes
- **GRPO training**: 2-3 hours (automated)
- **Download results**: 2 minutes
- **Update README**: 15 minutes
- **Write blog post**: 1 hour
- **Final updates**: 10 minutes

**Total**: ~4 hours (mostly automated training)

## 🆘 Troubleshooting

**If training fails**:
1. Check GPU is enabled in Colab
2. Verify environment URL is correct
3. Test environment connection first (cell 4)

**If rewards don't improve**:
- This is normal for short training (200 steps)
- The important thing is showing the training process
- Document the attempt and learning

**If Colab disconnects**:
- Training progress is lost
- Restart from beginning
- Consider using Colab Pro for longer sessions

## ✅ Success Criteria

You're done when you have:
- ✅ Training plot showing before/after comparison
- ✅ Plot added to README
- ✅ 700-word blog post published on HuggingFace
- ✅ Blog post linked in README
- ✅ All changes committed and pushed

## 🎉 Next Steps After Training

Once training is complete, you'll have:
1. **Evidence of RL training** (30% of hackathon score)
2. **Training plots** showing improvement
3. **Blog post** for storytelling (30% of score)
4. **Working demo** on HuggingFace Space

Your project will be complete and ready for submission!
