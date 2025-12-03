# 📊 How to Check Training Progress

## Where to Run the Command

**Run this in your Terminal** (or the terminal in your IDE):

```bash
cd /Users/mbgirish/emotion-aware-ai-tutor
tail -f training_full.log
```

## Step-by-Step:

1. **Open Terminal**
   - On Mac: Press `Cmd + Space`, type "Terminal", press Enter
   - Or use the terminal panel in your IDE (VS Code, Cursor, etc.)

2. **Navigate to project directory:**
   ```bash
   cd /Users/mbgirish/emotion-aware-ai-tutor
   ```

3. **Watch training progress:**
   ```bash
   tail -f training_full.log
   ```

4. **What you'll see:**
   - Real-time updates as training progresses
   - Epoch numbers, loss, accuracy
   - Progress for both face and audio models

5. **To stop watching** (but keep training running):
   - Press `Ctrl + C`
   - Training continues in background

## Alternative: Check Periodically

Instead of watching continuously, you can check periodically:

```bash
cd /Users/mbgirish/emotion-aware-ai-tutor
tail -30 training_full.log
```

## Check if Models are Created

```bash
cd /Users/mbgirish/emotion-aware-ai-tutor
ls -lh models/*.pth
```

When both models exist, training is complete!

---

**Quick Command:**
```bash
cd /Users/mbgirish/emotion-aware-ai-tutor && tail -f training_full.log
```

