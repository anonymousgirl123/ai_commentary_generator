# 🎙️ Sports Commentator from Video  
**Capstone Project – CS[05]**

https://github.com/anonymousgirl123/ai_commentary_generator/wiki

An AI-powered system that generates **timestamped, broadcast-style sports commentary** from raw video using a **two-stage Large Language Model (LLM) pipeline**.

--- 

## 🚀 Overview

This project combines **computer vision + AI language models** to automatically generate engaging sports commentary.

### 🔹 Pipeline
1. Frame Extraction (OpenCV)  
2. Scene Understanding (Gemini 2.5 Flash – Vision LLM)  
3. Commentary Generation (Gemini 2.5 Pro – Language LLM)  
4. Optional Text-to-Speech (TTS)

---

## ✨ Features

- 🎥 Works with **any sports video**
- ⏱️ Timestamped commentary `[MM:SS]`
- 🧠 Two-LLM architecture (Vision + Language)
- 🎙️ Optional Text-to-Speech output
- 🖥️ Interactive terminal dashboard
- ⚡ Two modes:
  - **Live Mode** (detailed, real-time style)
  - **Highlight Mode** (fast summary)

---

## 🏗️ Architecture
Video Input
↓
Frame Extraction (OpenCV)
↓
LLM-1 (Gemini Flash) → Scene Descriptions
↓
LLM-2 (Gemini Pro) → Commentary Script
↓
TTS (Optional)
↓
Output (Text + Audio)

---

## 📦 Project Structure

.
├── video_processor.py # Frame extraction & encoding
├── commentary_generator.py # Two-LLM pipeline
├── tts_engine.py # Speech synthesis
├── dashboard.py # Interactive CLI dashboard
├── main.py # Entry point
└── .env # API keys



---

## ⚙️ Tech Stack

- Python 3.9+
- OpenCV (cv2)
- Google Gemini API (`google-genai`)
- python-dotenv
- macOS `say` (TTS)
- ANSI Terminal UI

---

## 🔑 API Key Setup

Get a free API key:  
👉 https://aistudio.google.com/app/apikey

### Option A – CLI Argument

--api_key YOUR_KEY_HERE


### Option B – Environment Variable (Recommended)

**Linux / macOS**
```bash
export GOOGLE_API_KEY="YOUR_KEY_HERE"

```

Windows (CMD)
``` set GOOGLE_API_KEY=YOUR_KEY_HERE ```


📦 Installation
``` git clone https://github.com/your-username/sports-commentator.git
cd sports-commentator

pip install opencv-python google-genai python-dotenv
```

▶️ Usage
🖥️ Option A – Interactive Dashboard (Recommended)
``` python dashboard.py ```


📌 Dashboard Flow

Step 1 – Video & API Setup
Enter full video path
Enter API key (skip if env variable is set)

Step 2 – Commentary Mode
Highlight Mode (Default) → every 10s, faster
Live Mode → every 3s, more detailed

Step 3 – Match Details
Sport (default: football)
Team A / Team B
Context (optional)
Enable TTS (y/n)
Audio file (optional)
Output file (default: commentary_output.txt)
Confirmation
y → Run
n → Reconfigure
Output
Live progress in terminal
Final commentary displayed + saved

Feedback
Rate (1–5 stars)
Optional comment
💻 Option B – Direct CLI (Advanced)
Basic:
```python main.py --video match.mp4 --api_key YOUR_KEY_HERE```
Full Example:
``` python main.py \
  --video       match.mp4 \
  --api_key     YOUR_KEY_HERE \
  --mode        highlight \
  --sport       football \
  --team_a      "Manchester City" \
  --team_b      "Arsenal" \
  --context     "Semi-final, 0-0 at kick-off" \
  --output      commentary_output.txt \
  --tts
🧾 Arguments
Argument	Description
--video	Input video path (required)
--api_key	Gemini API key
--mode	live / highlight
--sport	Sport type
--team_a	Home team
--team_b	Away team
--context	Match context
--interval	Frame interval
--max_frames	Max frames
--output	Output file
--save_scenes	Save scene JSON
--tts	Enable TTS
--audio_output	Save audio file
```

📊 Performance Summary
Metric	Live Mode	Highlight Mode
Frame interval	3 seconds	10 seconds
Max frames	60	20
Avg. processing time	~3–5 min	~1–2 min
LLM-1 (per frame)	~2–3 sec	~2–3 sec
LLM-2 (full script)	~5–10 sec	~3–5 sec
Commentary accuracy	High	Moderate–High
🧪 Sample Output
[00:00] The match kicks off with high intensity.

[00:10] A brilliant through-ball breaks the defence — the striker is through!

[00:20] GOAL! A stunning finish into the top corner!

SUMMARY: A clinical and well-deserved goal from Team A.

📤 Output
Terminal → Timestamped commentary
commentary_output.txt → Full script
scene_analysis.json → (optional) intermediate data
Audio file → (if TTS enabled)

🤖 Models Used
Gemini 2.5 Flash → Vision (frame analysis)
Gemini 2.5 Pro → Language (commentary generation)

🔮 Future Improvements
Real-time live streaming
Player & team detection
Live score API integration
Multi-language commentary
Fine-tuned sports LLM
Web-based dashboard
🤝 Contributing

Pull requests are welcome. Feel free to fork and improve the project.

📜 License

For academic and research purposes.

👤 Author

Kamini

⭐ If you like this project, give it a star!


