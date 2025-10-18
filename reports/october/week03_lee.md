## 🧠 Developer Weekly Report Template

**Name:** LEE
**Role / Department:** Development Team
**Week #:week3
**Date:** 18/Oct 2025

---

### 1️⃣ Weekly Overview

Provide a short summary (3–5 sentences) of what you worked on this week. Focus on technical progress, learning outcomes, and any key milestones.

> 1. Fixed a bug where the system couldn’t return to normal conversation after the art_curation_node.
> 2. Reduced the runtime of the art_curation_node by 40% (from 60+ seconds to 40+ seconds).

---

### 2️⃣ Task Progress

| Category    | Task Description                                    | Status         | Related Link / File                     |
| ----------- | --------------------------------------------------- | -------------- | --------------------------------------- |
| 🔧 Model    | Now, ready to connect with the image outline engine             | 🔄 In Progress    | https://github.com/Arti-Team/Arti_prototype/tree/Lee_dev |
| 🧩 Data     | NaN           | NaN | NaN                  |
| ⚙️ System   | Implemented LangGraph phase-detection logic         | 🔄 In Progress | `/src/langgraph/phase_detector.py`      |
| 🧠 Research | NaN | NaN    | NaN       |

---

### 3️⃣ Key Results

* Improved model accuracy from **78% → 85%**
* Reduced inference time by **~30%** using optimized data loading
* Established shared module structure for team development

---

### 4️⃣ Challenges & Solutions

| Challenge                         | Root Cause                           | Solution / Next Step                                  |
| --------------------------------- | ------------------------------------ | ----------------------------------------------------- |
| Model instability during training | Small batch size and gradient spikes | Applied gradient clipping and learning rate scheduler |
| Dataset label imbalance           | Emotion classes unevenly distributed | Used SMOTE oversampling + class weighting             |
| API timeout issue                 | Inefficient caching structure        | Implemented Redis caching layer                       |

---

### 5️⃣ Next Week Goals

| Goal                             | Planned Action                | Expected Outcome                  |
| -------------------------------- | ----------------------------- | --------------------------------- |
| Refine emotion-to-art mapping    | Expand RAG knowledge base     | More accurate art recommendations |
| Integrate model with frontend UI | API endpoint testing          | Smooth prototype demo             |
| Prepare demo for presentation    | Record user interaction video | Ready for next mentor review      |

---

### 6️⃣ Collaboration & Requests

* Request UI team to test model latency in the frontend
* Need updated emotion icon set from design team
* Suggest a short sync-up on dataset alignment (Tue 8 PM)

---

### 7️⃣ Reflection

Write 2–3 sentences about your personal takeaways or team experience this week.

> Example: I realized that optimizing model performance isn’t just about training—it’s also about system design. Communication with the design team helped clarify user experience goals, which I’ll keep in mind during future iterations.

---

### 📎 Attachments

* Code: [GitHub Branch – dev/emotion_refactor](https://github.com/...)
* Report: `/reports/week04_model_update.pdf`
* Notes: [Notion Log – Week 4](https://notion.so/...)

---

### ✅ Best Practices

* **Keep it concise:** Each section should fit in 1 screen view.
* **Use markdown for clarity:** Easier to track changes in GitHub or Notion.
* **Submit by:** Every Sunday night or Monday morning.
* **Highlight impact:** Focus not just on *what* you did, but *why it matters* for the project goal.
