# Deploy the app so Ali can use it with just a link

Right now the app runs **locally** (only on your machine). To give Ali a **link** (e.g. `https://your-app.onrender.com`) so he can open it in a browser without installing anything, you need to put the app on a **host** on the internet.

Below is the simplest path: **Render** (free tier). Same idea works for **Railway** or **Hugging Face Spaces**.

---

## 1. Put the project on GitHub

If it’s not there yet:

- Create a repo on [github.com](https://github.com) (e.g. `SWIFT-Fraud-Detection`).
- From your project folder:

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/SWIFT-Fraud-Detection.git
git branch -M main
git push -u origin main
```

**Important:** The **trained model** must be in the repo so the website can load it. The repo is set up to allow `models/fraud_detector.pkl`. If it was never committed, run:

```bash
git add -f models/fraud_detector.pkl
git commit -m "Add trained model for deployment"
git push
```

---

## 2. Deploy on Render (free)

1. Go to [render.com](https://render.com) and sign up (free).
2. **New** → **Web Service**.
3. Connect your **GitHub** account and select the repo `SWIFT-Fraud-Detection`.
4. Configure:
   - **Name:** e.g. `swift-fraud-detection`
   - **Region:** pick one close to you or Ali.
   - **Branch:** `main`
   - **Runtime:** `Python 3`
   - **Build command:**  
     `pip install -r requirements.txt`
   - **Start command:**  
     `gunicorn --bind 0.0.0.0:$PORT app:app`
5. **Create Web Service**. Render will build and start the app.
6. When it’s live, Render gives you a URL like:  
   `https://swift-fraud-detection.onrender.com`  
   That’s the link you send to Ali. He opens it in a browser; no install, no backend on his side.

**Free tier note:** The app may “sleep” after 15 minutes of no use. The first open after that can take 30–60 seconds to wake up; after that it’s fast again.

---

## 3. Optional: Railway or Hugging Face

- **Railway:** [railway.app](https://railway.app) → New Project → Deploy from GitHub repo. Set start command to `gunicorn --bind 0.0.0.0:$PORT app:app`. Free tier has a monthly allowance.
- **Hugging Face Spaces:** [huggingface.co/spaces](https://huggingface.co/spaces) → Create Space → **Gradio** or **Docker**. You’d wrap the Flask app or expose a simple Gradio UI that calls your logic; good if you want a “demo” feel.

---

## Summary

| Step | What you do |
|------|-------------|
| 1 | Push the project (including `models/fraud_detector.pkl`) to GitHub. |
| 2 | On Render: New Web Service → connect repo → build `pip install -r requirements.txt`, start `gunicorn --bind 0.0.0.0:$PORT app:app`. |
| 3 | Send Ali the Render URL. He uses the site with just the link; nothing to download or run. |

The app is already set up to use the `PORT` and `0.0.0.0` host that Render (and similar hosts) provide, so no extra code changes are needed beyond what’s in the repo.
