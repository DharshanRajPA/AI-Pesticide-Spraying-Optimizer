# 🌱 AgriSprayAI - Complete Beginner Setup Guide

Welcome to AgriSprayAI! This guide will help you set up the project from scratch, even if you're completely new to computer science and programming.

## 📋 What is AgriSprayAI?

AgriSprayAI is an **AI-powered agricultural system** that:
- 📸 **Analyzes crop images** to detect pests and diseases
- 🎤 **Listens to farmer notes** (voice or text) about their crops
- 🧠 **Combines both** to understand the full picture
- 💊 **Calculates precise pesticide doses** needed (not too much, not too little)
- 🚁 **Creates flight plans** for drones to spray pesticides efficiently
- ✅ **Requires human approval** before any spraying happens

Think of it as a **smart farming assistant** that helps farmers use pesticides more efficiently and safely.

---

## 🎯 **BEST SETUP APPROACH FOR BEGINNERS**

### **Option 1: Docker Setup (RECOMMENDED ⭐)**
**Why this is best for beginners:**
- ✅ **No Python version conflicts** - Docker handles everything
- ✅ **One command setup** - Everything works out of the box
- ✅ **Easy to reset** - If something breaks, just restart
- ✅ **Same as production** - What you develop is what gets deployed
- ✅ **No system pollution** - Doesn't mess with your computer

### **Option 2: Manual Python Setup**
**Only if you want to learn Python environments:**
- ⚠️ **More complex** - Need to manage Python versions
- ⚠️ **Can break** - Environment conflicts are common
- ⚠️ **More steps** - Many things can go wrong

**I recommend Option 1 (Docker) for beginners!**

---

## 🛠️ **PREREQUISITES (What You Need First)**

### **For Docker Setup (Recommended):**
1. **Docker Desktop** - Download from [docker.com](https://www.docker.com/products/docker-desktop/)
2. **Git** - Download from [git-scm.com](https://git-scm.com/downloads)
3. **A text editor** - VS Code (recommended) from [code.visualstudio.com](https://code.visualstudio.com/)

### **For Manual Setup:**
1. **Python 3.9-3.11** (NOT 3.13 - causes issues)
2. **Git** - Download from [git-scm.com](https://git-scm.com/downloads)
3. **A text editor** - VS Code (recommended)

---

## 🚀 **SETUP METHOD 1: DOCKER (RECOMMENDED)**

### **Step 1: Install Docker Desktop**
1. Go to [docker.com](https://www.docker.com/products/docker-desktop/)
2. Download Docker Desktop for Windows
3. Install it (this may take 10-15 minutes)
4. **Restart your computer** after installation
5. Open Docker Desktop and make sure it's running (you'll see a whale icon in your system tray)

### **Step 2: Get the Project Code**
```bash
# Open Command Prompt or PowerShell
# Navigate to where you want the project
cd C:\Users\YourName\Documents

# Clone the project (if you have the repository URL)
git clone <your-repository-url> AI-Pesticide-Spraying-Optimizer
cd AI-Pesticide-Spraying-Optimizer
```

### **Step 3: Set Up Environment Variables**
1. Copy the example environment file:
```bash
copy env.example .env
```

2. Edit the `.env` file with your API keys:
```bash
# Open .env in a text editor (like Notepad or VS Code)
notepad .env
```

3. **Get your API keys:**
   - **Google Gemini API**: Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - **Kaggle API**: Go to [Kaggle Account Settings](https://www.kaggle.com/settings) → API → Create New Token

4. **Update the .env file:**
```bash
# Replace these with your actual keys
GEMINI_API_KEY=your_actual_gemini_api_key_here
KAGGLE_API_KEY=your_actual_kaggle_api_key_here
```

### **Step 4: Start the System**
```bash
# This will download everything and start all services
docker-compose up --build
```

**What this does:**
- Downloads all required software
- Sets up the database
- Starts the AI models
- Launches the web interface
- Everything runs in isolated containers

### **Step 5: Access the Application**
- **Web Interface**: Open your browser and go to `http://localhost:3000`
- **API Documentation**: Go to `http://localhost:8000/docs`

---

## 🐍 **SETUP METHOD 2: MANUAL PYTHON (Advanced)**

### **Step 1: Install Python 3.9-3.11**
⚠️ **Important**: Do NOT use Python 3.13 - it causes compatibility issues!

1. Go to [python.org](https://www.python.org/downloads/)
2. Download Python 3.11 (latest stable)
3. **During installation, check "Add Python to PATH"**
4. Verify installation:
```bash
python --version
# Should show Python 3.11.x
```

### **Step 2: Create Virtual Environment**
```bash
# Navigate to project directory
cd AI-Pesticide-Spraying-Optimizer

# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# You should see (venv) in your command prompt
```

### **Step 3: Install Dependencies**
```bash
# Make sure you're in the virtual environment
pip install --upgrade pip
pip install -r requirements.txt
```

### **Step 4: Set Up Environment Variables**
```bash
# Copy example file
copy env.example .env

# Edit with your API keys
notepad .env
```

### **Step 5: Start the System**
```bash
# Start the API server
python code/api/server.py

# In another terminal, start the web interface
cd ui
npm install
npm start
```

---

## 🔐 **SECRETS AND ENVIRONMENT FILES**

### **What are Secrets?**
Secrets are sensitive information like:
- API keys (passwords for external services)
- Database passwords
- Encryption keys
- Personal tokens

### **Files to Keep Secret (in .gitignore):**
```
.env                    # Your actual API keys
.env.local             # Local development overrides
secrets/               # Any secret files
*.pem                  # SSL certificates
*.key                  # Private keys
kaggle.json           # Kaggle API token
api_keys.json         # Other API keys
```

### **Files Safe to Share:**
```
env.example            # Template with fake keys
.gitignore            # Tells Git what to ignore
requirements.txt      # List of Python packages
```

### **How to Handle Secrets:**
1. **Never commit real secrets** to Git
2. **Use .env files** for local development
3. **Use environment variables** in production
4. **Share env.example** with fake values

---

## 📁 **PROJECT STRUCTURE EXPLAINED**

```
AI-Pesticide-Spraying-Optimizer/
├── 📁 code/                    # All the programming code
│   ├── 📁 api/                # Web server (talks to browsers)
│   ├── 📁 vision/             # Image analysis AI
│   ├── 📁 nlp/                # Text/voice processing AI
│   ├── 📁 fusion/             # Combines image + text
│   ├── 📁 action_engine/      # Calculates pesticide doses
│   └── 📁 planner/            # Creates drone flight plans
├── 📁 data/                   # All the data files
│   ├── 📁 raw/                # Original images
│   ├── 📁 annotated/          # Labeled data for training
│   └── 📁 field/              # Real farm data
├── 📁 configs/                # Configuration files
├── 📁 ui/                     # Web interface (React)
├── 📁 docs/                   # Documentation
├── 📁 tests/                  # Tests to make sure everything works
├── 📄 requirements.txt        # List of Python packages needed
├── 📄 docker-compose.yml      # Docker setup instructions
├── 📄 .env                    # Your secret API keys (DON'T SHARE!)
└── 📄 .gitignore             # Tells Git what files to ignore
```

---

## 🚀 **HOW TO START THE PROJECT**

### **Method 1: Docker (Easiest)**
```bash
# 1. Make sure Docker Desktop is running
# 2. Navigate to project folder
cd AI-Pesticide-Spraying-Optimizer

# 3. Start everything
docker-compose up --build

# 4. Open browser to http://localhost:3000
```

### **Method 2: Manual Python**
```bash
# Terminal 1: Start API server
cd AI-Pesticide-Spraying-Optimizer
python code/api/server.py

# Terminal 2: Start web interface
cd ui
npm install
npm start

# 3. Open browser to http://localhost:3000
```

---

## 🧪 **TESTING YOUR SETUP**

### **Test 1: Check if API is working**
```bash
# Open browser and go to:
http://localhost:8000/docs
```
You should see the API documentation page.

### **Test 2: Check if web interface is working**
```bash
# Open browser and go to:
http://localhost:3000
```
You should see the AgriSprayAI web interface.

### **Test 3: Test with a sample image**
1. Go to the web interface
2. Upload a test image
3. Add some text notes
4. Click "Analyze"
5. You should see results

---

## 🆘 **TROUBLESHOOTING**

### **Problem: "Docker not found"**
**Solution**: Install Docker Desktop and restart your computer.

### **Problem: "Python version error"**
**Solution**: Use Python 3.9-3.11, not 3.13.

### **Problem: "Permission denied"**
**Solution**: Run Command Prompt as Administrator.

### **Problem: "Port already in use"**
**Solution**: 
```bash
# Find what's using the port
netstat -ano | findstr :8000
# Kill the process (replace PID with actual number)
taskkill /PID <PID> /F
```

### **Problem: "API key not working"**
**Solution**: 
1. Check your `.env` file has real API keys
2. Make sure there are no extra spaces
3. Restart the application

### **Problem: "Whisper import error"**
**Solution**: This is fixed! We updated to use `openai-whisper` instead of `whisper`.

---

## 📚 **NEXT STEPS**

Once everything is running:

1. **Explore the web interface** - Upload some images and see how it works
2. **Read the documentation** - Check the `docs/` folder
3. **Try the API** - Use the documentation at `http://localhost:8000/docs`
4. **Look at the code** - Start with `code/api/server.py`
5. **Run tests** - `pytest tests/`

---

## 🎓 **LEARNING RESOURCES**

### **For Beginners:**
- [Python Tutorial](https://docs.python.org/3/tutorial/)
- [Docker Tutorial](https://docs.docker.com/get-started/)
- [Git Tutorial](https://learngitbranching.js.org/)

### **For This Project:**
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)

---

## 🤝 **GETTING HELP**

If you get stuck:
1. **Check this guide** - Most problems are covered here
2. **Check the logs** - Look in the terminal for error messages
3. **Google the error** - Copy the exact error message
4. **Ask for help** - Include the error message and what you were doing

---

## ✅ **SUCCESS CHECKLIST**

- [ ] Docker Desktop installed and running
- [ ] Project code downloaded
- [ ] `.env` file created with real API keys
- [ ] `docker-compose up --build` runs without errors
- [ ] Web interface opens at `http://localhost:3000`
- [ ] API documentation opens at `http://localhost:8000/docs`
- [ ] Can upload an image and get results

**Congratulations! You've successfully set up AgriSprayAI! 🎉**

---

*This guide was created to help beginners get started with AgriSprayAI. If you have any questions or run into issues, don't hesitate to ask for help!*
