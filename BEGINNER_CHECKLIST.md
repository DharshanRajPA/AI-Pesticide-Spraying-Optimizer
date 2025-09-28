# ✅ AgriSprayAI Beginner Setup Checklist

## 🎯 **Follow This Checklist Step by Step**

### **📋 Prerequisites (5 minutes)**
- [ ] **Windows 10 or 11** computer
- [ ] **8GB RAM minimum** (check in Task Manager)
- [ ] **10GB free disk space** (check in File Explorer)
- [ ] **Internet connection** working

### **🔧 Software Installation (10 minutes)**
- [ ] **Python 3.11** installed (NOT 3.13)
  - [ ] Downloaded from python.org
  - [ ] Checked "Add Python to PATH" during installation
  - [ ] Restarted computer after installation
  - [ ] Tested: `python --version` shows 3.11.x
- [ ] **Git** installed
  - [ ] Downloaded from git-scm.com
  - [ ] Installed with default settings
  - [ ] Tested: `git --version` works

### **📁 Project Setup (5 minutes)**
- [ ] **Project folder** in Documents or Desktop
- [ ] **Virtual environment** created: `python -m venv venv`
- [ ] **Virtual environment** activated: `venv\Scripts\activate`
- [ ] **Dependencies** installed: `pip install -r requirements.txt`

### **🔑 API Keys Setup (10 minutes)**
- [ ] **Google Gemini API Key** obtained
  - [ ] Went to https://makersuite.google.com/app/apikey
  - [ ] Signed in with Google account
  - [ ] Created new API key
  - [ ] Copied the key (starts with AIza...)
- [ ] **Kaggle API Key** obtained
  - [ ] Went to https://www.kaggle.com/settings
  - [ ] Signed in with Google account
  - [ ] Created new token
  - [ ] Downloaded and opened kaggle.json
  - [ ] Copied the API key
- [ ] **Environment file** set up
  - [ ] Copied `env.example` to `.env`
  - [ ] Replaced placeholder keys with real keys
  - [ ] Saved the file

### **🤖 AI Models Setup (5 minutes)**
- [ ] **Models downloaded**: `python download_models.py`
- [ ] **YOLOv8 model** downloaded (6.2MB)
- [ ] **Fusion model** created (1.9MB)
- [ ] **Segmentation model** created (1.5MB)
- [ ] **All models verified** in models/ folder

### **🚀 System Startup (2 minutes)**
- [ ] **API server** started: `python code/api/server.py`
- [ ] **Web interface** started: `cd ui && npm start`
- [ ] **Browser** opened to http://localhost:3000
- [ ] **API docs** accessible at http://localhost:8000/docs

### **🧪 Testing (5 minutes)**
- [ ] **System status** checked: `python check_system_status.py`
- [ ] **All 6/6 checks** passed
- [ ] **Test image** uploaded successfully
- [ ] **Analysis results** received
- [ ] **No error messages** in terminal

---

## 🎉 **SUCCESS INDICATORS**

### **✅ You're Done When:**
- [ ] **Web interface** opens at http://localhost:3000
- [ ] **API documentation** opens at http://localhost:8000/docs
- [ ] **Health check** shows all models loaded
- [ ] **Image upload** works and returns results
- [ ] **No warnings** about missing models
- [ ] **System status** shows 6/6 checks passed

### **🎊 Congratulations!**
You now have a **fully functional AI agricultural system**!

---

## 🆘 **If Something Goes Wrong**

### **Common Issues:**
1. **"Python not found"** → Reinstall Python, check "Add to PATH"
2. **"Permission denied"** → Run Command Prompt as Administrator
3. **"Port in use"** → Restart computer, close all browsers
4. **"API key error"** → Check .env file has real keys
5. **"Models not found"** → Run `python download_models.py`

### **Get Help:**
- **Check the guides**: ULTIMATE_BEGINNER_GUIDE.md
- **Run diagnostics**: `python check_system_status.py`
- **Ask for help** with specific error messages

---

## 🚀 **Next Steps After Setup**

### **Learn the System:**
1. **Upload different images** - Try various crop photos
2. **Test voice input** - Record audio notes
3. **Explore the API** - Check the documentation
4. **Read the code** - Start with `code/api/server.py`

### **Advanced Usage:**
1. **Train custom models** - Use your own crop data
2. **Deploy to production** - Use Docker when ready
3. **Integrate with drones** - Connect to real UAV systems
4. **Scale the system** - Handle multiple farms

---

**Total Setup Time: 30 minutes maximum!**
**You're now ready to revolutionize farming with AI! 🌾🤖**
