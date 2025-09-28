# 🆘 AgriSprayAI Troubleshooting Guide

This guide helps you solve common issues when setting up and running AgriSprayAI.

## 🚨 **Common Issues & Solutions**

### **1. Python Version Issues**

#### **Problem: "Python 3.13 compatibility error"**
```
TypeError: argument of type 'NoneType' is not iterable
```

**Solution:**
```bash
# Use Python 3.9-3.11 instead of 3.13
# Download from: https://python.org/downloads/
python --version  # Should show 3.9.x, 3.10.x, or 3.11.x
```

#### **Problem: "Python not found"**
```
'python' is not recognized as an internal or external command
```

**Solution:**
1. Install Python from [python.org](https://python.org/downloads/)
2. **During installation, check "Add Python to PATH"**
3. Restart your command prompt
4. Verify: `python --version`

---

### **2. Docker Issues**

#### **Problem: "Docker not found"**
```
'docker' is not recognized as an internal or external command
```

**Solution:**
1. Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/)
2. Install and restart your computer
3. Open Docker Desktop and wait for it to start
4. Verify: `docker --version`

#### **Problem: "Docker is not running"**
```
Cannot connect to the Docker daemon
```

**Solution:**
1. Open Docker Desktop application
2. Wait for the whale icon to appear in system tray
3. Make sure it shows "Docker Desktop is running"
4. Try: `docker info`

#### **Problem: "Port already in use"**
```
Port 8000 is already in use
```

**Solution:**
```bash
# Find what's using the port
netstat -ano | findstr :8000

# Kill the process (replace PID with actual number)
taskkill /PID <PID> /F

# Or use different ports
# Edit docker-compose.yml and change ports
```

---

### **3. API Key Issues**

#### **Problem: "API key not working"**
```
Invalid API key or quota exceeded
```

**Solution:**
1. **Check your .env file:**
   ```bash
   notepad .env
   ```

2. **Verify API keys are correct:**
   - No extra spaces
   - No quotes around the key
   - Key is complete

3. **Get new API keys:**
   - **Google Gemini**: [Google AI Studio](https://makersuite.google.com/app/apikey)
   - **Kaggle**: [Kaggle Settings](https://www.kaggle.com/settings) → API → Create New Token

4. **Test API keys:**
   ```bash
   python -c "import google.generativeai as genai; genai.configure(api_key='YOUR_KEY'); print('✅ Gemini API works!')"
   ```

---

### **4. Import Errors**

#### **Problem: "No module named 'whisper'"**
```
ModuleNotFoundError: No module named 'whisper'
```

**Solution:**
```bash
# Install the correct whisper package
pip uninstall whisper -y
pip install openai-whisper
```

#### **Problem: "Cannot import from code"**
```
ImportError: cannot import name 'MAVLinkGenerator'
```

**Solution:**
```bash
# This is already fixed in the latest version
# If you still see this error, update your code:
git pull origin main
```

#### **Problem: "No module named 'ultralytics'"**
```
ModuleNotFoundError: No module named 'ultralytics'
```

**Solution:**
```bash
# Install missing dependencies
pip install -r requirements.txt

# Or install individually
pip install ultralytics
```

---

### **5. File Permission Issues**

#### **Problem: "Permission denied"**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
1. **Run as Administrator:**
   - Right-click Command Prompt
   - Select "Run as administrator"

2. **Check file permissions:**
   - Make sure you own the project folder
   - Check if files are read-only

3. **Use different directory:**
   ```bash
   # Move project to a user directory
   cd C:\Users\YourName\Documents
   ```

---

### **6. Network Issues**

#### **Problem: "Failed to download models"**
```
ConnectionError: Failed to download model
```

**Solution:**
1. **Check internet connection**
2. **Use VPN if needed** (some regions block certain services)
3. **Download manually:**
   ```bash
   # Download models manually and place in models/ folder
   ```

#### **Problem: "Docker pull failed"**
```
failed to copy: httpReadSeeker: failed open
```

**Solution:**
1. **Check Docker network:**
   ```bash
   docker network ls
   ```

2. **Restart Docker:**
   - Close Docker Desktop
   - Restart your computer
   - Open Docker Desktop again

3. **Use different registry:**
   ```bash
   # Edit docker-compose.yml to use different image sources
   ```

---

### **7. Memory Issues**

#### **Problem: "Out of memory"**
```
CUDA out of memory
```

**Solution:**
1. **Reduce batch size:**
   ```bash
   # Edit configs/ to use smaller batch sizes
   batch_size: 8  # Instead of 32
   ```

2. **Use CPU instead of GPU:**
   ```bash
   # Set in .env file
   DEVICE=cpu
   ```

3. **Close other applications** to free up memory

---

### **8. Database Issues**

#### **Problem: "Database connection failed"**
```
ConnectionError: Could not connect to database
```

**Solution:**
1. **Check if database is running:**
   ```bash
   docker-compose ps
   ```

2. **Restart database:**
   ```bash
   docker-compose restart db
   ```

3. **Check database URL in .env:**
   ```bash
   DATABASE_URL=postgresql://agrispray:password@localhost:5432/agrispray
   ```

---

## 🔧 **Diagnostic Commands**

### **Check System Status:**
```bash
# Check Python version
python --version

# Check Docker status
docker --version
docker info

# Check if ports are free
netstat -ano | findstr :8000
netstat -ano | findstr :3000

# Check disk space
dir C:\

# Check memory usage
tasklist /fi "imagename eq python.exe"
```

### **Test Individual Components:**
```bash
# Test Python imports
python -c "import torch; print('✅ PyTorch works')"
python -c "import whisper; print('✅ Whisper works')"
python -c "import google.generativeai; print('✅ Gemini works')"

# Test API server
python -c "from code.api.server import app; print('✅ Server imports work')"

# Test Docker
docker run hello-world
```

---

## 📞 **Getting Help**

### **Before Asking for Help:**
1. **Check this guide** - Most issues are covered here
2. **Check the logs** - Look for error messages in terminal
3. **Try the diagnostic commands** above
4. **Restart everything** - Sometimes a simple restart fixes issues

### **When Asking for Help, Include:**
1. **Your operating system** (Windows 10/11)
2. **Python version** (`python --version`)
3. **Docker version** (`docker --version`)
4. **Exact error message** (copy and paste)
5. **What you were doing** when the error occurred
6. **Steps you've already tried**

### **Useful Resources:**
- **Docker Documentation**: [docs.docker.com](https://docs.docker.com/)
- **Python Documentation**: [docs.python.org](https://docs.python.org/)
- **FastAPI Documentation**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)
- **Google Gemini API**: [ai.google.dev](https://ai.google.dev/)

---

## 🎯 **Quick Fixes**

### **"Everything is broken" - Nuclear Option:**
```bash
# 1. Stop everything
docker-compose down
taskkill /f /im python.exe

# 2. Clean up
docker system prune -a
rmdir /s venv

# 3. Start fresh
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python setup_environment.py
python start_project.py
```

### **"Just want to test quickly":**
```bash
# Use Docker (easiest)
docker-compose up --build

# Or use the startup script
python start_project.py
```

---

## ✅ **Success Checklist**

Your system is working correctly when:
- [ ] `python --version` shows 3.9-3.11
- [ ] `docker --version` works
- [ ] `.env` file has real API keys
- [ ] `python start_project.py` runs without errors
- [ ] Web interface opens at http://localhost:3000
- [ ] API docs open at http://localhost:8000/docs
- [ ] You can upload an image and get results

---

*If you're still having issues after trying these solutions, please ask for help with the specific error message and your system details.*
