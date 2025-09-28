# 🚀 AgriSprayAI Quick Start Guide

**Get AgriSprayAI running in 5 minutes!**

## 🎯 **Choose Your Setup Method**

### **Method 1: Super Easy (Recommended) ⭐**
```bash
# Just double-click this file:
start_project.bat
```

### **Method 2: Docker (One Command)**
```bash
docker-compose up --build
```

### **Method 3: Interactive Script**
```bash
python start_project.py
```

---

## 📋 **Prerequisites Checklist**

Before starting, make sure you have:

- [ ] **Docker Desktop** installed and running (for Method 1 & 2)
- [ ] **Python 3.9-3.11** installed (NOT 3.13)
- [ ] **Git** installed
- [ ] **API Keys** ready (see below)

---

## 🔑 **Get Your API Keys (2 minutes)**

### **1. Google Gemini API Key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key (starts with `AIza...`)

### **2. Kaggle API Key:**
1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
2. Scroll to "API" section
3. Click "Create New Token"
4. Download the `kaggle.json` file
5. Open the file and copy the API key

---

## ⚡ **5-Minute Setup**

### **Step 1: Set Up Environment (1 minute)**
```bash
# Run the environment setup script
python setup_environment.py

# Follow the prompts to enter your API keys
# The script will create a secure .env file
```

### **Step 2: Start the System (2 minutes)**
```bash
# Choose one of these methods:

# Method A: Super Easy
start_project.bat

# Method B: Docker
docker-compose up --build

# Method C: Interactive
python start_project.py
```

### **Step 3: Access the Application (1 minute)**
- **Web Interface**: Open http://localhost:3000
- **API Documentation**: Open http://localhost:8000/docs

### **Step 4: Test the System (1 minute)**
1. Go to the web interface
2. Upload a test image (any plant/crop image)
3. Add some text notes (e.g., "I see some brown spots on the leaves")
4. Click "Analyze"
5. You should see results!

---

## 🎉 **Success! What's Next?**

### **Explore the System:**
- **Upload different images** - Try various crop photos
- **Test voice input** - Record audio notes about your crops
- **Check the API** - Use the documentation at http://localhost:8000/docs
- **View results** - See how the AI analyzes your images and text

### **Learn More:**
- **Read the full guide**: [BEGINNER_SETUP_GUIDE.md](BEGINNER_SETUP_GUIDE.md)
- **Troubleshooting**: [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)
- **Technical docs**: Check the `docs/` folder

---

## 🆘 **Quick Troubleshooting**

### **"Docker not found"**
- Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/)
- Restart your computer after installation

### **"Python version error"**
- Use Python 3.9-3.11 (download from [python.org](https://python.org/downloads/))
- During installation, check "Add Python to PATH"

### **"API key not working"**
- Check your `.env` file has real API keys (no extra spaces)
- Make sure the keys are complete and valid

### **"Port already in use"**
```bash
# Kill the process using the port
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### **"Permission denied"**
- Run Command Prompt as Administrator
- Or move the project to your Documents folder

---

## 📱 **What You Can Do Now**

### **Image Analysis:**
- Upload crop photos
- Detect pests and diseases
- Get severity assessments (0-3 scale)
- See bounding boxes around problems

### **Text Processing:**
- Enter farmer notes
- Process voice recordings
- Extract symptoms and causes
- Get treatment recommendations

### **Dose Optimization:**
- Calculate precise pesticide amounts
- Minimize chemical usage
- Ensure safety compliance
- Generate spray plans

### **Flight Planning:**
- Create drone flight paths
- Optimize spray routes
- Generate MAVLink missions
- Plan efficient coverage

---

## 🔧 **Advanced Usage**

### **API Integration:**
```bash
# Test the API directly
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@your_image.jpg" \
  -F "notes=Brown spots on leaves"
```

### **Batch Processing:**
```bash
# Process multiple images
python code/vision/batch_analyze.py --input_dir ./test_images
```

### **Model Training:**
```bash
# Train your own models
python code/vision/train_detector.py --config configs/yolov8_baseline.yaml
```

---

## 📊 **System Status**

### **Check if everything is running:**
```bash
# Check Docker containers
docker-compose ps

# Check API health
curl http://localhost:8000/health

# Check web interface
curl http://localhost:3000
```

### **View logs:**
```bash
# Docker logs
docker-compose logs -f

# Python logs
tail -f logs/app.log
```

---

## 🎯 **Performance Tips**

### **For Better Speed:**
- Use GPU if available (set `DEVICE=cuda` in .env)
- Increase batch size for multiple images
- Use smaller image sizes for faster processing

### **For Better Accuracy:**
- Use high-quality images
- Provide detailed text descriptions
- Use multiple images of the same problem
- Train models on your specific crop types

---

## 🔒 **Security Notes**

- **Never share your .env file** - It contains your API keys
- **Use HTTPS in production** - Set up SSL certificates
- **Regular backups** - Backup your models and data
- **Monitor usage** - Check API key quotas regularly

---

## 📈 **Scaling Up**

### **For Production:**
- Use Docker Swarm or Kubernetes
- Set up proper monitoring (Prometheus/Grafana)
- Implement proper logging and alerting
- Use production databases (PostgreSQL, Redis)

### **For Multiple Users:**
- Set up user authentication
- Implement rate limiting
- Use load balancers
- Set up proper backup strategies

---

## 🎊 **Congratulations!**

You now have a fully functional AI-powered agricultural system! 

**AgriSprayAI is helping farmers:**
- 🌱 **Detect problems early** with AI vision
- 💊 **Use pesticides efficiently** with precise dosing
- 🚁 **Plan drone missions** for optimal coverage
- ✅ **Ensure safety** with human oversight

**Happy farming! 🌾🤖**

---

*Need help? Check [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md) or ask for assistance with your specific issue.*
