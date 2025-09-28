# 🐳 Docker Fix Guide for AgriSprayAI

## 🚨 **Issue Identified**

Your Docker is configured to use a custom registry (`docker-images-prod.6aa30f8b08e16409b46e0173d5de2f56.r2.cloudflarestorage.com`) instead of the standard Docker Hub. This is causing the "no such host" error.

## 🔧 **Solutions (Try in Order)**

### **Solution 1: Reset Docker Desktop (Recommended) 🥇**

1. **Open Docker Desktop**
2. **Click the Settings gear icon** (top right)
3. **Go to "Troubleshoot"** in the left menu
4. **Click "Reset to factory defaults"**
5. **Click "Reset"** and wait for Docker to restart
6. **Try again:**
   ```bash
   docker-compose up --build
   ```

### **Solution 2: Change Docker Registry Settings 🥈**

1. **Open Docker Desktop**
2. **Go to Settings** (gear icon)
3. **Go to "Docker Engine"** in the left menu
4. **Look for registry configuration** and remove any custom registries
5. **Click "Apply & Restart"**
6. **Try again:**
   ```bash
   docker-compose up --build
   ```

### **Solution 3: Use Python Setup Instead (Quick Alternative) 🥉**

Since Docker is having issues, let's use the Python setup:

```bash
# 1. Set up environment
python setup_environment.py

# 2. Start with Python
python start_project.py
```

### **Solution 4: Manual Docker Registry Fix 🔧**

If the above doesn't work, try this:

1. **Open Command Prompt as Administrator**
2. **Run these commands:**
   ```bash
   # Stop Docker
   net stop com.docker.service
   
   # Clear Docker cache
   docker system prune -a -f
   
   # Reset Docker networks
   docker network prune -f
   
   # Start Docker Desktop again
   # Then try:
   docker-compose up --build
   ```

### **Solution 5: Use Different Network 🌐**

Sometimes network restrictions cause this issue:

1. **Try using mobile hotspot** instead of your regular internet
2. **Or use a VPN** to change your network location
3. **Then try:**
   ```bash
   docker-compose up --build
   ```

## 🚀 **Alternative: Python Setup (No Docker)**

If Docker continues to have issues, you can run the project directly with Python:

### **Step 1: Set Up Environment**
```bash
# Run the environment setup
python setup_environment.py

# Follow the prompts to enter your API keys
```

### **Step 2: Start the System**
```bash
# Use the interactive startup script
python start_project.py

# Choose option 2 (Python setup)
```

### **Step 3: Access the Application**
- **Web Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs

## 🧪 **Test Docker Fix**

After trying any solution, test if Docker is working:

```bash
# Test pulling a simple image
docker pull hello-world

# If that works, try the project
docker-compose up --build
```

## 🆘 **Still Having Issues?**

### **Check These:**
1. **Windows Firewall** - Make sure Docker is allowed through
2. **Antivirus Software** - Some antivirus blocks Docker
3. **Corporate Network** - Some corporate networks block Docker registries
4. **DNS Settings** - Try using Google DNS (8.8.8.8, 8.8.4.4)

### **Nuclear Option:**
```bash
# Complete Docker reset
# 1. Uninstall Docker Desktop
# 2. Delete C:\Users\YourName\AppData\Roaming\Docker Desktop
# 3. Delete C:\Users\YourName\AppData\Local\Docker
# 4. Restart computer
# 5. Reinstall Docker Desktop
# 6. Try again
```

## ✅ **Success Indicators**

You'll know Docker is fixed when:
- ✅ `docker pull hello-world` works
- ✅ `docker-compose up --build` starts without errors
- ✅ You can see containers running with `docker ps`
- ✅ Web interface opens at http://localhost:3000

## 🎯 **Recommended Next Steps**

1. **Try Solution 1** (Reset Docker Desktop) first
2. **If that doesn't work**, try Solution 3 (Python setup)
3. **Python setup is actually easier** for beginners anyway
4. **Once you're comfortable**, you can come back to Docker

## 💡 **Why This Happened**

This usually happens when:
- Docker Desktop was configured for a corporate environment
- A previous Docker installation had custom registry settings
- Network configuration changed
- Docker Desktop was updated and kept old settings

The good news is that **Python setup works just as well** and is often easier for beginners!

---

*Choose the solution that works best for you. The Python setup is actually recommended for beginners as it's simpler and more transparent.*
