import axios from 'axios';

// API base URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API functions
export const apiService = {
  // Health check
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  // Upload image and get predictions
  async predict(imageFile, transcript = null, metadata = null) {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    if (transcript) {
      formData.append('transcript', transcript);
    }
    
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata));
    }

    const response = await api.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  },

  // Generate flight plan
  async plan(predictions, location, constraints = null, fieldInfo = null) {
    const requestData = {
      predictions,
      location,
      constraints,
      field_info: fieldInfo,
    };

    const response = await api.post('/plan', requestData);
    return response.data;
  },

  // Approve or reject plan
  async approve(runId, decision, operatorId, comments = null, signature = null) {
    const requestData = {
      run_id: runId,
      decision,
      operator_id: operatorId,
      comments,
      signature,
    };

    const response = await api.post('/approve', requestData);
    return response.data;
  },

  // Get logs for a run
  async getLogs(runId) {
    const response = await api.get(`/logs/${runId}`);
    return response.data;
  },

  // Get explainability report
  async getExplainability(runId) {
    const response = await api.get(`/explain/${runId}`);
    return response.data;
  },

  // Upload audio file for transcription
  async transcribeAudio(audioFile) {
    const formData = new FormData();
    formData.append('audio', audioFile);

    const response = await api.post('/transcribe', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  },

  // Get model status
  async getModelStatus() {
    const response = await api.get('/models/status');
    return response.data;
  },

  // Get system metrics
  async getMetrics() {
    const response = await api.get('/metrics');
    return response.data;
  },
};

// Export individual functions for convenience
export const {
  healthCheck,
  predict,
  plan,
  approve,
  getLogs,
  getExplainability,
  transcribeAudio,
  getModelStatus,
  getMetrics,
} = apiService;

export default api;
