import Constants from 'expo-constants';
import { Platform } from 'react-native';

const getBaseUrl = () => {
  // If the user defined EXPO_PUBLIC_API_URL, use it
  if (process.env.EXPO_PUBLIC_API_URL) {
    return process.env.EXPO_PUBLIC_API_URL;
  }
  
  // Intelligent default resolution for local development
  if (__DEV__) {
    const debuggerHost = Constants.expoConfig?.hostUri;
    
    // Physical device or some custom setups will have hostUri (e.g., 192.168.x.x:8081)
    if (debuggerHost) {
      const ip = debuggerHost.split(':')[0];
      return `http://${ip}:8000`;
    }
    
    // Android Emulator specific bypass
    if (Platform.OS === 'android') {
      return 'http://10.0.2.2:8000';
    }
    
    // iOS Simulator or web fallback
    return 'http://127.0.0.1:8000';
  }
  
  // Replace with production URL eventually
  return 'https://api.nutrivision.app';
};

export const API_BASE_URL = getBaseUrl();

let currentToken = null;

export const setAuthToken = (token) => {
  currentToken = token;
};

const getHeaders = (upload = false) => {
  const headers = {
    'Cache-Control': 'no-cache, no-store, must-revalidate',
    'Pragma': 'no-cache',
    'Expires': '0',
  };
  
  if (!upload) {
    headers['Content-Type'] = 'application/json';
  }
  
  if (currentToken) {
    headers['Authorization'] = `Bearer ${currentToken}`;
  }
  
  return headers;
};

/**
 * Super simple wrapper around fetch to automatically prepend the base URL
 * and handle JSON parsing / error catching cleanly.
 */
export const apiClient = {
  get: async (endpoint) => {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'GET',
        headers: getHeaders(false),
      });
      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error(`[GET ${endpoint}] Error:`, error);
      throw error;
    }
  },
  post: async (endpoint, body) => {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: getHeaders(false),
        body: JSON.stringify(body)
      });
      if (!response.ok) throw new Error(`API Error: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error(`[POST ${endpoint}] Error:`, error);
      throw error;
    }
  },
  uploadImage: async (endpoint, uri) => {
    try {
      // Determine file extension and type
      const filename = uri.split('/').pop();
      const match = /\.(\w+)$/.exec(filename);
      const type = match ? `image/${match[1]}` : `image`;

      const formData = new FormData();
      formData.append('file', {
        uri,
        name: filename,
        type
      });

      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        body: formData,
        headers: getHeaders(true),
      });
      
      if (!response.ok) {
        throw new Error(`Upload Error: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error(`[UPLOAD ${endpoint}] Error:`, error);
      throw error;
    }
  }
};
