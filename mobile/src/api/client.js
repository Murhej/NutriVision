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

// Request timeout in milliseconds
const REQUEST_TIMEOUT_MS = 15000; // 15 seconds for file uploads, 8 seconds for regular requests
const REGULAR_TIMEOUT_MS = 8000;

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
 * Utility to wrap fetch with timeout
 */
const fetchWithTimeout = async (url, options, timeoutMs) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error(`Request timeout (${timeoutMs}ms). Check your internet connection or server availability.`);
    }
    throw error;
  }
};

/**
 * Super simple wrapper around fetch to automatically prepend the base URL
 * and handle JSON parsing / error catching cleanly.
 */
export const apiClient = {
  get: async (endpoint) => {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}${endpoint}`,
        {
          method: 'GET',
          headers: getHeaders(false),
        },
        REGULAR_TIMEOUT_MS
      );
      
      if (!response.ok) {
        const text = await response.text();
        let detail = `HTTP ${response.status}`;
        try {
          const json = JSON.parse(text);
          detail = json.detail || detail;
        } catch (e) {
          // text response
        }
        throw new Error(detail);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`[GET ${endpoint}] Error:`, error.message);
      throw new Error(error.message || 'Failed to fetch data');
    }
  },

  post: async (endpoint, body) => {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}${endpoint}`,
        {
          method: 'POST',
          headers: getHeaders(false),
          body: JSON.stringify(body),
        },
        REGULAR_TIMEOUT_MS
      );
      
      if (!response.ok) {
        const text = await response.text();
        let detail = `HTTP ${response.status}`;
        try {
          const json = JSON.parse(text);
          detail = json.detail || detail;
        } catch (e) {
          // text response
        }
        throw new Error(detail);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`[POST ${endpoint}] Error:`, error.message);
      throw new Error(error.message || 'Failed to post data');
    }
  },

  uploadImage: async (endpoint, uri) => {
    try {
      // Determine file extension and type
      const filename = uri.split('/').pop();
      const match = /\.(\w+)$/.exec(filename);
      const ext = match ? match[1].toLowerCase() : 'jpg';
      const mimeType = {
        jpg: 'image/jpeg',
        jpeg: 'image/jpeg',
        png: 'image/png',
        gif: 'image/gif',
        webp: 'image/webp',
      }[ext] || 'image/jpeg';

      const formData = new FormData();
      formData.append('file', {
        uri,
        name: filename,
        type: mimeType,
      });

      const response = await fetchWithTimeout(
        `${API_BASE_URL}${endpoint}`,
        {
          method: 'POST',
          body: formData,
          headers: getHeaders(true),
        },
        REQUEST_TIMEOUT_MS // Longer timeout for file uploads
      );
      
      if (!response.ok) {
        const text = await response.text();
        let detail = `HTTP ${response.status}`;
        try {
          const json = JSON.parse(text);
          detail = json.detail || detail;
        } catch (e) {
          // text response
        }
        throw new Error(detail);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`[UPLOAD ${endpoint}] Error:`, error.message);
      throw new Error(error.message || 'Failed to upload image');
    }
  },
};
