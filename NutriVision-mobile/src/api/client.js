import { Platform } from 'react-native';
import Constants from 'expo-constants';

function resolveApiBaseUrl() {
  const explicit = process.env.EXPO_PUBLIC_API_BASE_URL;
  if (explicit) return explicit;

  const hostUri =
    Constants?.expoConfig?.hostUri ||
    Constants?.manifest2?.extra?.expoClient?.hostUri ||
    Constants?.manifest?.hostUri;

  if (hostUri) {
    const host = String(hostUri).split(':')[0];
    if (host) return `http://${host}:8000`;
  }

  if (Platform.OS === 'android') return 'http://10.0.2.2:8000';
  return 'http://localhost:8000';
}

export const API_BASE_URL = resolveApiBaseUrl();
const REQUEST_TIMEOUT_MS = 20000;

function getFallbackBaseUrls(primaryBaseUrl) {
  // Try 8000, 8002, 8001 in order to handle common server configurations.
  const host = primaryBaseUrl.replace(/:\d+$/, '');
  const ports = ['8000', '8002', '8001'];
  const primary = primaryBaseUrl.match(/:(\d+)$/)?.[1] || '8000';
  const ordered = [primary, ...ports.filter((p) => p !== primary)];
  return Array.from(new Set(ordered.map((p) => `${host}:${p}`)));
}

function shouldRetryWithFallback(error) {
  // Retry on transient failures and compatibility misses on alternate ports.
  const status = Number(error?.status || 0);
  if ([404, 405, 408, 429, 500, 502, 503, 504].includes(status)) return true;

  const message = String(error?.message || '').toLowerCase();
  return (
    error instanceof TypeError ||
    message.includes('network request failed') ||
    message.includes('request timeout') ||
    message.includes('aborted') ||
    message.includes('failed to fetch')
  );
}

function mapLegacyProfilePayload(payload = {}) {
  const legacy = {};

  if (payload.fullName !== undefined) legacy.name = payload.fullName;
  if (payload.username !== undefined) legacy.username = payload.username;
  if (payload.phone !== undefined) legacy.phone = payload.phone;
  if (payload.country !== undefined) legacy.country = payload.country;
  if (payload.goal !== undefined) legacy.goal = payload.goal;
  if (payload.goalType !== undefined) legacy.goalType = payload.goalType;
  if (payload.activityMultiplier !== undefined) legacy.activityMultiplier = payload.activityMultiplier;
  if (payload.avatar !== undefined) legacy.avatar = payload.avatar;

  if (payload.dailyCalorieGoal !== undefined) legacy.dailyCalorieGoal = payload.dailyCalorieGoal;
  if (payload.proteinGoal !== undefined) legacy.proteinGoal = payload.proteinGoal;
  if (payload.carbsGoal !== undefined) legacy.carbsGoal = payload.carbsGoal;
  if (payload.fatGoal !== undefined) legacy.fatGoal = payload.fatGoal;
  if (payload.nutrientTargets !== undefined) legacy.nutrientTargets = payload.nutrientTargets;
  if (payload.nutrientTargetModes !== undefined) legacy.nutrientTargetModes = payload.nutrientTargetModes;

  if (payload?.settings?.units !== undefined) {
    legacy.unitSystem = payload.settings.units;
  }

  return legacy;
}

function normalizeProfileResponse(data = {}, requestedPatch = null) {
  const profile = data?.profile && typeof data.profile === 'object' ? { ...data.profile } : {};

  if (!profile.name && requestedPatch?.fullName) {
    profile.name = requestedPatch.fullName;
  }

  if (requestedPatch?.email && !profile.email) {
    profile.email = requestedPatch.email;
  }

  if (!profile.settings || typeof profile.settings !== 'object') {
    profile.settings = {};
  }

  if (!profile.settings.units && profile.unitSystem) {
    profile.settings.units = profile.unitSystem;
  }

  return {
    ...data,
    profile,
  };
}

async function parseResponse(response, { suppressHttpErrorLog = false } = {}) {
  const text = await response.text();
  let parsed = null;

  try {
    parsed = text ? JSON.parse(text) : null;
  } catch {
    parsed = { detail: text || 'Unknown response error' };
  }

  if (!response.ok) {
    const detail = parsed?.detail || parsed?.message || `Request failed with status ${response.status}`;
    const error = new Error(detail);
    error.status = response.status;
    error.payload = parsed;
    // Log for debugging - remove in production
    if (!suppressHttpErrorLog && response.status >= 400) {
      console.warn(`[API Error] ${response.status} ${response.url}:`, {detail, payload: parsed});
    }
    throw error;
  }

  return parsed;
}

export async function apiRequest(path, { method = 'GET', token, body, headers = {}, isFormData = false, timeoutMs = REQUEST_TIMEOUT_MS, suppressHttpErrorLog = false } = {}) {
  const requestHeaders = {
    ...headers,
  };

  if (!isFormData) {
    requestHeaders['Content-Type'] = 'application/json';
  }

  if (token) {
    requestHeaders.Authorization = `Bearer ${token}`;
  }

  const candidateBases = getFallbackBaseUrls(API_BASE_URL);
  let lastError = null;

  for (let i = 0; i < candidateBases.length; i += 1) {
    const base = candidateBases[i];
    let timeoutId;
    try {
      const controller = new AbortController();
      timeoutId = setTimeout(() => controller.abort(), timeoutMs);
      const response = await fetch(`${base}${path}`, {
        method,
        headers: requestHeaders,
        signal: controller.signal,
        body: body
          ? isFormData
            ? body
            : JSON.stringify(body)
          : undefined,
      });
      clearTimeout(timeoutId);

      return await parseResponse(response, { suppressHttpErrorLog });
    } catch (error) {
      if (timeoutId) clearTimeout(timeoutId);
      lastError = error;
      const canRetry = i < candidateBases.length - 1 && shouldRetryWithFallback(error);
      if (!canRetry) throw error;
    }
  }

  throw lastError;
}

export const authApi = {
  register: (payload) => apiRequest('/api/auth/register', { method: 'POST', body: payload, timeoutMs: 30000 }),
  login: (payload) => apiRequest('/api/auth/login', { method: 'POST', body: payload, timeoutMs: 30000 }),
  logout: (token) => apiRequest('/api/auth/logout', { method: 'POST', token }),
  deleteAccount: async (token) => {
    try {
      return await apiRequest('/api/auth/delete-account', { method: 'DELETE', token });
    } catch (error) {
      if (error?.status === 404 || error?.status === 405) {
        return apiRequest('/api/mobile/profile', { method: 'DELETE', token });
      }
      throw error;
    }
  },
  checkUsernameAvailable: async (token, username) => {
    try {
      return await apiRequest(`/api/auth/username-available?username=${encodeURIComponent(username)}`, { token });
    } catch (error) {
      if (error?.status === 404 || error?.status === 405) {
        return { username, available: true };
      }
      throw error;
    }
  },
};

export const mobileApi = {
  getProfile: (token) => apiRequest('/api/mobile/profile', { token }),
  getAchievements: async (token) => {
    try {
      return await apiRequest('/api/mobile/achievements', {
        token,
        suppressHttpErrorLog: true,
      });
    } catch (error) {
      if ([404, 405].includes(Number(error?.status || 0))) {
        const profileData = await apiRequest('/api/mobile/profile', { token });
        return { achievements: Array.isArray(profileData?.achievements) ? profileData.achievements : [] };
      }
      throw error;
    }
  },
  patchAchievements: async (token, achievements) => {
    const nextAchievements = Array.isArray(achievements) ? achievements : [];
    try {
      return await apiRequest('/api/mobile/achievements', {
        method: 'PATCH',
        token,
        body: { achievements: nextAchievements },
        suppressHttpErrorLog: true,
      });
    } catch (error) {
      if ([404, 405].includes(Number(error?.status || 0))) {
        return apiRequest('/api/mobile/profile', {
          method: 'PATCH',
          token,
          body: { achievements: nextAchievements },
        });
      }
      throw error;
    }
  },
  patchProfile: async (token, payload) => {
    try {
      const result = await apiRequest('/api/mobile/profile', { method: 'PATCH', token, body: payload });
      return normalizeProfileResponse(result, payload);
    } catch (patchError) {
      // Silently try POST as fallback if PATCH fails for any reason
      try {
        const result = await apiRequest('/api/mobile/profile', { method: 'POST', token, body: payload });
        return normalizeProfileResponse(result, payload);
      } catch (postError) {
        // If both PATCH and POST fail, throw the original error
        throw patchError;
      }
    }
  },
  changePassword: async (token, payload) => {
    try {
      return await apiRequest('/api/mobile/profile/change-password', { method: 'POST', token, body: payload });
    } catch (error) {
      if (error?.status === 404 || error?.status === 405) {
        return apiRequest('/api/auth/change-password', { method: 'POST', token, body: payload });
      }
      throw error;
    }
  },
  getFeed: async (token, topic = 'All') => {
    try {
      return await apiRequest(`/api/mobile/feed?topic=${encodeURIComponent(topic)}`, { token });
    } catch (error) {
      if (error?.status === 404 || error?.status === 405) {
        return {
          topics: ['All'],
          activeTopic: 'All',
          hasEnoughData: false,
          preparationMessage: 'Personalized feed is not available on this server yet.',
          articles: [],
        };
      }
      throw error;
    }
  },
  getLeaderboard: (token, scope = 'worldwide') =>
    apiRequest(`/api/mobile/leaderboard?scope=${encodeURIComponent(scope)}`, { token }),
  uploadAvatar: async (token, fileUri) => {
    const uri = String(fileUri || '');
    const cleanUri = uri.split('?')[0];
    const extMatch = cleanUri.match(/\.([a-zA-Z0-9]+)$/);
    const rawExt = (extMatch?.[1] || 'jpg').toLowerCase();
    const normalizedExt = ['jpg', 'jpeg', 'png', 'webp'].includes(rawExt) ? rawExt : 'jpg';
    const mime = normalizedExt === 'jpg' ? 'image/jpeg' : `image/${normalizedExt}`;
    const form = new FormData();
    form.append('file', {
      uri,
      name: `avatar.${normalizedExt}`,
      type: mime,
    });
    try {
      return await apiRequest('/api/mobile/profile/avatar', { method: 'POST', token, body: form, isFormData: true });
    } catch (error) {
      if (error?.status === 404 || error?.status === 405) {
        const fallbackError = new Error('Unable to update profile photo right now.');
        fallbackError.status = error.status;
        fallbackError.payload = error.payload;
        throw fallbackError;
      }
      throw error;
    }
  },
};

// Timeout constants for network resilience
const FOOD_PREDICTION_TIMEOUT_MS = 120000; // CPU inference + slow mobile networks
const NUTRITION_FETCH_TIMEOUT_MS = 15000;
const MEAL_LOG_TIMEOUT_MS = 15000;

/**
 * Fetch wrapper with timeout support
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
 * Food & Nutrition API client with timeout handling
 */
export const apiClient = {
  get: async (endpoint) => {
    const candidateBases = getFallbackBaseUrls(API_BASE_URL);
    let lastError = null;
    for (let i = 0; i < candidateBases.length; i += 1) {
      const base = candidateBases[i];
      try {
        const response = await fetchWithTimeout(
          `${base}${endpoint}`,
          {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            },
          },
          NUTRITION_FETCH_TIMEOUT_MS
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
          const err = new Error(detail);
          err.status = response.status;
          throw err;
        }

        return await response.json();
      } catch (error) {
        lastError = error;
        const canRetry = i < candidateBases.length - 1 && shouldRetryWithFallback(error);
        if (!canRetry) {
          console.error(`[GET ${endpoint}] Error:`, error.message);
          throw new Error(error.message || `Failed to fetch data from ${base}`);
        }
      }
    }
    throw lastError;
  },

  post: async (endpoint, body) => {
    const candidateBases = getFallbackBaseUrls(API_BASE_URL);
    let lastError = null;
    for (let i = 0; i < candidateBases.length; i += 1) {
      const base = candidateBases[i];
      try {
        const response = await fetchWithTimeout(
          `${base}${endpoint}`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
          },
          NUTRITION_FETCH_TIMEOUT_MS
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
          const err = new Error(detail);
          err.status = response.status;
          throw err;
        }

        return await response.json();
      } catch (error) {
        lastError = error;
        const canRetry = i < candidateBases.length - 1 && shouldRetryWithFallback(error);
        if (!canRetry) {
          console.error(`[POST ${endpoint}] Error:`, error.message);
          throw new Error(error.message || `Failed to post data to ${base}`);
        }
      }
    }
    throw lastError;
  },

  uploadImage: async (endpoint, uri) => {
    const candidateBases = getFallbackBaseUrls(API_BASE_URL);
    let lastError = null;
    for (let i = 0; i < candidateBases.length; i += 1) {
      const base = candidateBases[i];
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
          `${base}${endpoint}`,
          {
            method: 'POST',
            body: formData,
            headers: {
              // Don't set Content-Type for FormData - let runtime set boundary
            },
          },
          FOOD_PREDICTION_TIMEOUT_MS
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
          const err = new Error(detail);
          err.status = response.status;
          throw err;
        }

        return await response.json();
      } catch (error) {
        lastError = error;
        const canRetry = i < candidateBases.length - 1 && shouldRetryWithFallback(error);
        if (!canRetry) {
          console.error(`[UPLOAD ${endpoint}] Error:`, error.message);
          const msg = error.message || `Failed to upload image to ${base}`;
          throw new Error(msg);
        }
      }
    }
    throw lastError;
  },
};
