import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { authApi, mobileApi } from '../api/client';
import { useTheme } from '../theme/ThemeContext';

const STORAGE_KEY = 'nutrivision.session.v1';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const { setDarkMode } = useTheme();
  const [isLoading, setIsLoading] = useState(true);
  const [token, setToken] = useState(null);
  const [user, setUser] = useState(null);
  const [profile, setProfile] = useState(null);
  const [achievements, setAchievements] = useState([]);

  const safeGetItem = useCallback(async (key) => {
    try {
      return await AsyncStorage.getItem(key);
    } catch {
      return null;
    }
  }, []);

  const safeSetItem = useCallback(async (key, value) => {
    try {
      await AsyncStorage.setItem(key, value);
      return true;
    } catch {
      return false;
    }
  }, []);

  const safeRemoveItem = useCallback(async (key) => {
    try {
      await AsyncStorage.removeItem(key);
    } catch {
      // Local state reset still guarantees sign-out.
    }
  }, []);

  const persistSession = useCallback(async (nextToken, nextUser) => {
    await safeSetItem(
      STORAGE_KEY,
      JSON.stringify({
        token: nextToken,
        user: nextUser,
      })
    );
  }, [safeSetItem]);

  const clearLocalSession = useCallback(async () => {
    setToken(null);
    setUser(null);
    setProfile(null);
    setAchievements([]);
    await safeRemoveItem(STORAGE_KEY);
  }, [safeRemoveItem]);

  const resolveAuthToken = useCallback(async (overrideToken = null) => {
    if (overrideToken) return overrideToken;
    if (token) return token;

    const raw = await safeGetItem(STORAGE_KEY);
    if (!raw) return null;

    try {
      const parsed = JSON.parse(raw);
      return parsed?.token || null;
    } catch {
      return null;
    }
  }, [safeGetItem, token]);

  const refreshProfile = useCallback(async (existingToken) => {
    const authToken = await resolveAuthToken(existingToken);
    if (!authToken) return null;

    const data = await mobileApi.getProfile(authToken);
    const fetchedProfile = data.profile || null;

    setProfile((prev) => ({
      ...(prev || {}),
      ...(fetchedProfile || {}),
    }));
    if (fetchedProfile) {
      setUser((prev) => ({
        ...(prev || {}),
        name: fetchedProfile.name || prev?.name,
        email: fetchedProfile.email || prev?.email,
      }));
      if (typeof fetchedProfile?.settings?.darkMode === 'boolean') {
        setDarkMode(Boolean(fetchedProfile.settings.darkMode));
      }
    }
    setAchievements(data.achievements || []);
    return data;
  }, [resolveAuthToken, setDarkMode]);

  useEffect(() => {
    let mounted = true;

    const hydrate = async () => {
      try {
        const raw = await safeGetItem(STORAGE_KEY);
        if (!raw) return;

        const parsed = JSON.parse(raw);
        if (!parsed?.token) return;

        if (!mounted) return;
        setToken(parsed.token);
        setUser(parsed.user || null);

        try {
          await Promise.race([
            refreshProfile(parsed.token),
            new Promise((_, reject) => setTimeout(() => reject(new Error('Profile refresh timeout')), 10000)),
          ]);
        } catch {
          await clearLocalSession();
        }
      } finally {
        if (mounted) setIsLoading(false);
      }
    };

    hydrate();

    return () => {
      mounted = false;
    };
  }, [clearLocalSession, refreshProfile, safeGetItem]);

  const login = useCallback(async ({ email, password }) => {
    const result = await authApi.login({ email, password });
    setToken(result.token);
    setUser(result.user || null);
    await persistSession(result.token, result.user || null);
    await refreshProfile(result.token);
    return result;
  }, [persistSession, refreshProfile]);

  const register = useCallback(async ({ name, email, password }) => {
    const result = await authApi.register({ name, email, password });
    setToken(result.token);
    setUser(result.user || null);
    await persistSession(result.token, result.user || null);
    await refreshProfile(result.token);
    return result;
  }, [persistSession, refreshProfile]);

  const logout = useCallback(async () => {
    if (token) {
      try {
        await authApi.logout(token);
      } catch {
        // Local sign-out still proceeds if backend token was already invalid.
      }
    }
    await clearLocalSession();
  }, [clearLocalSession, token]);

  const deleteAccount = useCallback(async () => {
    if (!token) {
      await clearLocalSession();
      return;
    }
    try {
      await authApi.deleteAccount(token);
      await clearLocalSession();
    } catch (error) {
      if ([401, 404, 405].includes(Number(error?.status || 0))) {
        await clearLocalSession();
        return;
      }
      throw error;
    }
  }, [clearLocalSession, token]);

  const updateProfile = useCallback(async (patch, existingToken = null) => {
    const authToken = await resolveAuthToken(existingToken);
    if (!authToken) throw new Error('Not authenticated');
    const result = await mobileApi.patchProfile(authToken, patch);
    const merged = {
      ...(profile || {}),
      ...(result.profile || {}),
    };

    // Keep key profile fields locally consistent when backend responses omit them.
    if (patch?.fullName) merged.name = patch.fullName;
    if (patch?.username) merged.username = patch.username;
    if (patch?.email) merged.email = patch.email;
    if (patch?.phone !== undefined) merged.phone = patch.phone;
    if (patch?.country !== undefined) merged.country = patch.country;
    if (patch?.goal !== undefined) merged.goal = patch.goal;
    if (patch?.dietaryPreferences) merged.dietaryPreferences = patch.dietaryPreferences;
    if (patch?.allergies) merged.allergies = patch.allergies;
    if (patch?.activityLevel !== undefined) merged.activityLevel = patch.activityLevel;
    if (patch?.exerciseHabits !== undefined) merged.exerciseHabits = patch.exerciseHabits;

    // Keep local settings in sync when older backend contracts do not echo these fields.
    if (patch?.settings) {
      merged.settings = {
        ...(profile?.settings || {}),
        ...(result?.profile?.settings || {}),
        ...patch.settings,
      };
      if (patch.settings.units) {
        merged.unitSystem = patch.settings.units;
      }
    }

    setProfile(merged);
    const nextUser = {
      ...(user || {}),
      name: merged.name || user?.name,
      email: merged.email || user?.email,
    };
    setUser(nextUser);
    if (typeof merged?.settings?.darkMode === 'boolean') {
      setDarkMode(Boolean(merged.settings.darkMode));
    }
    await persistSession(authToken, nextUser);
    return merged;
  }, [persistSession, profile, resolveAuthToken, setDarkMode, user]);

  const checkUsernameAvailable = useCallback(async (username, existingToken = null) => {
    const authToken = await resolveAuthToken(existingToken);
    if (!authToken) throw new Error('Not authenticated');
    const result = await authApi.checkUsernameAvailable(authToken, username);
    return Boolean(result?.available);
  }, [resolveAuthToken]);

  const uploadAvatar = useCallback(async (fileUri, existingToken = null) => {
    const authToken = await resolveAuthToken(existingToken);
    if (!authToken) throw new Error('Not authenticated');
    const result = await mobileApi.uploadAvatar(authToken, fileUri);
    const avatar = result?.avatar || null;
    if (avatar) {
      setProfile((prev) => ({ ...(prev || {}), avatar }));
    }
    return avatar;
  }, [resolveAuthToken]);

  const changePassword = useCallback(async (payload, existingToken = null) => {
    const authToken = await resolveAuthToken(existingToken);
    if (!authToken) throw new Error('Not authenticated');
    return mobileApi.changePassword(authToken, payload);
  }, [resolveAuthToken]);

  const fetchFeed = useCallback(async (topic = 'All') => {
    if (!token) throw new Error('Not authenticated');
    return mobileApi.getFeed(token, topic);
  }, [token]);

  const fetchLeaderboard = useCallback(async (scope = 'worldwide') => {
    if (!token) throw new Error('Not authenticated');
    return mobileApi.getLeaderboard(token, scope);
  }, [token]);

  const saveAchievements = useCallback(async (nextAchievements, existingToken = null) => {
    const authToken = await resolveAuthToken(existingToken);
    if (!authToken) throw new Error('Not authenticated');

    const normalizedNext = Array.isArray(nextAchievements) ? nextAchievements : [];

    try {
      const result = await mobileApi.patchAchievements(authToken, normalizedNext);
      if (Array.isArray(result?.achievements)) {
        setAchievements(result.achievements);
      } else {
        setAchievements(normalizedNext);
      }
      refreshProfile(authToken).catch(() => {});
      return result;
    } catch (error) {
      if ([404, 405].includes(Number(error?.status || 0))) {
        try {
          const fallback = await mobileApi.patchProfile(authToken, { achievements: normalizedNext });
          if (Array.isArray(fallback?.achievements)) {
            setAchievements(fallback.achievements);
          } else {
            setAchievements(normalizedNext);
          }
          refreshProfile(authToken).catch(() => {});
          return fallback;
        } catch (fallbackError) {
          throw fallbackError;
        }
      }
      throw error;
    }
  }, [refreshProfile, resolveAuthToken]);

  const upsertAchievement = useCallback(async (achievement, existingToken = null) => {
    if (!achievement?.id) throw new Error('Achievement id is required');
    const now = new Date().toISOString();
    const current = Array.isArray(achievements) ? achievements : [];
    const next = (() => {
      const existing = current.find((a) => a.id === achievement.id);
      if (!existing) return [...current, { ...achievement, updated_at: now }];
      return current.map((a) => (a.id === achievement.id ? { ...a, ...achievement, updated_at: now } : a));
    })();

    // Optimistic local update keeps Home/Profile/Leaderboard in sync instantly.
    setAchievements(next);

    try {
      const persisted = await saveAchievements(next, existingToken);
      const persistedList = Array.isArray(persisted?.achievements) ? persisted.achievements : next;
      return persistedList.find((a) => a.id === achievement.id) || achievement;
    } catch (error) {
      setAchievements(current);
      throw error;
    }
  }, [achievements, saveAchievements]);

  const value = useMemo(() => ({
    isLoading,
    isAuthenticated: Boolean(token),
    token,
    user,
    profile,
    achievements,
    login,
    register,
    logout,
    deleteAccount,
    refreshProfile,
    updateProfile,
    uploadAvatar,
    changePassword,
    checkUsernameAvailable,
    fetchFeed,
    fetchLeaderboard,
    saveAchievements,
    upsertAchievement,
  }), [
    achievements,
    changePassword,
    checkUsernameAvailable,
    deleteAccount,
    fetchFeed,
    fetchLeaderboard,
    isLoading,
    login,
    logout,
    upsertAchievement,
    profile,
    refreshProfile,
    register,
    token,
    updateProfile,
    uploadAvatar,
    user,
    saveAchievements,
  ]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}
