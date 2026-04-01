import React, { createContext, useState, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { apiClient, setAuthToken } from '../api/client';

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadToken = async () => {
      try {
        const storedToken = await AsyncStorage.getItem('@auth_token');
        const storedUser = await AsyncStorage.getItem('@auth_user');
        if (storedToken && storedUser) {
          setToken(storedToken);
          setUser(JSON.parse(storedUser));
          setAuthToken(storedToken);
        }
      } catch (e) {
        console.error('Failed to load session:', e);
      } finally {
        setLoading(false);
      }
    };
    loadToken();
  }, []);

  const login = async (email, password) => {
    try {
      const data = await apiClient.post('/api/auth/login', { email, password });
      setToken(data.token);
      setUser(data.user);
      setAuthToken(data.token);
      await AsyncStorage.setItem('@auth_token', data.token);
      await AsyncStorage.setItem('@auth_user', JSON.stringify(data.user));
    } catch (error) {
      throw error;
    }
  };

  const register = async (name, email, password) => {
    try {
      const data = await apiClient.post('/api/auth/register', { name, email, password });
      setToken(data.token);
      setUser(data.user);
      setAuthToken(data.token);
      await AsyncStorage.setItem('@auth_token', data.token);
      await AsyncStorage.setItem('@auth_user', JSON.stringify(data.user));
    } catch (error) {
      throw error;
    }
  };

  const logout = async () => {
    setToken(null);
    setUser(null);
    setAuthToken(null);
    await AsyncStorage.removeItem('@auth_token');
    await AsyncStorage.removeItem('@auth_user');
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};
