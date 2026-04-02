import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import { Alert } from 'react-native';
import LoginScreen from '../src/screens/LoginScreen';
import { AuthContext } from '../src/context/AuthContext';

jest.mock('@react-native-async-storage/async-storage', () =>
  require('@react-native-async-storage/async-storage/jest/async-storage-mock')
);

jest.mock('../src/api/client', () => ({
  apiClient: {},
  setAuthToken: jest.fn(),
}));

jest.mock('../src/theme/ThemeContext', () => ({
  useTheme: () => ({
    colors: {
      background: '#ffffff',
      text: '#111111',
      textSecondary: '#666666',
      textTertiary: '#999999',
      border: '#dddddd',
      surface: '#f8f8f8',
    },
  }),
}));

jest.mock('../src/components', () => {
  const React = require('react');
  const { TouchableOpacity, Text } = require('react-native');

  return {
    Button: ({ title, onPress, disabled }) => (
      <TouchableOpacity onPress={onPress} disabled={disabled}>
        <Text>{title}</Text>
      </TouchableOpacity>
    ),
  };
});

describe('LoginScreen', () => {
  const navigation = {
    replace: jest.fn(),
  };

  const mockAlert = jest.fn();

  const renderScreen = (loginMock = jest.fn()) => {
    return render(
      <AuthContext.Provider value={{ login: loginMock }}>
        <LoginScreen navigation={navigation} />
      </AuthContext.Provider>
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    Alert.alert = mockAlert;
  });

  it('renders login screen text and inputs', () => {
    const { getByText, getByPlaceholderText } = renderScreen();

    expect(getByText('Welcome Back')).toBeTruthy();
    expect(getByText('Log in to continue')).toBeTruthy();
    expect(getByText('Email')).toBeTruthy();
    expect(getByText('Password')).toBeTruthy();
    expect(getByPlaceholderText('you@email.com')).toBeTruthy();
    expect(getByPlaceholderText('••••••••')).toBeTruthy();
    expect(getByText('Log In')).toBeTruthy();
    expect(getByText("Don't have an account? Sign Up")).toBeTruthy();
  });

  it('shows alert if fields are empty', () => {
    const loginMock = jest.fn();
    const { getByText } = renderScreen(loginMock);

    fireEvent.press(getByText('Log In'));

    expect(mockAlert).toHaveBeenCalledWith('Error', 'Please fill in all fields');
    expect(loginMock).not.toHaveBeenCalled();
  });

  it('calls login when email and password are entered', async () => {
    const loginMock = jest.fn().mockResolvedValueOnce(true);
    const { getByPlaceholderText, getByText } = renderScreen(loginMock);

    fireEvent.changeText(getByPlaceholderText('you@email.com'), 'test@example.com');
    fireEvent.changeText(getByPlaceholderText('••••••••'), 'mypassword');
    fireEvent.press(getByText('Log In'));

    await waitFor(() => {
      expect(loginMock).toHaveBeenCalledWith('test@example.com', 'mypassword');
    });
  });

  it('shows alert when login fails', async () => {
    const loginMock = jest.fn().mockRejectedValueOnce(new Error('Invalid credentials'));
    const { getByPlaceholderText, getByText } = renderScreen(loginMock);

    fireEvent.changeText(getByPlaceholderText('you@email.com'), 'test@example.com');
    fireEvent.changeText(getByPlaceholderText('••••••••'), 'wrongpassword');
    fireEvent.press(getByText('Log In'));

    await waitFor(() => {
      expect(mockAlert).toHaveBeenCalledWith('Login Failed', 'Invalid email or password');
    });
  });

  it('navigates to Register when sign up button is pressed', () => {
    const { getByText } = renderScreen();

    fireEvent.press(getByText("Don't have an account? Sign Up"));

    expect(navigation.replace).toHaveBeenCalledWith('Register');
  });
});