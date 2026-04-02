import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import { Alert } from 'react-native';
import RegisterScreen from '../src/screens/RegisterScreen';
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

describe('RegisterScreen', () => {
  const navigation = {
    replace: jest.fn(),
  };

  const mockAlert = jest.fn();

  const renderScreen = (registerMock = jest.fn()) => {
    return render(
      <AuthContext.Provider value={{ register: registerMock }}>
        <RegisterScreen navigation={navigation} />
      </AuthContext.Provider>
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    Alert.alert = mockAlert;
  });

  it('renders register screen text and inputs', () => {
    const { getByText, getByPlaceholderText } = renderScreen();

    expect(getByText('Create Account')).toBeTruthy();
    expect(
      getByText('Start your healthy journey tracking limits today')
    ).toBeTruthy();

    expect(getByText('Full Name')).toBeTruthy();
    expect(getByText('Email Address')).toBeTruthy();
    expect(getByText('Password')).toBeTruthy();

    expect(getByPlaceholderText('Jane Doe')).toBeTruthy();
    expect(getByPlaceholderText('you@email.com')).toBeTruthy();
    expect(getByPlaceholderText('••••••••')).toBeTruthy();

    expect(getByText('Sign Up')).toBeTruthy();
    expect(getByText('Already have an account? Log In')).toBeTruthy();
  });

  it('shows alert if fields are empty', () => {
    const registerMock = jest.fn();
    const { getByText } = renderScreen(registerMock);

    fireEvent.press(getByText('Sign Up'));

    expect(mockAlert).toHaveBeenCalledWith('Error', 'Please fill in all fields');
    expect(registerMock).not.toHaveBeenCalled();
  });

  it('shows alert if password is less than 6 characters', () => {
    const registerMock = jest.fn();
    const { getByPlaceholderText, getByText } = renderScreen(registerMock);

    fireEvent.changeText(getByPlaceholderText('Jane Doe'), 'Albin');
    fireEvent.changeText(getByPlaceholderText('you@email.com'), 'test@example.com');
    fireEvent.changeText(getByPlaceholderText('••••••••'), '123');

    fireEvent.press(getByText('Sign Up'));

    expect(mockAlert).toHaveBeenCalledWith(
      'Error',
      'Password must be at least 6 characters'
    );
    expect(registerMock).not.toHaveBeenCalled();
  });

  it('calls register when valid data is entered', async () => {
    const registerMock = jest.fn().mockResolvedValueOnce(true);
    const { getByPlaceholderText, getByText } = renderScreen(registerMock);

    fireEvent.changeText(getByPlaceholderText('Jane Doe'), 'Albin Chacko');
    fireEvent.changeText(getByPlaceholderText('you@email.com'), 'albin@example.com');
    fireEvent.changeText(getByPlaceholderText('••••••••'), 'mypassword');

    fireEvent.press(getByText('Sign Up'));

    await waitFor(() => {
      expect(registerMock).toHaveBeenCalledWith(
        'Albin Chacko',
        'albin@example.com',
        'mypassword'
      );
    });
  });

  it('shows alert when registration fails', async () => {
    const registerMock = jest
      .fn()
      .mockRejectedValueOnce(new Error('Registration failed'));

    const { getByPlaceholderText, getByText } = renderScreen(registerMock);

    fireEvent.changeText(getByPlaceholderText('Jane Doe'), 'Albin Chacko');
    fireEvent.changeText(getByPlaceholderText('you@email.com'), 'albin@example.com');
    fireEvent.changeText(getByPlaceholderText('••••••••'), 'mypassword');

    fireEvent.press(getByText('Sign Up'));

    await waitFor(() => {
      expect(mockAlert).toHaveBeenCalledWith(
        'Registration Failed',
        'Email might already be in use'
      );
    });
  });

  it('navigates to Login when log in button is pressed', () => {
    const { getByText } = renderScreen();

    fireEvent.press(getByText('Already have an account? Log In'));

    expect(navigation.replace).toHaveBeenCalledWith('Login');
  });
});