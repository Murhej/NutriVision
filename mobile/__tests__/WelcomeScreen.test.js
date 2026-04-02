import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import WelcomeScreen from '../src/screens/WelcomeScreen';

// Mock the theme hook
jest.mock('../src/theme/ThemeContext', () => ({
  useTheme: () => ({
    colors: {
      background: '#ffffff',
      text: '#111111',
      textSecondary: '#666666',
      surface: '#f8f8f8',
      border: '#dddddd',
      shadowColor: '#000000',
      shadowOpacity: 0.1,
      primarySoft: '#d1fae5',
    },
  }),
}));

describe('WelcomeScreen', () => {
  const navigation = {
    navigate: jest.fn(),
  };

  beforeEach(() => {
    navigation.navigate.mockClear();
  });

  it('renders welcome text and buttons correctly', () => {
    const { getByText } = render(<WelcomeScreen navigation={navigation} />);

    expect(getByText('Welcome to NutriVision')).toBeTruthy();
    expect(
      getByText('Your AI-powered nutrition companion for a healthier lifestyle')
    ).toBeTruthy();

    expect(getByText('Log In')).toBeTruthy();
    expect(getByText('Create Account')).toBeTruthy();
  });

  it('renders all feature labels', () => {
    const { getByText } = render(<WelcomeScreen navigation={navigation} />);

    expect(getByText('Track Goals')).toBeTruthy();
    expect(getByText('Log Meals')).toBeTruthy();
    expect(getByText('AI Scanner')).toBeTruthy();
    expect(getByText('Live Healthy')).toBeTruthy();
  });

  it('navigates to Login when Log In button is pressed', () => {
    const { getByText } = render(<WelcomeScreen navigation={navigation} />);

    fireEvent.press(getByText('Log In'));
    expect(navigation.navigate).toHaveBeenCalledWith('Login');
  });

  it('navigates to Register when Create Account button is pressed', () => {
    const { getByText } = render(<WelcomeScreen navigation={navigation} />);

    fireEvent.press(getByText('Create Account'));
    expect(navigation.navigate).toHaveBeenCalledWith('Register');
  });
});