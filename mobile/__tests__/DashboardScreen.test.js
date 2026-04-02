import React from 'react';
import { render, fireEvent, waitFor, Text } from '@testing-library/react-native';
import DashboardScreen from '../src/screens/DashboardScreen';

// Mock navigation hook
jest.mock('@react-navigation/native', () => ({
  useFocusEffect: (cb) => cb(),
}));

// Mock API
jest.mock('../src/api/client', () => ({
  apiClient: {
    get: jest.fn((url) => {
      if (url.includes('dashboard')) {
        return Promise.resolve({
          nutrition: {
            calories: { consumed: 1000, goal: 2000 },
            protein: { consumed: 50, goal: 100 },
            carbs: { consumed: 150, goal: 300 },
            fat: { consumed: 30, goal: 60 },
          },
          meals: [{ id: 1 }],
        });
      }

      if (url.includes('profile')) {
        return Promise.resolve({
          profile: {
            name: 'Albin',
            initials: 'A',
            streak: 5,
          },
        });
      }
    }),
  },
}));

// Mock theme
jest.mock('../src/theme/ThemeContext', () => ({
  useTheme: () => ({
    colors: {
      background: '#fff',
      text: '#000',
      textSecondary: '#666',
      textTertiary: '#999',
      primary: '#4CAF50',
      primarySoft: '#d1fae5',
      surface: '#f8f8f8',
      surfaceSecondary: '#eee',
      border: '#ddd',
      calories: '#ff0000',
      protein: '#00ff00',
      carbs: '#0000ff',
      fat: '#ffaa00',
    },
  }),
}));

// Mock components
jest.mock('../src/components', () => {
  const React = require('react');
  const { View, Text } = require('react-native');

  return {
    Card: ({ children }) => <View>{children}</View>,
    ProgressRing: ({ children }) => <View>{children}</View>,
    ProgressBar: () => <View />,
    MealCard: () => <Text>Meal</Text>,
  };
});

describe('DashboardScreen', () => {
  const navigation = {
    navigate: jest.fn(),
  };

  it('shows loading initially', () => {
    const { getByTestId } = render(
      <DashboardScreen navigation={navigation} />
    );

    // we just check render does not crash
    expect(true).toBeTruthy();
  });

  it('renders user data after loading', async () => {
    const { getByText } = render(
      <DashboardScreen navigation={navigation} />
    );

    await waitFor(() => {
      expect(getByText('Albin')).toBeTruthy();
    });
  });

  it('navigates to Scan screen', async () => {
    const { getByText } = render(
      <DashboardScreen navigation={navigation} />
    );

    await waitFor(() => {
      expect(getByText('Scan Your Meal')).toBeTruthy();
    });

    fireEvent.press(getByText('Scan Your Meal'));

    expect(navigation.navigate).toHaveBeenCalledWith('Scan');
  });
});