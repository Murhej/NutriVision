import React, { createContext, useContext, useState, useMemo } from 'react';
import { Colors } from './colors';

const ThemeContext = createContext();

export function ThemeProvider({ children }) {
  const [isDark, setIsDark] = useState(false);

  const theme = useMemo(() => ({
    isDark,
    colors: isDark ? Colors.dark : Colors.light,
    toggleTheme: () => setIsDark((prev) => !prev),
    setDarkMode: (nextIsDark) => setIsDark(Boolean(nextIsDark)),
  }), [isDark]);

  return (
    <ThemeContext.Provider value={theme}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
