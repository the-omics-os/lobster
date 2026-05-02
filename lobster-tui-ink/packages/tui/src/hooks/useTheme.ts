import React, { createContext, useContext } from "react";
import { resolveTheme, type Theme } from "../theme.js";

const ThemeContext = createContext<Theme>(resolveTheme());

interface ThemeProviderProps {
  children: React.ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  return React.createElement(
    ThemeContext.Provider,
    { value: resolveTheme() },
    children,
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
