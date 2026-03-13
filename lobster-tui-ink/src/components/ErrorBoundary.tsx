/**
 * React error boundary for graceful error handling.
 * Shows error message without crashing the app.
 */
import React, { Component } from "react";
import { Box, Text } from "ink";

interface Props {
  children: React.ReactNode;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <Box flexDirection="column" paddingX={1} marginY={1}>
          <Text bold color="red">
            Something went wrong
          </Text>
          <Text color="red">{this.state.error.message}</Text>
          <Text dimColor>The app is still running. Try sending a new message.</Text>
        </Box>
      );
    }

    return this.props.children;
  }
}
