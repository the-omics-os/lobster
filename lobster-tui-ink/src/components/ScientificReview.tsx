import React from "react";
import { Box, Text } from "ink";

interface ScientificReviewData {
  verdict?: string;
  confidence?: number;
  issues?: unknown[];
}

interface ScientificReviewProps {
  review: ScientificReviewData;
}

/**
 * Inline scientific review verdict below an assistant message.
 * Rendered from the durable message envelope's scientific_review field.
 */
export function ScientificReview({ review }: ScientificReviewProps) {
  const { verdict, confidence, issues } = review;

  if (!verdict) return null;

  const issueCount = issues?.length ?? 0;

  return (
    <Box gap={1} marginTop={1}>
      <Text dimColor>Review:</Text>
      <Text dimColor bold>
        {verdict}
      </Text>
      {confidence !== undefined && (
        <Text dimColor>({confidence}% confidence</Text>
      )}
      {issueCount > 0 && (
        <Text dimColor>
          {confidence !== undefined ? ", " : "("}
          {issueCount} issue{issueCount !== 1 ? "s" : ""}
          {")"}
        </Text>
      )}
      {issueCount === 0 && confidence !== undefined && (
        <Text dimColor>{")"}</Text>
      )}
    </Box>
  );
}
