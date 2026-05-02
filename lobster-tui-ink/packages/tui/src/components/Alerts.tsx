import React from "react";
import { useStore } from "zustand";
import type { AppStateStore } from "../utils/appStateStore.js";
import { AlertBlock } from "./AlertBlock.js";

interface AlertsProps {
  appStateStore: AppStateStore;
}

export function Alerts({ appStateStore }: AlertsProps) {
  const alerts = useStore(appStateStore, (state) => state.alerts);

  return (
    <>
      {alerts.map((alert, index) => (
        <AlertBlock
          key={`${alert.level}:${index}:${alert.message}`}
          level={alert.level}
          title={alert.title}
          message={alert.message}
        />
      ))}
    </>
  );
}
