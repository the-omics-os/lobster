import { createStore } from "zustand/vanilla";
import type { StoreApi } from "zustand/vanilla";
import {
  applyStatePatch,
  createInitialState,
  type AppState,
  type StateKey,
} from "./stateHandlers.js";

export type AppStateStore = StoreApi<AppState>;

export function createAppStateStore(initialState: AppState = createInitialState()): AppStateStore {
  return createStore<AppState>(() => initialState);
}

export function applyAppStatePatch(
  store: AppStateStore,
  key: StateKey,
  data: unknown,
) {
  store.setState((state) => applyStatePatch(state, key, data));
}

export function resetAppStateStore(store: AppStateStore) {
  store.setState(createInitialState());
}

export function clearRunActivity(store: AppStateStore) {
  store.setState((state) => ({
    ...state,
    activeAgent: null,
    agentStatus: null,
    activityEvents: [],
    progress: {},
  }));
}
