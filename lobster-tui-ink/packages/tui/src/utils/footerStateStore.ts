import { createStore } from "zustand/vanilla";
import type { StoreApi } from "zustand/vanilla";

export interface FooterState {
  completionVisible: boolean;
  inputBlocked: boolean;
  multiline: boolean;
}

export type FooterStateStore = StoreApi<FooterState>;

const INITIAL_FOOTER_STATE: FooterState = {
  completionVisible: false,
  inputBlocked: false,
  multiline: false,
};

export function createFooterStateStore(
  initialState: FooterState = INITIAL_FOOTER_STATE,
): FooterStateStore {
  return createStore<FooterState>(() => initialState);
}

export function setFooterState(
  store: FooterStateStore,
  patch: Partial<FooterState>,
) {
  store.setState((state) => ({ ...state, ...patch }));
}

export function resetFooterState(store: FooterStateStore) {
  store.setState(INITIAL_FOOTER_STATE);
}
