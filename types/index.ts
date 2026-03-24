/**
 * Shared TypeScript types (PRD Section 9 file structure, Section 7 API types).
 * Types only; re-exports from store for a single import surface.
 */

import type { TransactionParameters } from "../store/investigation";

// --- Graph (PRD Section 6, Feature 3 — Relationship Map) ---

export type NetworkNode = {
  id: string;
  label: string;
  type: "bank" | "corporate" | "individual" | "shell" | "unknown";
  riskLevel: "clean" | "suspicious" | "fraud";
  country: string;
  isSelected: boolean;
  isFocused: boolean;
};

export type NetworkEdge = {
  source: string;
  target: string;
  amount: number;
  timestamp: string;
  riskLevel: "clean" | "suspicious" | "fraud";
  messageType: "pacs.008" | "pacs.009" | "pacs.004";
};

// --- Archetypes (PRD Section 6, Feature 2 — GET /api/scenarios) ---

export type FraudArchetype = {
  id: string;
  name: string;
  description: string;
  icon: string;
  baseRiskLevel: "clean" | "suspicious" | "fraud" | "unknown";
  parameters: TransactionParameters;
};

// --- Re-export store types and hook ---

export type {
  TransactionParameters,
  ScoreResultFeature,
  ScoreResult,
  InvestigationStore,
} from "../store/investigation";

export { useInvestigationStore } from "../store/investigation";
