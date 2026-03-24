import { create } from "zustand";

// --- Types (aligned with PRD Section 6 archetypes + Section 7 API) ---

export type TransactionParameters = {
  amount: number;
  hour: number;
  dayOfWeek: number;
  accountAgeDays: number;
  senderCountry: string;
  receiverCountry: string;
  transactionVelocity: number;
  ipCountryMatchesSender: boolean;
  messageHasTypos: boolean;
  messageType: string;
  isRoundNumber: boolean;
};

export type ScoreResultFeature = {
  name: string;
  humanLabel: string;
  value: unknown;
  normalizedValue: number;
  contribution: number;
  severity: "critical" | "high" | "medium" | "low";
};

export type ScoreResult = {
  score: number;
  verdict: "clean" | "suspicious" | "fraud";
  features: ScoreResultFeature[];
};

export type InvestigationStore = {
  parameters: TransactionParameters;
  setParameters: (params: Partial<TransactionParameters>) => void;

  archetypeId: string;
  setArchetype: (id: string) => void;

  scoreResult: ScoreResult | null;
  setScoreResult: (result: ScoreResult) => void;

  focusedNodeId: string | null;
  setFocusedNode: (id: string | null) => void;

  showFlaggedOnly: boolean;
  toggleFlaggedOnly: () => void;
};

const defaultParameters: TransactionParameters = {
  amount: 187500,
  hour: 14,
  dayOfWeek: 1,
  accountAgeDays: 3,
  senderCountry: "USA",
  receiverCountry: "Nigeria",
  transactionVelocity: 8,
  ipCountryMatchesSender: false,
  messageHasTypos: true,
  messageType: "pacs.008",
  isRoundNumber: false,
};

export const useInvestigationStore = create<InvestigationStore>((set) => ({
  parameters: defaultParameters,
  setParameters: (params) =>
    set((state) => ({
      parameters: { ...state.parameters, ...params },
    })),

  archetypeId: "bec",
  setArchetype: (id) => set({ archetypeId: id }),

  scoreResult: null,
  setScoreResult: (result) => set({ scoreResult: result }),

  focusedNodeId: null,
  setFocusedNode: (id) => set({ focusedNodeId: id }),

  showFlaggedOnly: false,
  toggleFlaggedOnly: () =>
    set((state) => ({ showFlaggedOnly: !state.showFlaggedOnly })),
}));
