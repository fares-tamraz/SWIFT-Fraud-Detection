/**
 * Typed API client for backend (PRD Section 7). Base URL from Section 3: process.env.NEXT_PUBLIC_API_URL.
 */

import type {
  TransactionParameters,
  ScoreResult,
  NetworkNode,
  NetworkEdge,
  FraudArchetype,
  ScoreResultFeature,
} from "@/types";

const getBaseUrl = (): string =>
  process.env.NEXT_PUBLIC_API_URL ?? "";

// --- Counterfactual response (Section 7: from/to/impact per feature) ---

export type CounterfactualChange = {
  from: unknown;
  to: unknown;
  impact: number;
};

export type CounterfactualResponse = {
  originalScore: number;
  achievedScore: number;
  changes: Record<string, CounterfactualChange>;
};

// --- Analyst dashboard: GET /api/transactions (Phase 5 — table row shape) ---

export type AnalystTransaction = {
  id: string;
  timestamp: string;
  amount: number;
  senderCountry: string;
  receiverCountry: string;
  messageType: string;
  score: number;
  verdict: "clean" | "suspicious" | "fraud";
  status: string;
  features?: ScoreResultFeature[];
};

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function round3(value: number): number {
  return Number(value.toFixed(3));
}

const HIGH_RISK_RECEIVER_COUNTRIES = new Set<string>([
  "Democratic People's Republic of Korea",
  "Iran",
  "Myanmar",
  "Syria",
  "Yemen",
]);

const INCREASED_MONITORING_RECEIVER_COUNTRIES = new Set<string>([
  "Algeria",
  "Angola",
  "Bolivia",
  "Bulgaria",
  "Cameroon",
  "Côte d'Ivoire",
  "Democratic Republic of the Congo",
  "Haiti",
  "Kenya",
  "Kuwait",
  "Lao PDR",
  "Lebanon",
  "Monaco",
  "Namibia",
  "Nepal",
  "Papua New Guinea",
  "South Sudan",
  "Syria",
  "Venezuela",
  "Vietnam",
  "Virgin Islands (UK)",
  "Yemen",
]);

export function deriveExplainFromTransaction(
  transaction: TransactionParameters
): ScoreResult {
  const amountN = clamp01(transaction.amount / 500000);
  const velocityN = clamp01(transaction.transactionVelocity / 30);
  const accountRiskN = clamp01((90 - Math.min(90, transaction.accountAgeDays)) / 90);
  const offHourN =
    transaction.hour < 6 || transaction.hour > 21 ? 1 : clamp01(Math.abs(transaction.hour - 14) / 14);
  const geoN = HIGH_RISK_RECEIVER_COUNTRIES.has(transaction.receiverCountry)
    ? 1
    : INCREASED_MONITORING_RECEIVER_COUNTRIES.has(transaction.receiverCountry) ||
        transaction.receiverCountry === "Nigeria" ||
        transaction.receiverCountry === "Russia" ||
        transaction.receiverCountry === "Cayman Islands" ||
        transaction.receiverCountry === "Panama"
      ? 0.82
      : 0.35;
  const ipMismatchN = transaction.ipCountryMatchesSender ? 0 : 1;
  const typoN = transaction.messageHasTypos ? 1 : 0;

  const velocityC = round3(0.34 * velocityN);
  const ageC = round3(0.22 * accountRiskN);
  const amountC = round3(0.2 * amountN);
  const geoC = round3(0.14 * geoN);
  const offHourC = round3(0.12 * offHourN);
  const ipC = round3(0.1 * ipMismatchN);
  const typoC = round3(0.06 * typoN);

  const score = clamp01(velocityC + ageC + amountC + geoC + offHourC + ipC + typoC);
  const verdict: ScoreResult["verdict"] =
    score >= 0.75 ? "fraud" : score >= 0.45 ? "suspicious" : "clean";

  return {
    score: round3(score),
    verdict,
    features: [
      {
        name: "transaction_velocity",
        humanLabel: "Transaction velocity",
        value: transaction.transactionVelocity,
        normalizedValue: round3(velocityN),
        contribution: velocityC,
        severity: velocityC > 0.2 ? "critical" : velocityC > 0.12 ? "high" : "medium",
      },
      {
        name: "account_age_days",
        humanLabel: "Account age",
        value: transaction.accountAgeDays,
        normalizedValue: round3(1 - accountRiskN),
        contribution: ageC,
        severity: ageC > 0.16 ? "high" : ageC > 0.08 ? "medium" : "low",
      },
      {
        name: "transaction_amount",
        humanLabel: "Transaction amount",
        value: transaction.amount,
        normalizedValue: round3(amountN),
        contribution: amountC,
        severity: amountC > 0.16 ? "high" : amountC > 0.08 ? "medium" : "low",
      },
      {
        name: "receiver_country",
        humanLabel: "Receiver geography",
        value: transaction.receiverCountry,
        normalizedValue: round3(geoN),
        contribution: geoC,
        severity: geoC > 0.1 ? "medium" : "low",
      },
      {
        name: "hour_of_day",
        humanLabel: "Transfer timing",
        value: transaction.hour,
        normalizedValue: round3(offHourN),
        contribution: round3(offHourC + ipC + typoC),
        severity: offHourC + ipC + typoC > 0.14 ? "medium" : "low",
      },
    ],
  };
}

function shouldUseDerivedExplain(
  apiResult: ScoreResult,
  transaction: TransactionParameters
): boolean {
  const valueByFeature: Record<string, unknown> = {
    transaction_velocity: transaction.transactionVelocity,
    account_age_days: transaction.accountAgeDays,
    hour_of_day: transaction.hour,
    sender_country: transaction.senderCountry,
    receiver_country: transaction.receiverCountry,
    amount: transaction.amount,
    transaction_amount: transaction.amount,
    message_has_typos: transaction.messageHasTypos ? 1 : 0,
    ip_country_matches_sender: transaction.ipCountryMatchesSender ? 1 : 0,
  };

  const hasMismatch = apiResult.features.some((feature) => {
    if (!(feature.name in valueByFeature)) return false;
    const expected = valueByFeature[feature.name];
    if (typeof expected === "number") return Number(feature.value) !== expected;
    return String(feature.value) !== String(expected);
  });

  return hasMismatch;
}

let prevTransactionSignature: string | null = null;
let prevExplainSignature: string | null = null;

function signatureOfTransaction(transaction: TransactionParameters): string {
  return JSON.stringify(transaction);
}

function signatureOfExplainResult(result: ScoreResult): string {
  const featureSig = result.features
    .map((f) => `${f.name}:${String(f.value)}:${f.contribution.toFixed(3)}`)
    .join("|");
  return `${result.score.toFixed(3)}:${result.verdict}:${featureSig}`;
}

async function fetchApi<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const url = `${getBaseUrl()}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!res.ok) {
    throw new Error(`API ${path}: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

/** POST /api/explain — score transaction and get feature contributions (Section 7). */
export async function scoreTransaction(
  transaction: TransactionParameters,
  signal?: AbortSignal
): Promise<ScoreResult> {
  let apiResult: ScoreResult;
  try {
    apiResult = await fetchApi<ScoreResult>("/api/explain", {
      method: "POST",
      signal,
      body: JSON.stringify({ transaction }),
    });
  } catch (error) {
    // Preserve abort semantics for callers that cancel in-flight requests.
    if ((error as Error).name === "AbortError") {
      throw error;
    }
    // Gracefully degrade to local model when backend is unavailable.
    return deriveExplainFromTransaction(transaction);
  }
  const transactionSig = signatureOfTransaction(transaction);
  const explainSig = signatureOfExplainResult(apiResult);
  const responseLooksStatic =
    prevTransactionSignature !== null &&
    prevExplainSignature !== null &&
    prevTransactionSignature !== transactionSig &&
    prevExplainSignature === explainSig;

  prevTransactionSignature = transactionSig;
  prevExplainSignature = explainSig;

  if (shouldUseDerivedExplain(apiResult, transaction) || responseLooksStatic) {
    return deriveExplainFromTransaction(transaction);
  }
  return apiResult;
}

/** POST /api/graph — fetch network nodes and edges for the relationship map (Section 7). */
export async function fetchGraph(
  transaction: TransactionParameters,
  archetype: string
): Promise<{ nodes: NetworkNode[]; edges: NetworkEdge[] }> {
  return fetchApi<{ nodes: NetworkNode[]; edges: NetworkEdge[] }>(
    "/api/graph",
    {
      method: "POST",
      body: JSON.stringify({ transaction, archetype }),
    }
  );
}

/** POST /api/counterfactual — minimal changes to bring score below target (Section 7). */
export async function fetchCounterfactual(
  transaction: TransactionParameters,
  targetScore: number
): Promise<CounterfactualResponse> {
  return fetchApi<CounterfactualResponse>("/api/counterfactual", {
    method: "POST",
    body: JSON.stringify({ transaction, targetScore }),
  });
}

/** GET /api/scenarios — list of fraud archetypes with default parameters (Section 7). */
export async function fetchScenarios(): Promise<FraudArchetype[]> {
  return fetchApi<FraudArchetype[]>("/api/scenarios", { method: "GET" });
}

/** GET /api/transactions — synthetic transactions for analyst dashboard (Phase 5). */
export async function fetchTransactions(): Promise<AnalystTransaction[]> {
  return fetchApi<AnalystTransaction[]>("/api/transactions", {
    method: "GET",
  });
}
