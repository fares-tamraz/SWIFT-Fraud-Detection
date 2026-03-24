"use client";

import type { ScoreResult, TransactionParameters } from "@/types";
import { deriveExplainFromTransaction } from "@/lib/api";
import { CounterfactualPanel } from "./CounterfactualPanel";
import { FindingsPanel } from "./FindingsPanel";
import { VerdictHeader } from "./VerdictHeader";

const BORDER_COLOR = "#2E2E32";
const SURFACE_COLOR = "#202022";

export type CounterfactualChanges = Record<
  string,
  { from: unknown; to: unknown; impact: number }
>;

export type CaseFilePanelProps = {
  scoreResult: ScoreResult | null;
  parameters: TransactionParameters;
  transactionId: string;
  flaggedAt: string;
  onTryScenario: (changes: CounterfactualChanges) => void;
};

function CaseFileSkeleton() {
  return (
    <div className="flex h-full flex-col gap-6" aria-label="Loading">
      <div
        className="h-32 animate-pulse rounded-lg"
        style={{ backgroundColor: "rgba(46, 46, 50, 0.6)" }}
      />
      <div
        className="h-12 w-full animate-pulse rounded"
        style={{ backgroundColor: "rgba(46, 46, 50, 0.6)" }}
      />
      <div className="flex flex-col gap-2">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="h-20 animate-pulse rounded-lg"
            style={{ backgroundColor: "rgba(46, 46, 50, 0.6)" }}
          />
        ))}
      </div>
    </div>
  );
}

function buildPlaceholderCounterfactual(
  scoreResult: ScoreResult,
  parameters: TransactionParameters
): CounterfactualChanges {
  const sorted = [...scoreResult.features].sort(
    (a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)
  );
  const top4 = sorted.slice(0, 4);
  const changes: CounterfactualChanges = {};
  for (const f of top4) {
    switch (f.name) {
      case "transaction_velocity":
        changes.transactionVelocity = {
          from: f.value,
          to: Math.max(1, Number(f.value) - 3),
          impact: Math.abs(f.contribution),
        };
        break;
      case "account_age_days":
        changes.accountAgeDays = {
          from: f.value,
          to: Math.max(30, Number(f.value) + 90),
          impact: Math.abs(f.contribution),
        };
        break;
      case "hour_of_day":
        changes.hour = {
          from: f.value,
          to: 10,
          impact: Math.abs(f.contribution),
        };
        break;
      case "transaction_amount":
      case "amount":
        changes.amount = {
          from: f.value,
          to: Math.max(1000, Math.round(Number(f.value) * 0.62)),
          impact: Math.abs(f.contribution),
        };
        break;
      case "receiver_country":
        changes.receiverCountry = {
          from: f.value,
          to: "UK",
          impact: Math.abs(f.contribution),
        };
        break;
      default:
        break;
    }
  }
  if (!changes.ipCountryMatchesSender) {
    changes.ipCountryMatchesSender = {
      from: parameters.ipCountryMatchesSender,
      to: true,
      impact: 0.08,
    };
  }
  if (!changes.messageHasTypos) {
    changes.messageHasTypos = {
      from: parameters.messageHasTypos,
      to: false,
      impact: 0.06,
    };
  }
  return changes;
}

function applyCounterfactualChanges(
  parameters: TransactionParameters,
  changes: CounterfactualChanges
): TransactionParameters {
  const next: TransactionParameters = { ...parameters };
  for (const [key, change] of Object.entries(changes)) {
    if (key === "amount" && typeof change.to === "number") next.amount = change.to;
    if (key === "hour" && typeof change.to === "number") next.hour = change.to;
    if (key === "accountAgeDays" && typeof change.to === "number") {
      next.accountAgeDays = change.to;
    }
    if (key === "transactionVelocity" && typeof change.to === "number") {
      next.transactionVelocity = change.to;
    }
    if (key === "receiverCountry" && typeof change.to === "string") {
      next.receiverCountry = change.to;
    }
    if (key === "ipCountryMatchesSender" && typeof change.to === "boolean") {
      next.ipCountryMatchesSender = change.to;
    }
    if (key === "messageHasTypos" && typeof change.to === "boolean") {
      next.messageHasTypos = change.to;
    }
  }
  return next;
}

export function CaseFilePanel({
  scoreResult,
  parameters,
  transactionId,
  flaggedAt,
  onTryScenario,
}: CaseFilePanelProps) {
  if (scoreResult === null) {
    return (
      <div
        className="no-scrollbar h-full overflow-y-auto"
        style={{
          backgroundColor: SURFACE_COLOR,
          borderRight: `1px solid ${BORDER_COLOR}`,
          padding: 24,
        }}
      >
        <CaseFileSkeleton />
      </div>
    );
  }

  const placeholderChanges = buildPlaceholderCounterfactual(scoreResult, parameters);
  const projectedParams = applyCounterfactualChanges(parameters, placeholderChanges);
  const achievedScore = deriveExplainFromTransaction(projectedParams).score;

  return (
    <div
      className="no-scrollbar h-full overflow-y-auto"
      style={{
        backgroundColor: SURFACE_COLOR,
        borderRight: `1px solid ${BORDER_COLOR}`,
        padding: 24,
      }}
    >
      <section>
        <VerdictHeader
          score={scoreResult.score}
          verdict={scoreResult.verdict}
          transactionId={transactionId}
          timestamp={flaggedAt}
        />
      </section>

      <section
        className="mt-5 rounded-xl border p-4"
        style={{ borderColor: BORDER_COLOR, backgroundColor: "#1D1D20" }}
      >
        <h3
          className="mb-3 text-sm font-semibold uppercase tracking-[0.18em]"
          style={{ color: "#64748B" }}
        >
          Key findings
        </h3>
        <FindingsPanel features={scoreResult.features} />
      </section>

      <section className="mt-5">
        <CounterfactualPanel
          originalScore={scoreResult.score}
          achievedScore={achievedScore}
          changes={placeholderChanges}
          currentValues={parameters as unknown as Record<string, unknown>}
          onTryScenario={onTryScenario}
        />
      </section>
    </div>
  );
}