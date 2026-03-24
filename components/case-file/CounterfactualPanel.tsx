"use client";

import { ChevronDown, ChevronRight } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";

const SURFACE_COLOR = "#202022";
const BORDER_COLOR = "#2E2E32";
const TEXT_PRIMARY = "#F8FAFC";
const TEXT_MUTED = "#475569";
const ACCENT_COLOR = "#0EA5E9";
const ACCENT_HOVER = "#0284C7";
const CLEAN_COLOR = "#22C55E";

export type CounterfactualChange = {
  from: unknown;
  to: unknown;
  impact: number;
};

export type CounterfactualPanelProps = {
  originalScore: number;
  achievedScore: number;
  changes: Record<string, CounterfactualChange>;
  currentValues: Record<string, unknown>;
  onTryScenario: (changes: Record<string, CounterfactualChange>) => void;
};

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "--";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    return value.toFixed(2);
  }
  if (typeof value === "string") return value;
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return String(value);
}

function formatFeatureLabel(key: string): string {
  return key
    .replace(/([A-Z])/g, " $1")
    .replace(/^./, (s) => s.toUpperCase())
    .replace(/_/g, " ")
    .trim();
}

export function CounterfactualPanel({
  originalScore,
  achievedScore,
  changes,
  currentValues,
  onTryScenario,
}: CounterfactualPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const entries = Object.entries(changes);
  const isEmpty = entries.length === 0;
  const actionableEntries = entries.filter(([key, change]) => {
    const current = currentValues[key];
    if (typeof change.to === "number") return Number(current) !== Number(change.to);
    if (typeof change.to === "boolean") return Boolean(current) !== change.to;
    return String(current) !== String(change.to);
  });
  const isApplied = !isEmpty && actionableEntries.length === 0;

  return (
    <div
      className="overflow-hidden rounded-lg border"
      style={{
        backgroundColor: SURFACE_COLOR,
        borderColor: BORDER_COLOR,
      }}
    >
      <button
        type="button"
        onClick={() => setIsOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-4 py-3 text-left transition-colors hover:bg-[#1A1A24]"
        style={{ color: TEXT_PRIMARY }}
        aria-expanded={isOpen}
      >
        {isOpen ? (
          <ChevronDown className="h-4 w-4 shrink-0" aria-hidden />
        ) : (
          <ChevronRight className="h-4 w-4 shrink-0" aria-hidden />
        )}
        <span className="text-sm font-semibold uppercase tracking-[0.18em] text-[#64748B]">
          COUNTERFACTUAL ANALYSIS
        </span>
      </button>

      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="border-t px-4 pb-4 pt-3" style={{ borderColor: BORDER_COLOR }}>
              <p
                className="text-xs font-medium uppercase tracking-wider"
                style={{ color: TEXT_MUTED }}
              >
                Minimum changes to drop below threshold
              </p>

              {isEmpty ? (
                <p
                  className="mt-3 text-sm"
                  style={{ color: TEXT_MUTED }}
                >
                  Transaction is already below risk threshold
                </p>
              ) : (
                <>
                  <div className="mt-3 space-y-2">
                    {entries.map(([key, change]) => (
                      <div
                        key={key}
                        className="flex flex-wrap items-baseline gap-x-2 gap-y-1 text-sm"
                      >
                        <span
                          className="font-medium"
                          style={{ color: TEXT_PRIMARY }}
                        >
                          {formatFeatureLabel(key)}:
                        </span>
                        <span
                          className="font-mono text-xs"
                          style={{ color: TEXT_MUTED }}
                        >
                          {formatValue(change.from)} -&gt; {formatValue(change.to)}
                        </span>
                        <span
                          className="rounded px-1.5 py-0.5 text-xs font-medium"
                          style={{
                            backgroundColor: `${CLEAN_COLOR}20`,
                            color: CLEAN_COLOR,
                          }}
                        >
                          -{Math.round(change.impact * 100)}% risk
                        </span>
                      </div>
                    ))}
                  </div>

                  <p
                    className="mt-2 text-xs"
                    style={{ color: TEXT_MUTED }}
                  >
                    Original score: {(originalScore * 100).toFixed(1)}% -&gt; achieved:{" "}
                    {(achievedScore * 100).toFixed(1)}%
                  </p>

                  <button
                    type="button"
                    onClick={() => {
                      if (actionableEntries.length === 0) return;
                      onTryScenario(Object.fromEntries(actionableEntries));
                    }}
                    disabled={isApplied}
                    className="mt-4 flex items-center gap-1.5 rounded-md px-3 py-2 text-sm font-semibold transition-colors"
                    style={{
                      backgroundColor: isApplied ? "#2E2E32" : ACCENT_COLOR,
                      color: isApplied ? "#64748B" : TEXT_PRIMARY,
                      cursor: isApplied ? "not-allowed" : "pointer",
                    }}
                    onMouseEnter={(e) => {
                      if (!isApplied) e.currentTarget.style.backgroundColor = ACCENT_HOVER;
                    }}
                    onMouseLeave={(e) => {
                      if (!isApplied) e.currentTarget.style.backgroundColor = ACCENT_COLOR;
                    }}
                  >
                    {isApplied ? "Scenario applied" : "Try this scenario"}
                    {!isApplied && <span aria-hidden>-&gt;</span>}
                  </button>
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}