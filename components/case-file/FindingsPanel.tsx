"use client";

import { motion } from "motion/react";
import type { ScoreResultFeature } from "@/types";

const CRITICAL_COLOR = "#EF4444";
const HIGH_COLOR = "#F59E0B";
const MEDIUM_COLOR = "#EAB308";
const LOW_COLOR = "#22C55E";
const SURFACE_COLOR = "#202022";
const BORDER_COLOR = "#2E2E32";
const TEXT_PRIMARY = "#F8FAFC";
const TEXT_MUTED = "#475569";

const SEVERITY_CONFIG: Record<
  ScoreResultFeature["severity"],
  { label: string; color: string }
> = {
  critical: { label: "CRITICAL", color: CRITICAL_COLOR },
  high: { label: "HIGH", color: HIGH_COLOR },
  medium: { label: "MEDIUM", color: MEDIUM_COLOR },
  low: { label: "LOW", color: LOW_COLOR },
};

export type FindingsPanelProps = {
  features: ScoreResultFeature[];
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

function sortByContributionDesc(features: ScoreResultFeature[]): ScoreResultFeature[] {
  return [...features].sort(
    (a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)
  );
}

export function FindingsPanel({ features }: FindingsPanelProps) {
  const sorted = sortByContributionDesc(features);

  if (sorted.length === 0) {
    return (
      <p
        className="py-6 text-center text-sm"
        style={{ color: TEXT_MUTED }}
      >
        No findings available
      </p>
    );
  }

  return (
    <motion.div
      className="flex flex-col gap-3"
      initial="hidden"
      animate="visible"
      variants={{
        visible: {
          transition: {
            delayChildren: 0.1,
            staggerChildren: 0.08,
          },
        },
        hidden: {},
      }}
    >
      {sorted.map((f) => {
        const config = SEVERITY_CONFIG[f.severity];
        const pct = Math.round(f.contribution * 100);
        const isPositive = f.contribution >= 0;
        const contributionLabel = isPositive ? `+${pct}%` : `${pct}%`;
        const barColor = isPositive ? CRITICAL_COLOR : LOW_COLOR;

        return (
          <motion.div
            key={f.name}
            variants={{
              hidden: { opacity: 0, y: 8 },
              visible: { opacity: 1, y: 0 },
            }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="rounded-lg border"
            style={{
              backgroundColor: SURFACE_COLOR,
              borderColor: BORDER_COLOR,
              borderRadius: 8,
              padding: 16,
            }}
          >
            <div className="flex items-start gap-2">
              <span
                className="shrink-0 rounded px-1.5 py-0.5 text-xs font-semibold uppercase tracking-[0.12em]"
                style={{
                  backgroundColor: `${config.color}20`,
                  color: config.color,
                }}
              >
                {config.label}
              </span>
              <div className="min-w-0 flex-1">
                <p
                  className="text-[15px] font-medium leading-snug"
                  style={{ color: TEXT_PRIMARY }}
                >
                  {f.humanLabel}
                </p>
                <p
                  className="mt-1 text-sm leading-snug"
                  style={{ color: TEXT_MUTED }}
                >
                  {formatValue(f.value)}
                </p>
              </div>
            </div>
            <div className="mt-3 flex items-center gap-2">
              <span
                className="text-xs font-medium tracking-wide"
                style={{ color: TEXT_MUTED }}
              >
                Contribution:
              </span>
              <span
                className="text-xs font-semibold tabular-nums"
                style={{ color: barColor }}
              >
                {contributionLabel} risk
              </span>
              <div
                className="ml-auto h-1.5 flex-1 max-w-24 rounded-full overflow-hidden"
                style={{ backgroundColor: BORDER_COLOR }}
              >
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${Math.min(100, Math.abs(f.contribution) * 100)}%`,
                    backgroundColor: barColor,
                  }}
                />
              </div>
            </div>
          </motion.div>
        );
      })}
    </motion.div>
  );
}