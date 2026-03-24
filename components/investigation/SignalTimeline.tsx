"use client";

import { useMemo } from "react";
import type { ScoreResult, ScoreResultFeature, TransactionParameters } from "@/types";

const BG = "#161618";
const SURFACE = "#202022";
const BORDER = "#2E2E32";

function formatCurrencyUsd(amount: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(amount);
}

function formatFeatureValuePlain(value: unknown): string {
  if (value === null || value === undefined) {
    return "No value recorded";
  }
  if (typeof value === "boolean") {
    return value ? "Yes -- this condition applies" : "No -- this condition does not apply";
  }
  if (typeof value === "number") {
    if (Number.isInteger(value) && Math.abs(value) > 1000) {
      return `Numeric value ${value.toLocaleString("en-US")}`;
    }
    return `Numeric value ${value}`;
  }
  if (typeof value === "string") {
    return value.length > 0 ? value : "Empty text";
  }
  return String(value);
}

function verdictBorder(verdict: ScoreResult["verdict"]): string {
  switch (verdict) {
    case "fraud":
      return "rgba(239,68,68,0.3)";
    case "suspicious":
      return "rgba(245,158,11,0.3)";
    case "clean":
      return "rgba(34,197,94,0.3)";
    default:
      return BORDER;
  }
}

function verdictScoreColor(verdict: ScoreResult["verdict"]): string {
  switch (verdict) {
    case "fraud":
      return "#EF4444";
    case "suspicious":
      return "#F59E0B";
    case "clean":
      return "#22C55E";
    default:
      return "#F8FAFC";
  }
}

function severityDotColor(severity: ScoreResultFeature["severity"]): string {
  switch (severity) {
    case "critical":
    case "high":
      return "#EF4444";
    case "medium":
      return "#F59E0B";
    case "low":
      return "#475569";
    default:
      return "#475569";
  }
}

function severityBadgeStyles(severity: ScoreResultFeature["severity"]): {
  color: string;
  background: string;
} {
  switch (severity) {
    case "critical":
    case "high":
      return { color: "#EF4444", background: "rgba(239,68,68,0.12)" };
    case "medium":
      return { color: "#F59E0B", background: "rgba(245,158,11,0.12)" };
    case "low":
      return { color: "#475569", background: "rgba(71,85,105,0.15)" };
    default:
      return { color: "#475569", background: "rgba(71,85,105,0.15)" };
  }
}

type SignalTimelineProps = {
  scoreResult: ScoreResult | null;
  parameters: TransactionParameters;
  flaggedAt: string;
};

function parseFlaggedAt(flaggedAt: string): { hour: number; minute: number } {
  const match = flaggedAt.match(/^(\d{2}):(\d{2})/);
  if (!match) return { hour: 14, minute: 0 };
  return { hour: Number(match[1]), minute: Number(match[2]) };
}

function formatTime(hour: number, minute: number): string {
  const normalizedHour = ((hour % 24) + 24) % 24;
  const normalizedMinute = ((minute % 60) + 60) % 60;
  return `${String(normalizedHour).padStart(2, "0")}:${String(normalizedMinute).padStart(2, "0")}`;
}

export function SignalTimeline({ scoreResult, parameters, flaggedAt }: SignalTimelineProps) {
  const sortedFeatures = useMemo(() => {
    if (!scoreResult?.features.length) return [];
    return [...scoreResult.features]
      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
      .slice(0, 4);
  }, [scoreResult]);
  const chronologicalFeatures = useMemo(
    () => [...sortedFeatures].reverse(),
    [sortedFeatures]
  );

  const lastCriticalIndex = useMemo(() => {
    let last = -1;
    chronologicalFeatures.forEach((f, i) => {
      if (f.severity === "critical") last = i;
    });
    return last;
  }, [chronologicalFeatures]);

  const verdict = scoreResult?.verdict ?? "clean";
  const borderColor = scoreResult ? verdictBorder(verdict) : BORDER;
  const scoreColor = scoreResult ? verdictScoreColor(verdict) : "#64748B";
  const liveRiskScore = scoreResult ? Math.round(scoreResult.score * 100) : null;
  const flagged = parseFlaggedAt(flaggedAt);
  const chainStartHour = flagged.hour;
  const chainStartMinute = flagged.minute;

  return (
    <div className="no-scrollbar h-full overflow-y-auto p-8" style={{ backgroundColor: BG }}>
      <div className="grid grid-cols-[minmax(0,1fr)_170px] items-stretch gap-5">
        <div className="min-w-0 flex-1">
          <div
            className="h-full min-h-[154px] rounded-2xl border px-5 py-4"
            style={{
              borderColor: BORDER,
              background:
                "linear-gradient(145deg, rgba(32,32,34,0.95) 0%, rgba(22,22,24,0.95) 100%)",
            }}
          >
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#64748B]">
              Alert Narrative
            </p>
            <p className="mt-2 text-[21px] leading-[1.36] text-[#E2E8F0]">
              A{" "}
              <span className="rounded-md bg-[#F59E0B]/15 px-2 py-0.5 font-semibold text-[#FBBF24]">
                {parameters.accountAgeDays}-day-old account
              </span>{" "}
              pushed{" "}
              <span className="rounded-md bg-[#EF4444]/15 px-2 py-0.5 font-semibold text-[#F87171]">
                {formatCurrencyUsd(parameters.amount)}
              </span>{" "}
              to <span className="font-semibold text-[#F8FAFC]">{parameters.receiverCountry}</span> at{" "}
              <span className="rounded-md bg-[#1E293B]/70 px-2 py-0.5 font-semibold text-[#CBD5E1]">
                {String(parameters.hour).padStart(2, "0")}:00 UTC
              </span>
              .
            </p>
            <p className="mt-2 text-[15px] leading-relaxed text-[#94A3B8]">
              Transfer pace spiked to{" "}
              <span className="font-semibold text-[#F87171]">
                {parameters.transactionVelocity}x baseline
              </span>
              , signaling atypical outbound behavior.
            </p>
          </div>
        </div>
        <div
          className="rounded-2xl border px-4 py-3.5 text-center"
          style={{
            height: "100%",
            background:
              "linear-gradient(155deg, rgba(32,32,34,0.96) 0%, rgba(22,22,24,0.98) 100%)",
            borderColor: borderColor,
          }}
        >
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#64748B]">
            Live score
          </div>
          <div
            className="mt-1 font-mono text-[32px] leading-none tabular-nums"
            style={{ color: scoreColor }}
          >
            {liveRiskScore ?? <span className="text-[#64748B]">--</span>}
          </div>
          <div className="mt-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-[#94A3B8]">
            Fraud risk %
          </div>
        </div>
      </div>

      <div className="my-7 border-t" style={{ borderColor: BORDER }} aria-hidden />

      <div>
        <div className="mb-4 text-[14px] font-semibold uppercase tracking-[0.18em] text-[#94A3B8]">
          Signal chain
        </div>

        {scoreResult === null ? (
          <div>
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="mb-3 h-14 animate-pulse rounded-lg"
                style={{ backgroundColor: SURFACE }}
              />
            ))}
          </div>
        ) : (
          <div>
            <TimelineRow
              timeStr={formatTime(chainStartHour, chainStartMinute)}
              dotColor="#EF4444"
              dotGlow
              showConnectorLine
              title="Wire transfer initiated"
              subtitle={`${formatCurrencyUsd(parameters.amount)} -> ${parameters.receiverCountry} | ${parameters.messageType}`}
              badgeLabel="Flagged"
              badgeColor="#EF4444"
              badgeBg="rgba(239,68,68,0.12)"
              withBottomBorder
            />
            {chronologicalFeatures.map((feature, index) => {
              const pct = Math.round(Math.abs(feature.contribution) * 100);
              const sign = feature.contribution >= 0 ? "+" : "-";
              const badgeLabel = `${sign}${pct}% risk`;
              const { color: badgeColor, background: badgeBg } =
                severityBadgeStyles(feature.severity);
              const timeStr = formatTime(
                chainStartHour,
                chainStartMinute + (index + 1) * 3
              );
              const isLastCriticalDot =
                index === lastCriticalIndex && feature.severity === "critical";

              return (
                <TimelineRow
                  key={`${feature.name}-${index}`}
                  timeStr={timeStr}
                  dotColor={severityDotColor(feature.severity)}
                  dotGlow={isLastCriticalDot}
                  showConnectorLine={index < chronologicalFeatures.length - 1}
                  title={feature.humanLabel}
                  subtitle={formatFeatureValuePlain(feature.value)}
                  badgeLabel={badgeLabel}
                  badgeColor={badgeColor}
                  badgeBg={badgeBg}
                  withBottomBorder={index < chronologicalFeatures.length - 1}
                />
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

type TimelineRowProps = {
  timeStr: string;
  dotColor: string;
  dotGlow?: boolean;
  showConnectorLine: boolean;
  title: string;
  subtitle: string;
  badgeLabel: string;
  badgeColor: string;
  badgeBg: string;
  withBottomBorder: boolean;
};

function TimelineRow({
  timeStr,
  dotColor,
  dotGlow,
  showConnectorLine,
  title,
  subtitle,
  badgeLabel,
  badgeColor,
  badgeBg,
  withBottomBorder,
}: TimelineRowProps) {
  return (
    <div
      className="mb-5 flex gap-4 pb-5"
      style={
        withBottomBorder
          ? { borderBottomWidth: 1, borderBottomColor: BORDER }
          : undefined
      }
    >
      <div
        className="shrink-0 pt-0.5 font-mono text-[12px] text-[#475569]"
        style={{ width: 44 }}
      >
        {timeStr}
      </div>
      <div className="relative flex w-5 shrink-0 justify-center self-stretch">
        <div
          className="z-10 h-2.5 w-2.5 shrink-0 rounded-full"
          style={{
            backgroundColor: dotColor,
            boxShadow: dotGlow ? "0 0 0 3px rgba(239,68,68,0.2)" : undefined,
          }}
        />
        {showConnectorLine && (
          <div
            className="absolute left-1/2 top-2.5 bottom-0 w-px -translate-x-1/2"
            style={{ backgroundColor: BORDER }}
            aria-hidden
          />
        )}
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-[16px] font-medium text-[#F8FAFC]">{title}</div>
        <div className="mt-[4px] text-[13px] text-[#64748B]">{subtitle}</div>
        <span
          className="mt-[8px] inline-block rounded-[10px] px-2.5 py-0.5 text-[11px] font-semibold tracking-wide"
          style={{ color: badgeColor, backgroundColor: badgeBg }}
        >
          {badgeLabel}
        </span>
      </div>
    </div>
  );
}