"use client";

import {
  AlertTriangle,
  CheckCircle2,
  ShieldAlert,
} from "lucide-react";
import { motion } from "motion/react";

const CLEAN_COLOR = "#22C55E";
const SUSPICIOUS_COLOR = "#F59E0B";
const FRAUD_COLOR = "#EF4444";
const SURFACE_COLOR = "#202022";
const BORDER_COLOR = "#2E2E32";
const TEXT_PRIMARY = "#F8FAFC";
const TEXT_MUTED = "#475569";

export type Verdict = "clean" | "suspicious" | "fraud";

export type VerdictHeaderProps = {
  score: number;
  verdict: Verdict;
  transactionId: string;
  timestamp: string;
};

const VERDICT_CONFIG: Record<
  Verdict,
  { label: string; color: string; Icon: typeof CheckCircle2 }
> = {
  clean: {
    label: "CLEAN",
    color: CLEAN_COLOR,
    Icon: CheckCircle2,
  },
  suspicious: {
    label: "SUSPICIOUS",
    color: SUSPICIOUS_COLOR,
    Icon: AlertTriangle,
  },
  fraud: {
    label: "FRAUD DETECTED",
    color: FRAUD_COLOR,
    Icon: ShieldAlert,
  },
};

export function VerdictHeader({
  verdict,
  transactionId,
  timestamp,
}: VerdictHeaderProps) {
  const config = VERDICT_CONFIG[verdict];
  const { Icon } = config;

  return (
    <div
      className="rounded-lg border p-6"
      style={{
        backgroundColor: SURFACE_COLOR,
        borderColor: BORDER_COLOR,
      }}
    >
      <div className="flex items-center justify-center">
        <div className="flex items-center gap-3 sm:gap-4">
          <div className="relative shrink-0">
            {verdict === "fraud" && (
              <motion.span
                className="absolute inset-[-3px] rounded-xl"
                style={{
                  boxShadow: `0 0 0 2px ${FRAUD_COLOR}`,
                }}
                animate={{ opacity: [0.4, 1, 0.4] }}
                transition={{
                  duration: 1.5,
                  repeat: Number.POSITIVE_INFINITY,
                  ease: "easeInOut",
                }}
                aria-hidden
              />
            )}
            <div
              className="flex items-center gap-2 rounded-lg px-4 py-2.5"
              style={{
                backgroundColor: config.color,
                border:
                  verdict === "fraud"
                    ? "none"
                    : `1px solid ${BORDER_COLOR}`,
              }}
            >
              <Icon
                className="h-6 w-6 shrink-0"
                style={{ color: TEXT_PRIMARY }}
                strokeWidth={2}
              />
              <span
                className="text-base font-bold tracking-[0.08em] sm:text-lg"
                style={{ color: TEXT_PRIMARY }}
              >
                {config.label}
              </span>
            </div>
          </div>

        </div>

      </div>

      <div className="mt-4 flex flex-col gap-0.5 border-t pt-4 font-mono text-sm" style={{ borderColor: BORDER_COLOR }}>
        <span style={{ color: TEXT_MUTED }}>
          Transaction {transactionId}
        </span>
        <span style={{ color: TEXT_MUTED }}>
          Flagged at {timestamp}
        </span>
      </div>
    </div>
  );
}