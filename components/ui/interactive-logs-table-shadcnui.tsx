"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Check, ChevronDown, Filter, Search, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { ScoreResult, TransactionParameters } from "@/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export type MonitoredTransaction = {
  id: string;
  timestamp: string;
  transactionId: string;
  flaggedAt: string;
  parameters: TransactionParameters;
  scoreResult: ScoreResult | null;
};

type Filters = {
  verdict: string[];
  country: string[];
};

function verdictBadge(verdict: string) {
  if (verdict === "fraud") return "bg-red-500/15 text-red-400";
  if (verdict === "suspicious") return "bg-amber-500/15 text-amber-400";
  return "bg-green-500/15 text-green-400";
}

function shortFlagged(flaggedAt: string): string {
  return flaggedAt.replace(" UTC", "");
}

function Row({
  tx,
  expanded,
  selected,
  onToggle,
  onSelect,
  onDelete,
  onHover,
}: {
  tx: MonitoredTransaction;
  expanded: boolean;
  selected: boolean;
  onToggle: () => void;
  onSelect: () => void;
  onDelete: () => void;
  onHover: (id: string | null) => void;
}) {
  const score = tx.scoreResult ? Math.round(tx.scoreResult.score * 100) : "--";
  const verdict = tx.scoreResult?.verdict ?? "pending";
  return (
    <>
      <motion.div
        onMouseEnter={() => onHover(tx.id)}
        onMouseLeave={() => onHover(null)}
        className={`w-full border-b border-[#2E2E32] p-3 transition-colors ${selected ? "bg-[#1F2937]/50" : "hover:bg-[#1A1A24]"}`}
      >
        <div className="flex items-center gap-2">
          <button type="button" onClick={onToggle}>
            <motion.div animate={{ rotate: expanded ? 180 : 0 }}>
              <ChevronDown className="h-4 w-4 text-[#94A3B8]" />
            </motion.div>
          </button>
          <button type="button" onClick={onSelect} className="flex flex-1 items-center gap-2 text-left">
            <Badge variant="secondary" className={`w-[84px] justify-center capitalize ${verdictBadge(verdict)}`}>
              {verdict}
            </Badge>
            <span className="w-[72px] font-mono text-xs text-[#94A3B8]">{shortFlagged(tx.flaggedAt)}</span>
            <span className="min-w-0 flex-1 truncate text-sm text-[#F8FAFC]">
              {tx.parameters.senderCountry} -&gt; {tx.parameters.receiverCountry}
            </span>
            <span className="w-[34px] text-right font-mono text-sm text-[#CBD5E1]">{score}</span>
          </button>
          <button
            type="button"
            aria-label="Delete transaction"
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
            className="rounded border border-[#2E2E32] p-1 text-[#94A3B8] transition-colors hover:border-red-400/60 hover:bg-red-500/10 hover:text-red-300"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>
      </motion.div>
      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden border-b border-[#2E2E32] bg-[#17171A]"
          >
            <div className="grid grid-cols-2 gap-2 p-3 text-xs text-[#94A3B8]">
              <div>Amount: <span className="font-mono text-[#E2E8F0]">${tx.parameters.amount.toLocaleString()}</span></div>
              <div>Velocity: <span className="font-mono text-[#E2E8F0]">{tx.parameters.transactionVelocity}x</span></div>
              <div>Hour: <span className="font-mono text-[#E2E8F0]">{tx.parameters.hour}:00</span></div>
              <div>Type: <span className="font-mono text-[#E2E8F0]">{tx.parameters.messageType}</span></div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

export function InteractiveLogsTable({
  transactions,
  activeId,
  onSelect,
  onDelete,
  onVisibleChange,
  onHoverTransaction,
}: {
  transactions: MonitoredTransaction[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onVisibleChange?: (ids: string[]) => void;
  onHoverTransaction?: (id: string | null) => void;
}) {
  const [search, setSearch] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState<Filters>({ verdict: [], country: [] });

  const filtered = useMemo(() => {
    return transactions.filter((tx) => {
      const verdict = tx.scoreResult?.verdict ?? "pending";
      const country = tx.parameters.receiverCountry;
      const q = search.toLowerCase();
      const matchSearch =
        tx.parameters.senderCountry.toLowerCase().includes(q) ||
        tx.parameters.receiverCountry.toLowerCase().includes(q) ||
        tx.id.toLowerCase().includes(q);
      const matchVerdict =
        filters.verdict.length === 0 || filters.verdict.includes(verdict);
      const matchCountry =
        filters.country.length === 0 || filters.country.includes(country);
      return matchSearch && matchVerdict && matchCountry;
    });
  }, [transactions, search, filters]);

  useEffect(() => {
    onVisibleChange?.(filtered.map((tx) => tx.id));
  }, [filtered, onVisibleChange]);

  const verdicts = Array.from(
    new Set(transactions.map((t) => t.scoreResult?.verdict ?? "pending"))
  );
  const countries = Array.from(
    new Set(transactions.map((t) => t.parameters.receiverCountry))
  );

  const toggleFilter = (key: keyof Filters, value: string) => {
    setFilters((f) => ({
      ...f,
      [key]: f[key].includes(value)
        ? f[key].filter((v) => v !== value)
        : [...f[key], value],
    }));
  };

  return (
    <div className="overflow-hidden rounded-lg border border-[#2E2E32] bg-[#17171A]">
      <div className="border-b border-[#2E2E32] p-3">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-xs font-semibold uppercase tracking-[0.15em] text-[#94A3B8]">
            Monitored Transactions
          </h3>
          <span className="text-xs text-[#64748B]">{filtered.length} active</span>
        </div>
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-[#64748B]" />
            <Input
              placeholder="Search route or tx id..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="h-8 border-[#2E2E32] bg-[#202022] pl-8 text-[#F8FAFC]"
            />
          </div>
          <Button
            type="button"
            variant={showFilters ? "default" : "outline"}
            size="sm"
            onClick={() => setShowFilters((v) => !v)}
            className="h-8 border-[#2E2E32]"
          >
            <Filter className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      <AnimatePresence initial={false}>
        {showFilters && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden border-b border-[#2E2E32] bg-[#1B1B1E]"
          >
            <div className="grid grid-cols-2 gap-2 p-3">
              {verdicts.map((v) => {
                const selected = filters.verdict.includes(v);
                return (
                  <button
                    key={v}
                    type="button"
                    onClick={() => toggleFilter("verdict", v)}
                    className={`flex items-center justify-between rounded border px-2 py-1 text-xs ${selected ? "border-sky-500 bg-sky-500/10 text-sky-300" : "border-[#2E2E32] text-[#94A3B8]"}`}
                  >
                    <span className="capitalize">{v}</span>
                    {selected && <Check className="h-3 w-3" />}
                  </button>
                );
              })}
              {countries.map((c) => {
                const selected = filters.country.includes(c);
                return (
                  <button
                    key={c}
                    type="button"
                    onClick={() => toggleFilter("country", c)}
                    className={`flex items-center justify-between rounded border px-2 py-1 text-xs ${selected ? "border-slate-400 bg-slate-500/10 text-slate-200" : "border-[#2E2E32] text-[#94A3B8]"}`}
                  >
                    <span>{c}</span>
                    {selected && <Check className="h-3 w-3" />}
                  </button>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="no-scrollbar h-[340px] overflow-y-auto overscroll-contain">
        {filtered.length === 0 ? (
          <div className="p-6 text-center text-sm text-[#64748B]">
            No transactions match current filters.
          </div>
        ) : (
          filtered.map((tx) => (
            <Row
              key={tx.id}
              tx={tx}
              expanded={expandedId === tx.id}
              selected={activeId === tx.id}
              onToggle={() => setExpandedId((curr) => (curr === tx.id ? null : tx.id))}
              onSelect={() => onSelect(tx.id)}
              onDelete={() => onDelete(tx.id)}
              onHover={(id) => onHoverTransaction?.(id)}
            />
          ))
        )}
      </div>
    </div>
  );
}
