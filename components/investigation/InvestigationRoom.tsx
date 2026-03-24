"use client";

import { useDebouncedCallback } from "use-debounce";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CaseFilePanel } from "@/components/case-file/CaseFilePanel";
import type { CounterfactualChanges } from "@/components/case-file/CaseFilePanel";
import { SignalTimeline } from "./SignalTimeline";
import { ScenarioComposer } from "./ScenarioComposer";
import { deriveExplainFromTransaction, scoreTransaction } from "@/lib/api";
import type { ScoreResult, TransactionParameters } from "@/types";
import { Globe, type GlobeArc, type GlobeMarker } from "@/components/ui/cobe-globe-cdn";
import type { MonitoredTransaction } from "@/components/ui/interactive-logs-table-shadcnui";

const BACKGROUND_COLOR = "#161618";
const PANEL_WIDTH = 360;
const BORDER_COLOR = "#2E2E32";

const COUNTRY_COORDS: Record<string, [number, number]> = {
  USA: [38.9, -77.03],
  UK: [51.5, -0.12],
  Germany: [52.52, 13.4],
  Canada: [45.42, -75.69],
  France: [48.86, 2.35],
  Switzerland: [46.95, 7.45],
  UAE: [24.45, 54.38],
  Singapore: [1.29, 103.85],
  Japan: [35.68, 139.69],
  Australia: [-35.28, 149.13],
  Netherlands: [52.37, 4.9],
  Sweden: [59.33, 18.07],
  Spain: [40.42, -3.7],
  Italy: [41.9, 12.5],
  Brazil: [-15.79, -47.88],
  India: [28.61, 77.21],
  "South Africa": [-25.75, 28.19],
  Nigeria: [6.52, 3.37],
  Russia: [55.75, 37.62],
  "Cayman Islands": [19.31, -81.25],
  Panama: [8.98, -79.52],
  Iran: [35.69, 51.39],
  "Democratic People's Republic of Korea": [39.04, 125.75],
  Myanmar: [19.75, 96.1],
  Syria: [33.51, 36.29],
  Yemen: [15.35, 44.21],
  Venezuela: [10.48, -66.9],
  Haiti: [18.54, -72.34],
  Lebanon: [33.89, 35.5],
  Kuwait: [29.38, 47.99],
  Vietnam: [21.03, 105.85],
  Kenya: [-1.29, 36.82],
  Algeria: [36.75, 3.04],
  Angola: [-8.83, 13.23],
  Cameroon: [3.85, 11.5],
  "Democratic Republic of the Congo": [-4.32, 15.31],
  "South Sudan": [4.85, 31.6],
  Nepal: [27.71, 85.32],
  "Papua New Guinea": [-9.44, 147.18],
  Monaco: [43.73, 7.42],
  Bulgaria: [42.7, 23.32],
  Bolivia: [-16.5, -68.15],
  Namibia: [-22.57, 17.08],
  "Lao PDR": [17.97, 102.61],
  "Côte d'Ivoire": [5.34, -4.03],
  "Virgin Islands (UK)": [18.43, -64.62],
};
const COUNTRY_LIST = Object.keys(COUNTRY_COORDS);
const MESSAGE_TYPES = ["pacs.008", "pacs.009", "pacs.004"] as const;

const DEFAULT_PARAMETERS: TransactionParameters = {
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

function pad2(n: number): string {
  return String(n).padStart(2, "0");
}

function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function createTransaction(parameters: TransactionParameters): MonitoredTransaction {
  const id = `tx-${Date.now()}-${randomInt(100, 999)}`;
  const chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  const transactionId = `TX-${Array.from({ length: 6 }, () => chars[randomInt(0, chars.length - 1)]).join("")}`;
  const minute = randomInt(6, 54);
  const second = randomInt(8, 58);
  return {
    id,
    timestamp: new Date().toISOString(),
    transactionId,
    flaggedAt: `${pad2(parameters.hour)}:${pad2(minute)}:${pad2(second)} UTC`,
    parameters: { ...parameters },
    scoreResult: null,
  };
}

function createSeedTransaction(parameters: TransactionParameters): MonitoredTransaction {
  return {
    id: "tx-seed-1",
    timestamp: "2026-01-01T14:22:11.000Z",
    transactionId: "TX-SEED01",
    flaggedAt: `${pad2(parameters.hour)}:22:11 UTC`,
    parameters: { ...parameters },
    scoreResult: null,
  };
}

const TRANSACTION_PARAM_KEYS: Set<keyof TransactionParameters> = new Set([
  "amount",
  "hour",
  "dayOfWeek",
  "accountAgeDays",
  "senderCountry",
  "receiverCountry",
  "transactionVelocity",
  "ipCountryMatchesSender",
  "messageHasTypos",
  "messageType",
  "isRoundNumber",
]);

function changesToParameters(
  changes: CounterfactualChanges
): Partial<TransactionParameters> {
  const params: Partial<TransactionParameters> = {};
  const numberKeys: Set<keyof TransactionParameters> = new Set([
    "amount",
    "hour",
    "dayOfWeek",
    "accountAgeDays",
    "transactionVelocity",
  ]);
  const booleanKeys: Set<keyof TransactionParameters> = new Set([
    "ipCountryMatchesSender",
    "messageHasTypos",
    "isRoundNumber",
  ]);
  for (const [key, value] of Object.entries(changes)) {
    const k = key as keyof TransactionParameters;
    if (!TRANSACTION_PARAM_KEYS.has(k)) continue;
    const to = value.to;
    if (numberKeys.has(k)) {
      const n = typeof to === "number" ? to : Number(to);
      if (Number.isFinite(n)) (params as Record<string, number>)[k] = n;
    } else if (booleanKeys.has(k)) {
      if (typeof to === "boolean") (params as Record<string, boolean>)[k] = to;
    } else if (typeof to === "string") {
      (params as Record<string, string>)[k] = to;
    }
  }
  return params;
}

export function InvestigationRoom() {
  const [transactions, setTransactions] = useState<MonitoredTransaction[]>(() => [
    createSeedTransaction(DEFAULT_PARAMETERS),
  ]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [draftParameters, setDraftParameters] =
    useState<TransactionParameters>(DEFAULT_PARAMETERS);
  const [isSimulatingRealtime, setIsSimulatingRealtime] = useState(false);
  const [visibleTransactionIds, setVisibleTransactionIds] = useState<string[]>([]);
  const [hoveredTransactionId, setHoveredTransactionId] = useState<string | null>(null);
  const requestSeqRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const simulationTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const simulationRemainingRef = useRef(0);

  const activeTransaction = useMemo(
    () => transactions.find((t) => t.id === activeId) ?? null,
    [transactions, activeId]
  );
  const activeParameters = activeTransaction?.parameters ?? null;
  const selectedParameters = useMemo(
    () => activeTransaction?.parameters ?? draftParameters,
    [activeTransaction, draftParameters]
  );
  const selectedScoreResult = useMemo(
    () => activeTransaction?.scoreResult ?? deriveExplainFromTransaction(selectedParameters),
    [activeTransaction, selectedParameters]
  );
  const displayScoreResult = selectedScoreResult;

  const setActiveScore = useCallback((id: string, result: ScoreResult | null) => {
    setTransactions((curr) =>
      curr.map((tx) => {
        if (tx.id !== id) return tx;
        if (JSON.stringify(tx.scoreResult) === JSON.stringify(result)) return tx;
        return { ...tx, scoreResult: result };
      })
    );
  }, []);

  const debouncedScore = useDebouncedCallback(
    async (params: TransactionParameters, txId: string, seq: number) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const result = await scoreTransaction(params, controller.signal);
        if (requestSeqRef.current === seq) {
          setActiveScore(txId, result);
        }
      } catch (error) {
        if ((error as Error).name === "AbortError") return;
        if (requestSeqRef.current === seq) {
          setActiveScore(txId, null);
        }
      }
    },
    180
  );

  useEffect(() => {
    if (!activeId || !activeParameters) return;
    requestSeqRef.current += 1;
    const seq = requestSeqRef.current;
    debouncedScore(activeParameters, activeId, seq);
  }, [activeId, activeParameters, debouncedScore]);

  useEffect(
    () => () => {
      abortRef.current?.abort();
    },
    []
  );

  const onParametersChange = useCallback(
    (params: Partial<TransactionParameters>) => {
      if (activeId) {
        setTransactions((curr) =>
          curr.map((tx) =>
            tx.id === activeId
              ? { ...tx, parameters: { ...tx.parameters, ...params } }
              : tx
          )
        );
        return;
      }
      setDraftParameters((curr) => ({ ...curr, ...params }));
    },
    [activeId]
  );

  const onAddTransaction = useCallback(() => {
    const created = createTransaction(selectedParameters);
    setTransactions((curr) => [created, ...curr]);
    setActiveId(null);
  }, [selectedParameters]);

  const onDeleteTransaction = useCallback((id: string) => {
    setTransactions((curr) => curr.filter((tx) => tx.id !== id));
    setActiveId((curr) => (curr === id ? null : curr));
  }, []);

  const handleTryScenario = useCallback(
    (changes: CounterfactualChanges) => {
      if (!activeId) return;
      const params = changesToParameters(changes);
      setTransactions((curr) =>
        curr.map((tx) =>
          tx.id === activeId
            ? { ...tx, parameters: { ...tx.parameters, ...params } }
            : tx
        )
      );
    },
    [activeId]
  );

  const randomParameters = useCallback((): TransactionParameters => {
    const sender = COUNTRY_LIST[randomInt(0, COUNTRY_LIST.length - 1)] ?? "USA";
    let receiver = COUNTRY_LIST[randomInt(0, COUNTRY_LIST.length - 1)] ?? "UK";
    if (receiver === sender) {
      receiver =
        COUNTRY_LIST[(COUNTRY_LIST.indexOf(sender) + 1) % COUNTRY_LIST.length] ?? "UK";
    }
    return {
      amount: randomInt(1000, 500000),
      hour: randomInt(0, 23),
      dayOfWeek: randomInt(0, 6),
      accountAgeDays: randomInt(0, 1500),
      senderCountry: sender,
      receiverCountry: receiver,
      transactionVelocity: randomInt(0, 30),
      ipCountryMatchesSender: randomInt(0, 1) === 1,
      messageHasTypos: randomInt(0, 1) === 1,
      messageType: MESSAGE_TYPES[randomInt(0, MESSAGE_TYPES.length - 1)],
      isRoundNumber: randomInt(0, 1) === 1,
    };
  }, []);

  const onSimulateRealtime = useCallback(() => {
    if (isSimulatingRealtime) return;
    setIsSimulatingRealtime(true);
    if (simulationTimerRef.current) {
      clearInterval(simulationTimerRef.current);
      simulationTimerRef.current = null;
    }

    simulationRemainingRef.current = 15;

    const tick = () => {
      setTransactions((curr) => {
        if (simulationRemainingRef.current <= 0) {
          if (simulationTimerRef.current) {
            clearInterval(simulationTimerRef.current);
            simulationTimerRef.current = null;
          }
          setIsSimulatingRealtime(false);
          return curr;
        }
        const tx = createTransaction(randomParameters());
        tx.scoreResult = deriveExplainFromTransaction(tx.parameters);
        simulationRemainingRef.current -= 1;
        return [tx, ...curr];
      });
    };

    tick();
    simulationTimerRef.current = setInterval(tick, 650);
  }, [isSimulatingRealtime, randomParameters]);

  useEffect(
    () => () => {
      if (simulationTimerRef.current) clearInterval(simulationTimerRef.current);
    },
    []
  );

  const globeMarkers = useMemo<GlobeMarker[]>(() => {
    const uniq = new Map<string, GlobeMarker>();
    const visible = new Set(visibleTransactionIds.length ? visibleTransactionIds : transactions.map((t) => t.id));
    transactions.forEach((tx) => {
      if (!visible.has(tx.id)) return;
      [tx.parameters.senderCountry, tx.parameters.receiverCountry].forEach((country) => {
        const location = COUNTRY_COORDS[country];
        if (!location || uniq.has(country)) return;
        uniq.set(country, {
          id: country,
          location,
          label: country,
        });
      });
    });
    return Array.from(uniq.values());
  }, [transactions, visibleTransactionIds]);

  const globeArcs = useMemo<GlobeArc[]>(() => {
    const visible = new Set(visibleTransactionIds.length ? visibleTransactionIds : transactions.map((t) => t.id));
    return transactions
      .filter((tx) => visible.has(tx.id))
      .filter((tx) => COUNTRY_COORDS[tx.parameters.senderCountry] && COUNTRY_COORDS[tx.parameters.receiverCountry])
      .map((tx, i) => {
        const from = COUNTRY_COORDS[tx.parameters.senderCountry];
        const to = COUNTRY_COORDS[tx.parameters.receiverCountry];
        const jitter = ((i % 5) - 2) * 0.9;
        return {
        id: tx.id,
        from: [from[0] + jitter, from[1]] as [number, number],
        to: [to[0] - jitter, to[1]] as [number, number],
        label: `${tx.parameters.senderCountry} -> ${tx.parameters.receiverCountry} (${tx.transactionId}) ${
          tx.scoreResult ? Math.round(tx.scoreResult.score * 100) : "--"
        }%`,
      };
      });
  }, [transactions, visibleTransactionIds]);

  return (
    <div
      className="flex h-screen overflow-hidden"
      style={{ backgroundColor: BACKGROUND_COLOR }}
    >
      <aside
        className="no-scrollbar h-full shrink-0 overflow-y-auto border-r"
        style={{ width: PANEL_WIDTH, borderColor: BORDER_COLOR }}
      >
        <ScenarioComposer
          parameters={selectedParameters}
          scoreResult={selectedScoreResult}
          transactions={transactions}
          activeId={activeId}
          onSelectTransaction={(id) =>
            setActiveId((current) => (current === id ? null : id))
          }
          onClearSelection={() => setActiveId(null)}
          onParametersChange={onParametersChange}
          onAddTransaction={onAddTransaction}
          onDeleteTransaction={onDeleteTransaction}
          onSimulateRealtime={onSimulateRealtime}
          isSimulatingRealtime={isSimulatingRealtime}
          onVisibleChange={setVisibleTransactionIds}
          onHoverTransaction={setHoveredTransactionId}
        />
      </aside>
      <main className="flex min-w-0 flex-1 flex-col overflow-hidden border-r" style={{ borderColor: BORDER_COLOR }}>
        <section className="border-b p-6" style={{ borderColor: BORDER_COLOR }}>
          <div className="rounded-2xl border border-[#2E2E32] bg-[#1A1A1D] p-4">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-sm font-semibold uppercase tracking-[0.14em] text-[#94A3B8]">
                Transaction Globe
              </h3>
              <span className="text-xs text-[#64748B]">{globeArcs.length} active routes</span>
            </div>
            <Globe
              markers={globeMarkers}
              arcs={globeArcs}
              highlightedArcId={activeId}
              hoveredArcId={hoveredTransactionId}
              className="mx-auto w-full max-w-[420px]"
            />
          </div>
        </section>
        <div className="h-full overflow-hidden border-t" style={{ borderColor: BORDER_COLOR }}>
          <SignalTimeline
            scoreResult={displayScoreResult}
            parameters={selectedParameters}
            flaggedAt={activeTransaction?.flaggedAt ?? `${pad2(selectedParameters.hour)}:00:00 UTC`}
          />
        </div>
      </main>
      <aside
        className="no-scrollbar h-full shrink-0 overflow-y-auto"
        style={{ width: PANEL_WIDTH }}
      >
        <CaseFilePanel
          scoreResult={activeId ? displayScoreResult : null}
          parameters={selectedParameters}
          transactionId={activeTransaction?.transactionId ?? "TX-000000"}
          flaggedAt={activeTransaction?.flaggedAt ?? `${pad2(selectedParameters.hour)}:00:00 UTC`}
          onTryScenario={handleTryScenario}
        />
      </aside>
    </div>
  );
}