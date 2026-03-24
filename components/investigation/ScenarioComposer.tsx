"use client";

import { Plus, Route } from "lucide-react";
import type { ScoreResult, TransactionParameters } from "@/types";
import { Slider } from "@/components/ui/slider";
import {
  Combobox,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
  ComboboxPopup,
} from "@/components/ui/combobox";
import { CheckboxGroup } from "@/components/ui/checkbox-group";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  InteractiveLogsTable,
  type MonitoredTransaction,
} from "@/components/ui/interactive-logs-table-shadcnui";

const BACKGROUND_COLOR = "#202022";
const BORDER_COLOR = "#2E2E32";
const TEXT_PRIMARY = "#F8FAFC";
const TEXT_MUTED = "#475569";

const MESSAGE_TYPES = ["pacs.008", "pacs.009", "pacs.004"] as const;

const COUNTRIES = [
  "USA",
  "UK",
  "Germany",
  "Canada",
  "France",
  "Switzerland",
  "UAE",
  "Singapore",
  "Japan",
  "Australia",
  "Netherlands",
  "Sweden",
  "Spain",
  "Italy",
  "Brazil",
  "India",
  "South Africa",
  "Nigeria",
  "Russia",
  "Cayman Islands",
  "Panama",
  "Iran",
  "Democratic People's Republic of Korea",
  "Myanmar",
  "Syria",
  "Yemen",
  "Venezuela",
  "Haiti",
  "Lebanon",
  "Kuwait",
  "Vietnam",
  "Kenya",
  "Algeria",
  "Angola",
  "Cameroon",
  "Democratic Republic of the Congo",
  "South Sudan",
  "Nepal",
  "Papua New Guinea",
  "Monaco",
  "Bulgaria",
  "Bolivia",
  "Namibia",
  "Lao PDR",
  "Côte d'Ivoire",
  "Virgin Islands (UK)",
];

const countryItems = COUNTRIES.map((c) => ({ label: c, value: c }));
const messageTypeItems = MESSAGE_TYPES.map((m) => ({ label: m, value: m }));

export function ScenarioComposer({
  parameters,
  scoreResult,
  transactions,
  activeId,
  onSelectTransaction,
  onClearSelection,
  onParametersChange,
  onAddTransaction,
  onDeleteTransaction,
  onSimulateRealtime,
  isSimulatingRealtime,
  onVisibleChange,
  onHoverTransaction,
}: {
  parameters: TransactionParameters;
  scoreResult: ScoreResult | null;
  transactions: MonitoredTransaction[];
  activeId: string | null;
  onSelectTransaction: (id: string) => void;
  onClearSelection: () => void;
  onParametersChange: (params: Partial<TransactionParameters>) => void;
  onAddTransaction: () => void;
  onDeleteTransaction: (id: string) => void;
  onSimulateRealtime: () => void;
  isSimulatingRealtime: boolean;
  onVisibleChange: (ids: string[]) => void;
  onHoverTransaction: (id: string | null) => void;
}) {
  const selectedFlags = [
    ...(parameters.ipCountryMatchesSender ? ["ipCountryMatchesSender"] : []),
    ...(parameters.messageHasTypos ? ["messageHasTypos"] : []),
    ...(parameters.isRoundNumber ? ["isRoundNumber"] : []),
  ];

  return (
    <div
      className="flex h-full w-full shrink-0 flex-col overflow-hidden"
      style={{
        backgroundColor: BACKGROUND_COLOR,
        padding: 24,
      }}
    >
      <div className="no-scrollbar overflow-y-auto pr-1">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wider" style={{ color: TEXT_MUTED }}>
            Transaction Composer
          </h2>
          <span className="rounded-md border border-[#2E2E32] px-2 py-1 text-xs text-[#94A3B8]">
            Active score: {scoreResult ? `${Math.round(scoreResult.score * 100)}%` : "--"}
          </span>
        </div>
        {activeId && (
          <div className="mb-3 flex items-center justify-between rounded-md border border-[#2E2E32] bg-[#17171A] px-2.5 py-1.5">
            <span className="text-[11px] text-[#94A3B8]">Viewing selected transaction</span>
            <button
              type="button"
              onClick={onClearSelection}
              className="text-[11px] font-semibold text-sky-300 hover:text-sky-200"
            >
              Clear selection
            </button>
          </div>
        )}

        <div className="space-y-4 rounded-xl border border-[#2E2E32] bg-[#1B1B1E] p-4">
        <label className="block">
          <span className="mb-1 block text-xs" style={{ color: TEXT_MUTED }}>
            Amount (0-500k)
          </span>
          <Slider
            min={0}
            max={500000}
            step={1000}
            value={[parameters.amount]}
            onValueChange={(value) => onParametersChange({ amount: value[0] ?? 0 })}
            className="w-full [&_[data-slot=slider-track]]:bg-[#3A3A3F] [&_[data-slot=slider-range]]:bg-[#94A3B8] [&_[data-slot=slider-thumb]]:border-[#CBD5E1] [&_[data-slot=slider-thumb]]:bg-[#E2E8F0]"
          />
          <span className="mt-1 block font-mono text-xs" style={{ color: TEXT_PRIMARY }}>
            {parameters.amount.toLocaleString()}
          </span>
        </label>

        <label className="block">
          <span className="mb-1 block text-xs" style={{ color: TEXT_MUTED }}>
            Transaction velocity (0-30)
          </span>
          <Slider
            min={0}
            max={30}
            step={1}
            value={[parameters.transactionVelocity]}
            onValueChange={(value) =>
              onParametersChange({ transactionVelocity: value[0] ?? 0 })
            }
            className="w-full [&_[data-slot=slider-track]]:bg-[#3A3A3F] [&_[data-slot=slider-range]]:bg-[#94A3B8] [&_[data-slot=slider-thumb]]:border-[#CBD5E1] [&_[data-slot=slider-thumb]]:bg-[#E2E8F0]"
          />
          <span className="mt-1 block font-mono text-xs" style={{ color: TEXT_PRIMARY }}>
            {parameters.transactionVelocity}x
          </span>
        </label>

        <div className="grid grid-cols-2 gap-3">
          <label className="block">
            <span className="mb-1 block text-xs" style={{ color: TEXT_MUTED }}>
              Sender country
            </span>
            <Combobox
              items={countryItems}
              value={countryItems.find((c) => c.value === parameters.senderCountry) ?? null}
              onValueChange={(item) =>
                item?.value && onParametersChange({ senderCountry: item.value })
              }
            >
              <ComboboxInput placeholder="Select sender" aria-label="Select sender country" />
              <ComboboxPopup>
                <ComboboxEmpty>No country found.</ComboboxEmpty>
                <ComboboxList>
                  {(item) => (
                    <ComboboxItem key={item.value} value={item}>
                      {item.label}
                    </ComboboxItem>
                  )}
                </ComboboxList>
              </ComboboxPopup>
            </Combobox>
          </label>

          <label className="block">
            <span className="mb-1 block text-xs" style={{ color: TEXT_MUTED }}>
              Receiver country
            </span>
            <Combobox
              items={countryItems}
              value={countryItems.find((c) => c.value === parameters.receiverCountry) ?? null}
              onValueChange={(item) =>
                item?.value && onParametersChange({ receiverCountry: item.value })
              }
            >
              <ComboboxInput placeholder="Select receiver" aria-label="Select receiver country" />
              <ComboboxPopup>
                <ComboboxEmpty>No country found.</ComboboxEmpty>
                <ComboboxList>
                  {(item) => (
                    <ComboboxItem key={item.value} value={item}>
                      {item.label}
                    </ComboboxItem>
                  )}
                </ComboboxList>
              </ComboboxPopup>
            </Combobox>
          </label>
        </div>

        <label className="block">
          <span className="mb-1 block text-xs" style={{ color: TEXT_MUTED }}>
            Message type
          </span>
          <Combobox
            items={messageTypeItems}
            value={messageTypeItems.find((m) => m.value === parameters.messageType) ?? null}
            onValueChange={(item) =>
              item?.value && onParametersChange({ messageType: item.value })
            }
          >
            <ComboboxInput placeholder="Select message type" aria-label="Select message type" />
            <ComboboxPopup>
              <ComboboxEmpty>No type found.</ComboboxEmpty>
              <ComboboxList>
                {(item) => (
                  <ComboboxItem key={item.value} value={item}>
                    {item.label}
                  </ComboboxItem>
                )}
              </ComboboxList>
            </ComboboxPopup>
          </Combobox>
        </label>

        <CheckboxGroup
          aria-label="Transaction flags"
          value={selectedFlags}
          onValueChange={(values) => {
            const selected = new Set(values as string[]);
            onParametersChange({
              ipCountryMatchesSender: selected.has("ipCountryMatchesSender"),
              messageHasTypos: selected.has("messageHasTypos"),
              isRoundNumber: selected.has("isRoundNumber"),
            });
          }}
          className="rounded-lg border border-[#2E2E32] bg-[#202022] p-3"
        >
          <Label className="text-[#CBD5E1]">
            <Checkbox value="ipCountryMatchesSender" />
            IP matches sender
          </Label>
          <Label className="text-[#CBD5E1]">
            <Checkbox value="messageHasTypos" />
            Message has typos
          </Label>
          <Label className="text-[#CBD5E1]">
            <Checkbox value="isRoundNumber" />
            Round number
          </Label>
        </CheckboxGroup>

        <Button type="button" onClick={onAddTransaction} className="w-full bg-sky-600 text-white hover:bg-sky-500">
          <Plus className="mr-1 h-4 w-4" />
          Add transaction to monitor
        </Button>
        <Button
          type="button"
          onClick={onSimulateRealtime}
          disabled={isSimulatingRealtime}
          className="w-full border border-[#2E2E32] bg-[#1F2937] text-[#E2E8F0] hover:bg-[#273549] disabled:opacity-60"
        >
          {isSimulatingRealtime ? "Simulating real-time data..." : "Simulate real-time data"}
        </Button>
        </div>

        <div className="mt-3 mb-4 flex items-center gap-2 text-[11px] text-[#64748B]">
          <Route className="h-3.5 w-3.5" />
          Select any transaction to inspect and run counterfactuals.
        </div>
      </div>

      <div className="my-4 border-t" style={{ borderColor: BORDER_COLOR }} aria-hidden />

      <div className="min-h-0">
        <InteractiveLogsTable
          transactions={transactions}
          activeId={activeId}
          onSelect={onSelectTransaction}
          onDelete={onDeleteTransaction}
          onVisibleChange={onVisibleChange}
          onHoverTransaction={onHoverTransaction}
        />
      </div>
    </div>
  );
}