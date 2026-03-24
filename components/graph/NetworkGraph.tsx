"use client";

import dynamic from "next/dynamic";
import { Maximize2, ZoomIn, ZoomOut } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { NetworkEdge, NetworkNode } from "@/types";

const BORDER_COLOR = "#2E2E32";
const SURFACE_COLOR = "#202022";
const CLEAN_COLOR = "#22C55E";
const SUSPICIOUS_COLOR = "#F59E0B";
const FRAUD_COLOR = "#EF4444";
const TEXT_PRIMARY = "#F8FAFC";
const TEXT_MUTED = "#475569";

const NODE_RADIUS: Record<NetworkNode["type"], number> = {
  bank: 18,
  corporate: 14,
  individual: 10,
  shell: 12,
  unknown: 10,
};

const RISK_COLOR: Record<NetworkNode["riskLevel"], string> = {
  clean: CLEAN_COLOR,
  suspicious: SUSPICIOUS_COLOR,
  fraud: FRAUD_COLOR,
};

type GraphRef = {
  zoom: (rate: number) => void;
  zoomToFit: (duration?: number, padding?: number) => void;
};

const ForceGraph2D = dynamic(
  () => import("react-force-graph-2d").then((mod) => mod.default),
  { ssr: false }
);

export type NetworkGraphProps = {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  onNodeClick: (node: NetworkNode) => void;
  focusedNodeId: string | null;
};

type NodeWithPosition = NetworkNode & { x?: number; y?: number };
type LinkWithNodes = { source: NodeWithPosition | string; target: NodeWithPosition | string } & NetworkEdge;

function amountToWidth(amount: number): number {
  const log = Math.log10(Math.max(1, amount));
  const minW = 1;
  const maxW = 4;
  const t = Math.min(1, Math.max(0, log / 5));
  return minW + t * (maxW - minW);
}

export function NetworkGraph({
  nodes,
  edges,
  onNodeClick,
  focusedNodeId,
}: NetworkGraphProps) {
  const graphRef = useRef<GraphRef | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [tooltip, setTooltip] = useState<{
    label: string;
    country: string;
    x: number;
    y: number;
  } | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0]?.contentRect ?? { width: 800, height: 600 };
      setDimensions({ width: Math.max(1, width), height: Math.max(1, height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!tooltip) return;
    const onMove = (e: MouseEvent) => {
      setTooltip((prev) => (prev ? { ...prev, x: e.clientX, y: e.clientY } : null));
    };
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, [tooltip]);

  const graphData = useCallback(() => {
    const links: LinkWithNodes[] = edges.map((e) => ({
      ...e,
      source: e.source,
      target: e.target,
    }));
    return {
      nodes: nodes.map((n) => ({ ...n })),
      links,
    };
  }, [nodes, edges]);

  const handleZoomIn = useCallback(() => {
    graphRef.current?.zoom(1.4);
  }, []);

  const handleZoomOut = useCallback(() => {
    graphRef.current?.zoom(0.6);
  }, []);

  const handleRecenter = useCallback(() => {
    graphRef.current?.zoomToFit(400, 50);
  }, []);

  const data = graphData();

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full overflow-hidden rounded-xl border"
      style={{
        backgroundColor: SURFACE_COLOR,
        borderColor: BORDER_COLOR,
        borderWidth: 1,
        borderRadius: 12,
      }}
    >
      <ForceGraph2D
        ref={graphRef as React.RefObject<unknown>}
        graphData={data}
        backgroundColor="transparent"
        width={dimensions.width}
        height={dimensions.height}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const n = node as NodeWithPosition;
          const radius = NODE_RADIUS[n.type ?? "unknown"];
          const color = RISK_COLOR[n.riskLevel ?? "clean"];
          const x = n.x ?? 0;
          const y = n.y ?? 0;
          const isFocused = n.id === focusedNodeId;

          if (isFocused) {
            ctx.beginPath();
            ctx.arc(x, y, radius + 8, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.3;
            ctx.fill();
            ctx.globalAlpha = 1;
            ctx.beginPath();
            ctx.arc(x, y, radius + 4, 0, 2 * Math.PI);
            ctx.strokeStyle = color;
            ctx.globalAlpha = 0.3;
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.globalAlpha = 1;
          }

          ctx.beginPath();
          ctx.arc(x, y, radius, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.strokeStyle = BORDER_COLOR;
          ctx.lineWidth = 1;
          ctx.stroke();
        }}
        linkCanvasObject={(link, ctx) => {
          const l = link as LinkWithNodes;
          const src = typeof l.source === "object" && l.source !== null ? l.source : { x: 0, y: 0 };
          const tgt = typeof l.target === "object" && l.target !== null ? l.target : { x: 0, y: 0 };
          const sx = (src as NodeWithPosition).x ?? 0;
          const sy = (src as NodeWithPosition).y ?? 0;
          const tx = (tgt as NodeWithPosition).x ?? 0;
          const ty = (tgt as NodeWithPosition).y ?? 0;
          const color = RISK_COLOR[l.riskLevel ?? "clean"];
          const width = Math.min(4, Math.max(1, amountToWidth(l.amount)));

          ctx.beginPath();
          ctx.moveTo(sx, sy);
          ctx.lineTo(tx, ty);
          ctx.strokeStyle = color;
          ctx.lineWidth = width;
          if (l.riskLevel === "fraud") {
            ctx.setLineDash([6, 4]);
          } else {
            ctx.setLineDash([]);
          }
          ctx.stroke();
        }}
        onNodeClick={(node) => onNodeClick(node as NetworkNode)}
        onNodeHover={(node) => {
          if (node) {
            const n = node as NodeWithPosition;
            setTooltip({
              label: n.label ?? n.id,
              country: n.country ?? "",
              x: 0,
              y: 0,
            });
          } else {
            setTooltip(null);
          }
        }}
        cooldownTicks={100}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
      />

      {tooltip && (
        <div
          className="pointer-events-none fixed z-50 rounded px-2 py-1.5 text-xs shadow-lg"
          style={{
            left: tooltip.x,
            top: tooltip.y,
            transform: "translate(10px, 10px)",
            backgroundColor: SURFACE_COLOR,
            border: `1px solid ${BORDER_COLOR}`,
            color: TEXT_PRIMARY,
          }}
        >
          <div className="font-medium">{tooltip.label}</div>
          <div style={{ color: TEXT_MUTED }}>{tooltip.country}</div>
        </div>
      )}

      <div className="absolute right-3 top-3 flex gap-1">
        <button
          type="button"
          onClick={handleZoomIn}
          className="rounded-md p-2 transition-colors hover:bg-white/10"
          style={{ color: TEXT_PRIMARY }}
          aria-label="Zoom in"
        >
          <ZoomIn className="h-4 w-4" />
        </button>
        <button
          type="button"
          onClick={handleZoomOut}
          className="rounded-md p-2 transition-colors hover:bg-white/10"
          style={{ color: TEXT_PRIMARY }}
          aria-label="Zoom out"
        >
          <ZoomOut className="h-4 w-4" />
        </button>
        <button
          type="button"
          onClick={handleRecenter}
          className="rounded-md p-2 transition-colors hover:bg-white/10"
          style={{ color: TEXT_PRIMARY }}
          aria-label="Re-center"
        >
          <Maximize2 className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
