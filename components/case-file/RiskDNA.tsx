"use client";

import { motion } from "motion/react";
import { useCallback, useState } from "react";

const FRAUD_COLOR = "#EF4444";
const CLEAN_COLOR = "#22C55E";
const GUIDE_COLOR = "#2E2E32";

export type RiskDNAFeature = {
  name: string;
  value: number;
  contribution: number;
};

export type RiskDNAProps = {
  features: RiskDNAFeature[];
  width?: number;
  height?: number;
  showLabels?: boolean;
  animated?: boolean;
};

export function RiskDNA({
  features,
  width = 320,
  height = 64,
  showLabels = false,
  animated = false,
}: RiskDNAProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  const segmentWidth = features.length > 0 ? width / features.length : 0;
  const maxH = height / 2;
  const centerY = maxH;

  const formatContribution = useCallback((c: number) => {
    const pct = Math.round(c * 100);
    return pct >= 0 ? `+${pct}%` : `${pct}%`;
  }, []);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<SVGRectElement>, index: number) => {
      setHoveredIndex(index);
      setTooltipPos({ x: e.clientX, y: e.clientY });
    },
    []
  );

  if (features.length === 0) {
    return (
      <svg width={width} height={height} aria-hidden>
        <rect x={0} y={0} width={width} height={height} fill="transparent" />
      </svg>
    );
  }

  return (
    <div className="inline-block">
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="overflow-visible"
        aria-label="Risk signature"
      >
        {[0.25, 0.5, 0.75].map((p) => (
          <line
            key={p}
            x1={width * p}
            y1={4}
            x2={width * p}
            y2={height - 4}
            stroke={GUIDE_COLOR}
            strokeWidth={1}
            opacity={0.55}
          />
        ))}
        {features.map((f, i) => {
          const barH = Math.max(4, Math.abs(f.contribution) * maxH);
          const y =
            f.contribution > 0 ? centerY - barH : centerY;
          const color = f.contribution > 0 ? FRAUD_COLOR : CLEAN_COLOR;
          const opacity = 0.3 + 0.7 * Math.abs(f.contribution);
          const x = i * segmentWidth + 1;
          const segWidth = segmentWidth - 2;

          const rectProps = {
            x,
            y,
            width: segWidth,
            height: barH,
            fill: color,
            opacity,
            rx: 1,
            onMouseEnter: (e: React.MouseEvent<SVGRectElement>) => {
              setHoveredIndex(i);
              setTooltipPos({ x: e.clientX, y: e.clientY });
            },
            onMouseMove: (e: React.MouseEvent<SVGRectElement>) =>
              handleMouseMove(e, i),
            onMouseLeave: () => setHoveredIndex(null),
          };

          const title = (
            <title>
              {f.name}: {formatContribution(f.contribution)}
            </title>
          );

          if (animated) {
            return (
              <g key={f.name}>
                <motion.rect
                  {...rectProps}
                  initial={{ height: 0, y: centerY, opacity: 0 }}
                  animate={{ height: barH, y, opacity }}
                  transition={{
                    duration: 0.25,
                    delay: i * 0.03,
                    ease: "easeOut",
                  }}
                />
                {title}
              </g>
            );
          }

          return (
            <g key={f.name}>
              <rect {...rectProps} />
              {title}
            </g>
          );
        })}
      </svg>
      {hoveredIndex !== null && features[hoveredIndex] && (
        <div
          className="pointer-events-none fixed z-50 rounded px-2 py-1 text-xs font-medium text-[#F8FAFC] shadow-lg"
          style={{
            left: tooltipPos.x,
            top: tooltipPos.y - 8,
            transform: "translate(-50%, -100%)",
            backgroundColor: "#1A1A24",
            border: "1px solid #2E2E32",
          }}
        >
          {features[hoveredIndex].name}:{" "}
          {formatContribution(features[hoveredIndex].contribution)}
        </div>
      )}
      {showLabels && features.length > 0 && (
        <div
          className="mt-2 flex text-[11px] font-medium text-[#64748B]"
          style={{ width }}
        >
          {features.map((f) => (
            <div
              key={f.name}
              className="px-1 text-center leading-tight whitespace-normal"
              style={{
                width: segmentWidth,
                wordBreak: "normal",
                overflowWrap: "normal",
              }}
              title={f.name}
            >
              {f.name}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}