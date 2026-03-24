"use client";

import { useEffect, useRef, useCallback } from "react";
import createGlobe from "cobe";

interface Marker {
  id: string;
  location: [number, number];
  label: string;
}

interface Arc {
  id: string;
  from: [number, number];
  to: [number, number];
  label?: string;
}

export type GlobeMarker = Marker;
export type GlobeArc = Arc;

const DEFAULT_MARKER_COLOR: [number, number, number] = [0.44, 0.58, 0.96];
const DEFAULT_BASE_COLOR: [number, number, number] = [0.13, 0.13, 0.15];
const DEFAULT_ARC_COLOR: [number, number, number] = [0.95, 0.41, 0.37];
const DEFAULT_GLOW_COLOR: [number, number, number] = [0.27, 0.29, 0.36];
const SELECTED_ARC_BOOST = 80;
const DIMMED_ARC_MULTIPLIER = 0.45;

interface GlobeProps {
  markers?: Marker[];
  arcs?: Arc[];
  className?: string;
  markerColor?: [number, number, number];
  baseColor?: [number, number, number];
  arcColor?: [number, number, number];
  glowColor?: [number, number, number];
  dark?: number;
  mapBrightness?: number;
  markerSize?: number;
  markerElevation?: number;
  arcWidth?: number;
  arcHeight?: number;
  speed?: number;
  theta?: number;
  diffuse?: number;
  mapSamples?: number;
  highlightedArcId?: string | null;
  hoveredArcId?: string | null;
}

export function Globe({
  markers = [],
  arcs = [],
  className = "",
  markerColor = DEFAULT_MARKER_COLOR,
  baseColor = DEFAULT_BASE_COLOR,
  arcColor = DEFAULT_ARC_COLOR,
  glowColor = DEFAULT_GLOW_COLOR,
  dark = 1,
  mapBrightness = 3,
  markerSize = 0.02,
  markerElevation = 0.02,
  arcWidth = 0.7,
  arcHeight = 0.22,
  speed = 0.003,
  theta = 0.2,
  diffuse = 1.3,
  mapSamples = 14000,
  highlightedArcId = null,
  hoveredArcId = null,
}: GlobeProps) {
  const toCssSafeId = (value: string): string =>
    value.replace(/[^a-zA-Z0-9_-]/g, "_");

  const markerRuntime = markers.map((m, index) => ({
    ...m,
    cssId: `mk-${index}-${toCssSafeId(m.id)}`,
  }));
  const arcRuntime = arcs.map((a, index) => ({
    ...a,
    cssId: `arc-${index}-${toCssSafeId(a.id)}`,
  }));

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pointerInteracting = useRef<{ x: number; y: number } | null>(null);
  const lastPointer = useRef<{ x: number; y: number; t: number } | null>(null);
  const dragOffset = useRef({ phi: 0, theta: 0 });
  const velocity = useRef({ phi: 0, theta: 0 });
  const phiOffsetRef = useRef(0);
  const thetaOffsetRef = useRef(0);
  const isPausedRef = useRef(false);

  useEffect(() => {
    const handlePointerMove = (e: PointerEvent) => {
      if (pointerInteracting.current !== null) {
        const deltaX = e.clientX - pointerInteracting.current.x;
        const deltaY = e.clientY - pointerInteracting.current.y;
        dragOffset.current = { phi: deltaX / 300, theta: deltaY / 1000 };
        const now = Date.now();
        if (lastPointer.current) {
          const dt = Math.max(now - lastPointer.current.t, 1);
          const maxVelocity = 0.15;
          velocity.current = {
            phi: Math.max(
              -maxVelocity,
              Math.min(maxVelocity, ((e.clientX - lastPointer.current.x) / dt) * 0.3)
            ),
            theta: Math.max(
              -maxVelocity,
              Math.min(maxVelocity, ((e.clientY - lastPointer.current.y) / dt) * 0.08)
            ),
          };
        }
        lastPointer.current = { x: e.clientX, y: e.clientY, t: now };
      }
    };
    window.addEventListener("pointermove", handlePointerMove, { passive: true });
    return () => window.removeEventListener("pointermove", handlePointerMove);
  }, []);

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      pointerInteracting.current = { x: e.clientX, y: e.clientY };
      if (canvasRef.current) canvasRef.current.style.cursor = "grabbing";
      isPausedRef.current = true;
    },
    []
  );

  const handlePointerUp = useCallback(() => {
    if (pointerInteracting.current !== null) {
      phiOffsetRef.current += dragOffset.current.phi;
      thetaOffsetRef.current += dragOffset.current.theta;
      dragOffset.current = { phi: 0, theta: 0 };
      lastPointer.current = null;
    }
    pointerInteracting.current = null;
    if (canvasRef.current) canvasRef.current.style.cursor = "grab";
    isPausedRef.current = false;
  }, []);

  useEffect(() => {
    window.addEventListener("pointerup", handlePointerUp, { passive: true });
    return () => {
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [handlePointerUp]);

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    let globe: ReturnType<typeof createGlobe> | null = null;
    let animationId: number;
    let phi = 0;

    function init() {
      const width = canvas.offsetWidth;
      if (width === 0 || globe) return;
      const selectedArc = highlightedArcId
        ? arcRuntime.find((a) => a.id === highlightedArcId) ?? null
        : null;
      const nonSelectedArcs = selectedArc
        ? arcRuntime.filter((a) => a.id !== selectedArc.id)
        : arcRuntime;
      const selectedArcBoost = selectedArc
        ? Array.from({ length: SELECTED_ARC_BOOST }, () => selectedArc)
        : [];
      const frameArcs = [...nonSelectedArcs, ...selectedArcBoost];
      const frameArcColor: [number, number, number] = selectedArc
        ? [
            arcColor[0] * DIMMED_ARC_MULTIPLIER,
            arcColor[1] * DIMMED_ARC_MULTIPLIER,
            arcColor[2] * DIMMED_ARC_MULTIPLIER,
          ]
        : arcColor;

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      globe = createGlobe(canvas, {
        devicePixelRatio: dpr,
        width,
        height: width,
        phi: 0,
        theta,
        dark,
        diffuse,
        mapSamples,
        mapBrightness,
        baseColor,
        markerColor,
        glowColor,
        markerElevation,
        markers: markerRuntime.map((m) => ({
          location: m.location,
          size: markerSize,
          id: m.cssId,
        })),
        arcs: frameArcs.map((a) => ({
          from: a.from,
          to: a.to,
          id: a.cssId,
        })),
        arcColor: frameArcColor,
        arcWidth: selectedArc ? Math.max(arcWidth, 1.35) : arcWidth,
        arcHeight,
        opacity: 0.9,
      });

      function animate() {
        if (!isPausedRef.current) {
          phi += speed;
          if (
            Math.abs(velocity.current.phi) > 0.0001 ||
            Math.abs(velocity.current.theta) > 0.0001
          ) {
            phiOffsetRef.current += velocity.current.phi;
            thetaOffsetRef.current += velocity.current.theta;
            velocity.current.phi *= 0.95;
            velocity.current.theta *= 0.95;
          }
          const thetaMin = -0.4;
          const thetaMax = 0.4;
          if (thetaOffsetRef.current < thetaMin) {
            thetaOffsetRef.current += (thetaMin - thetaOffsetRef.current) * 0.1;
          } else if (thetaOffsetRef.current > thetaMax) {
            thetaOffsetRef.current += (thetaMax - thetaOffsetRef.current) * 0.1;
          }
        }
        const selectedFrameArc = highlightedArcId
          ? arcRuntime.find((a) => a.id === highlightedArcId) ?? null
          : null;
        const nonSelectedFrameArcs = selectedFrameArc
          ? arcRuntime.filter((a) => a.id !== selectedFrameArc.id)
          : arcRuntime;
        const selectedFrameBoost = selectedFrameArc
          ? Array.from({ length: SELECTED_ARC_BOOST }, () => selectedFrameArc)
          : [];
        const animatedArcs = [...nonSelectedFrameArcs, ...selectedFrameBoost];
        const animatedArcColor: [number, number, number] = selectedFrameArc
          ? [
              arcColor[0] * DIMMED_ARC_MULTIPLIER,
              arcColor[1] * DIMMED_ARC_MULTIPLIER,
              arcColor[2] * DIMMED_ARC_MULTIPLIER,
            ]
          : arcColor;
        globe?.update({
          phi: phi + phiOffsetRef.current + dragOffset.current.phi,
          theta: theta + thetaOffsetRef.current + dragOffset.current.theta,
          dark,
          mapBrightness,
          markerColor,
          baseColor,
          arcColor: animatedArcColor,
          arcWidth: selectedFrameArc ? Math.max(arcWidth, 1.35) : arcWidth,
          markerElevation,
          markers: markerRuntime.map((m) => ({
            location: m.location,
            size: markerSize,
            id: m.cssId,
          })),
          arcs: animatedArcs.map((a) => ({
            from: a.from,
            to: a.to,
            id: a.cssId,
          })),
        });
        animationId = requestAnimationFrame(animate);
      }
      animate();
      setTimeout(() => {
        if (canvas) canvas.style.opacity = "1";
      });
    }

    if (canvas.offsetWidth > 0) {
      init();
    } else {
      const ro = new ResizeObserver((entries) => {
        if (entries[0]?.contentRect.width > 0) {
          ro.disconnect();
          init();
        }
      });
      ro.observe(canvas);
    }

    return () => {
      if (animationId) cancelAnimationFrame(animationId);
      if (globe) globe.destroy();
    };
  }, [
    markers,
    arcs,
    markerRuntime,
    arcRuntime,
    markerColor,
    baseColor,
    arcColor,
    glowColor,
    dark,
    mapBrightness,
    markerSize,
    markerElevation,
    arcWidth,
    arcHeight,
    speed,
    theta,
    diffuse,
    mapSamples,
    highlightedArcId,
    hoveredArcId,
  ]);

  return (
    <div className={`relative aspect-square select-none ${className}`}>
      <canvas
        ref={canvasRef}
        onPointerDown={handlePointerDown}
        style={{
          width: "100%",
          height: "100%",
          cursor: "grab",
          opacity: 0,
          transition: "opacity 0.8s ease",
          borderRadius: "50%",
          touchAction: "none",
        }}
      />
      {markerRuntime.map((m) => (
        <div
          key={m.cssId}
          style={{
            position: "absolute",
            positionAnchor: `--cobe-${m.cssId}`,
            bottom: "anchor(top)",
            left: "anchor(center)",
            translate: "-50% 0",
            marginBottom: 8,
            padding: "2px 6px",
            background: "#1a1a2e",
            color: "#fff",
            fontFamily: "monospace",
            fontSize: "0.6rem",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            whiteSpace: "nowrap",
            pointerEvents: "none",
            opacity: highlightedArcId
              ? `calc(var(--cobe-visible-${m.cssId}, 0.92) * 0.35)`
              : `var(--cobe-visible-${m.cssId}, 0.92)`,
            filter: `blur(calc((1 - var(--cobe-visible-${m.cssId}, 0.92)) * 8px))`,
            transition: "opacity 0.8s, filter 0.8s",
          }}
        >
          {m.label}
          <span
            style={{
              position: "absolute",
              top: "100%",
              left: "50%",
              transform: "translate3d(-50%, -1px, 0)",
              border: "5px solid transparent",
              borderTopColor: "#1a1a2e",
            }}
          />
        </div>
      ))}
      {arcRuntime
        .filter((a) => a.label)
        .map((a) => (
          <div
            key={a.cssId}
            style={{
              position: "absolute",
              positionAnchor: `--cobe-arc-${a.cssId}`,
              bottom: "anchor(top)",
              left: "anchor(center)",
              translate: "-50% 0",
              marginBottom: 8,
              padding: "2px 6px",
              background: "#fff",
              color: "#1a1a2e",
              fontFamily: "monospace",
              fontSize: "0.6rem",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              whiteSpace: "nowrap",
              pointerEvents: "none",
              boxShadow: "0 1px 4px rgba(0,0,0,0.1)",
              opacity:
                !hoveredArcId
                  ? "0"
                  : hoveredArcId === a.id
                    ? `var(--cobe-visible-arc-${a.cssId}, 0.92)`
                    : "0",
              filter: `blur(calc((1 - var(--cobe-visible-arc-${a.cssId}, 0.92)) * 8px))`,
              transition: "opacity 0.8s, filter 0.8s",
            }}
          >
            {a.label}
            <span
              style={{
                position: "absolute",
                top: "100%",
                left: "50%",
                transform: "translate3d(-50%, -1px, 0)",
                border: "5px solid transparent",
                borderTopColor: "#fff",
              }}
            />
          </div>
        ))}
    </div>
  );
}
