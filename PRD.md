# Fraud Detection Platform — UI Redesign PRD
### Project: ISO 20022 Fraud Detection System
### Version: 1.0 | Author: Fares | Status: Ready for Implementation

---

## 0. Document Purpose

This PRD is written as a direct input for Cursor (AI-assisted development). Every section is structured to give the AI maximum context with minimum ambiguity. Follow the implementation phases in strict order — each phase is a shippable increment.

---

## 1. Vision & Framing

### The Concept

Stop building a "dashboard." Build a **case room.**

Every existing fraud detection UI looks like a Bloomberg terminal — tables, scores, submit buttons. The framing is always "system processes transaction." This product's framing is the inverse: **the analyst investigates a case.** This single conceptual shift drives every design and feature decision in this document.

### Who This Is For

- **Primary audience:** LinkedIn / public portfolio (fintech engineers, hiring managers, the payments/banking community on LinkedIn)
- **Secondary audience:** Academic presentations (NSERC CREATE Conference)
- **Not the primary goal:** Impressing a specific recruiter (that goal is already achieved)

### What Success Looks Like

Someone lands on the live URL, spends 3+ minutes exploring without being told what to do, and thinks: *"I've never seen a fraud tool demo that looks like this."*

---

## 2. Current State

### Stack (existing)
- **Backend:** Flask (Python), `app.py` with `/api/predict` and `/api/batch` endpoints
- **Model:** RandomForestClassifier, `models/fraud_detector.pkl`, ~99.9% accuracy on synthetic data
- **Frontend:** Plain HTML/CSS/JS in `templates/index.html` — single transaction form + batch CSV upload
- **Deployed:** `https://swift-fraud-detection.onrender.com/`

### Problems with Current UI
- Input form with a submit button. Forgettable.
- No visual identity. Looks like every other ML demo.
- User has to do work to understand what the system does.
- No narrative, no investigation metaphor, no "wow" moment.

---

## 3. Target Stack

### Frontend (new)
| Layer | Choice | Rationale |
|---|---|---|
| Framework | **Next.js 14** (App Router) | Already using at Reroute; familiar; file-based routing maps cleanly to our views |
| Language | **TypeScript** | Non-negotiable for a portfolio piece in 2025 |
| Styling | **Tailwind CSS v4** | Latest version; works with shadcn CLI 3.0 |
| Component library | **shadcn/ui** (CLI 3.0) | Copy-owned components, no vendor lock-in, Recharts charts baked in |
| Animation | **Motion (formerly Framer Motion) v12** | Gold standard for React UI animation; declarative API; spring physics; 30M+ npm downloads/month |
| Graph visualization | **react-force-graph** (vasturiano) | Canvas-based, WebGL-ready, force-directed, draggable nodes, zero config, best maintained |
| Icons | **lucide-react** | Ships with shadcn; consistent stroke weight |
| State management | **Zustand** | Lightweight; no boilerplate; perfect for shared investigation state |
| Data fetching | **TanStack Query (React Query) v5** | Caching, loading states, refetch — essential for real-time feel |

### Backend (minimal changes)
- Keep Flask as-is
- Add CORS headers for Next.js dev server
- Add one new endpoint: `POST /api/explain` that returns SHAP-style feature contributions per transaction
- Add `POST /api/graph` that returns a synthetic network of related transactions given a transaction ID

### Deployment
- Frontend: **Vercel** (free tier, instant deploys from GitHub, auto HTTPS)
- Backend: Keep on Render, add environment variable for allowed CORS origin

---

## 4. Design System

### Color Palette

This is a dark-first design. Light mode is supported but the primary experience is dark.

```
Background:     #0A0A0F  (near-black, slightly blue-tinted)
Surface:        #111118  (cards, panels)
Surface raised: #1A1A24  (modals, dropdowns)
Border:         #2A2A3A  (subtle dividers)

Risk — Clean:       #22C55E  (green-500)
Risk — Suspicious:  #F59E0B  (amber-500)
Risk — Fraud:       #EF4444  (red-500)
Risk — Unknown:     #6366F1  (indigo-500)

Accent:         #6366F1  (indigo — primary interactive)
Accent hover:   #818CF8  (indigo-400)

Text primary:   #F8FAFC
Text secondary: #94A3B8
Text muted:     #475569
```

### Typography
- **Font:** Geist (Vercel's open-source font, available via `next/font/google`) — `Geist` for sans, `Geist Mono` for numbers/code
- **Scale:** 12 / 14 / 16 / 20 / 24 / 32px — nothing else

### Motion Principles
- Entry animations: fade-in + translateY(8px) → translateY(0), 200ms ease-out
- Hover states: scale(1.02), 150ms
- Risk score changes: spring physics (stiffness 300, damping 30)
- No animation longer than 400ms
- All animations respect `prefers-reduced-motion`

### Spacing
- Base unit: 4px
- All spacing multiples of 4
- Card padding: 24px
- Section gaps: 32px

---

## 5. Information Architecture

```
/                   → Landing / Investigation Room (main experience)
/case/[id]          → Individual Case File view
/network            → Full-screen Relationship Map
/scenarios          → Scenario Composer
/analyst            → Analyst Dashboard (table view)
```

The `/` route is the hero experience. Everything else is reachable from it.

---

## 6. Feature Specifications

---

### Feature 1 — The Investigation Room (Homepage)

**What it is:** The landing page IS the product. No hero text, no marketing copy. The user arrives directly inside the investigation interface.

**Layout (three-panel):**
```
┌─────────────────┬──────────────────────┬───────────────────┐
│  LEFT PANEL     │   CENTER PANEL       │   RIGHT PANEL     │
│  Scenario       │   Network Graph      │   Case File       │
│  Composer       │   (primary visual)   │   (score +        │
│  (inputs)       │                      │    explanation)   │
│                 │                      │                   │
│  340px wide     │   flex-1             │   380px wide      │
└─────────────────┴──────────────────────┴───────────────────┘
```

On mobile: stacked vertically, center panel first.

**Behavior:**
1. On load: network graph renders with a pre-loaded synthetic scenario (BEC fraud archetype). The case file on the right shows the flagged transaction already scored. This means the user sees a fully alive, working system in the first 500ms — not an empty form.
2. User can modify parameters in the left panel → network and case file update live (debounced 300ms)
3. User can click any node in the network graph → case file updates to show that node's transaction

---

### Feature 2 — Scenario Composer (Left Panel)

**What it is:** Instead of raw number inputs, the user picks a named fraud archetype. The system pre-populates all parameters with a realistic profile matching that archetype. The user can then fine-tune individual values.

**Archetypes (pre-loaded):**
```typescript
type FraudArchetype = {
  id: string
  name: string
  description: string
  icon: string  // lucide icon name
  baseRiskLevel: 'clean' | 'suspicious' | 'fraud'
  parameters: TransactionParameters
}

const ARCHETYPES: FraudArchetype[] = [
  {
    id: 'bec',
    name: 'Business Email Compromise',
    description: 'Corporate account redirected, large wire, new beneficiary',
    icon: 'mail-warning',
    baseRiskLevel: 'fraud',
    parameters: { amount: 187500, hour: 14, dayOfWeek: 1, accountAgeDays: 3, ... }
  },
  {
    id: 'romance-scam',
    name: 'Romance Scam',
    description: 'Individual victim, repeated transfers, offshore destination',
    icon: 'heart-crack',
    baseRiskLevel: 'fraud',
    parameters: { ... }
  },
  {
    id: 'structuring',
    name: 'Structuring / Smurfing',
    description: 'Multiple sub-threshold transactions, same beneficiary',
    icon: 'layers',
    baseRiskLevel: 'suspicious',
    parameters: { ... }
  },
  {
    id: 'money-mule',
    name: 'Money Mule Chain',
    description: 'Rapid pass-through, multiple hops, mismatched countries',
    icon: 'arrow-right-left',
    baseRiskLevel: 'fraud',
    parameters: { ... }
  },
  {
    id: 'corporate-payroll',
    name: 'Corporate Payroll Run',
    description: 'Batch, regular schedule, established accounts, domestic',
    icon: 'building-2',
    baseRiskLevel: 'clean',
    parameters: { ... }
  },
  {
    id: 'custom',
    name: 'Custom Transaction',
    description: 'Define all parameters manually',
    icon: 'settings-2',
    baseRiskLevel: 'unknown',
    parameters: { ...defaults }
  }
]
```

**UI components (all shadcn):**
- Archetype selector: custom card grid, one card per archetype, colored border matching risk level
- Parameter sliders: `Slider` component for amount, account age, transaction velocity
- Parameter selects: `Select` for countries, message type
- Toggle switches: `Switch` for boolean flags (is_round_number, ip_country_matches, etc.)
- Each parameter shows its contribution to the risk score as a small colored tag: `+12% risk` / `-8% risk`

**Implementation notes:**
```typescript
// Debounced scoring — don't hit the API on every keystroke
const debouncedScore = useDebouncedCallback(async (params) => {
  const result = await scoreTransaction(params)
  setScoreResult(result)
}, 300)

// Call debouncedScore on every parameter change
useEffect(() => {
  debouncedScore(parameters)
}, [parameters])
```

---

### Feature 3 — Relationship Map (Center Panel)

**What it is:** A live force-directed graph showing the network of accounts and transactions connected to the currently selected scenario. Not decorative — it IS the data.

**Library:** `react-force-graph-2d` from `vasturiano/react-force-graph`

**Installation:**
```bash
npm install react-force-graph-2d
```

**Node types:**
```typescript
type NetworkNode = {
  id: string
  label: string
  type: 'bank' | 'corporate' | 'individual' | 'shell' | 'unknown'
  riskLevel: 'clean' | 'suspicious' | 'fraud'
  country: string
  isSelected: boolean
  isFocused: boolean  // currently inspected in case file
}
```

**Edge types:**
```typescript
type NetworkEdge = {
  source: string
  target: string
  amount: number
  timestamp: string
  riskLevel: 'clean' | 'suspicious' | 'fraud'
  messageType: 'pacs.008' | 'pacs.009' | 'pacs.004'
}
```

**Visual encoding:**
- Node size: scaled by transaction volume (log scale, min 6px max 22px)
- Node color: risk level (green/amber/red per palette above)
- Node shape: type — circle = individual, square = corporate, triangle = shell/offshore
- Edge color: risk level of that transaction
- Edge thickness: transaction amount (log scaled)
- Edge animation: fraud edges pulse/dash, clean edges are solid
- Selected node: glowing ring (CSS box-shadow on canvas layer via custom renderer)

**Interactions:**
- Click node → updates Case File panel with that node's primary transaction
- Hover node → shows tooltip with entity name, country, total volume
- Drag nodes → repositions within force simulation
- Scroll to zoom
- Double-click canvas → re-centers and re-fits graph

**Key implementation pattern:**
```tsx
import ForceGraph2D from 'react-force-graph-2d'

<ForceGraph2D
  graphData={{ nodes, links }}
  nodeCanvasObject={(node, ctx, globalScale) => {
    // custom renderer — draw circle with risk color + glow for selected
    drawNode(node, ctx, globalScale, selectedNodeId)
  }}
  linkCanvasObject={(link, ctx) => {
    // custom renderer — draw animated dashed line for fraud edges
    drawEdge(link, ctx)
  }}
  onNodeClick={(node) => setFocusedTransaction(node.id)}
  cooldownTicks={100}
  d3AlphaDecay={0.02}
  d3VelocityDecay={0.3}
  backgroundColor="transparent"
/>
```

**Graph data generation:**
- `POST /api/graph` on backend: accepts current transaction parameters, returns a synthetic network of 8-15 connected entities with realistic risk distribution matching the selected archetype
- The network changes meaningfully when you switch archetypes (BEC = small tight cluster with one anomalous edge; Money Mule = linear chain; Corporate Payroll = hub-and-spoke clean network)

---

### Feature 4 — Case File (Right Panel)

**What it is:** The investigation result, presented as a structured document — not a results page. Formatted like a forensic brief. Readable by a non-technical person, precise enough for an analyst.

**Sections (rendered top to bottom with Motion stagger animation):**

#### 4a. Verdict Header
```
┌──────────────────────────────────────┐
│  ⬟ FRAUD DETECTED                    │
│  Confidence: 94.2%                   │
│  Transaction #TX-8821                │
│  Flagged at 14:23:07 UTC             │
└──────────────────────────────────────┘
```
- Large risk badge with animated border (pulsing for fraud, static for clean)
- Score rendered as an arc gauge (Recharts RadialBarChart)
- Score animates from 0 to final value on mount (Motion useMotionValue + animate)

#### 4b. Risk DNA Strip
```
┌──────────────────────────────────────┐
│  RISK SIGNATURE                      │
│  ████░░██░░░░███░░░░░░░░████████     │
│  Amount  Time  Vel  Country  Pattern │
└──────────────────────────────────────┘
```
- A horizontal barcode-style strip, one segment per feature
- Segment height = feature's contribution to the fraud score (SHAP value)
- Segment color = direction (red = pushes toward fraud, blue = pulls away)
- Hover any segment → tooltip showing feature name, value, contribution %
- This is the unique visual fingerprint of a transaction — no two transactions look the same
- Implemented as a custom SVG component, ~80 lines

#### 4c. Findings (Explainability Panel)
```
┌──────────────────────────────────────┐
│  FINDINGS                            │
│                                      │
│  [!] CRITICAL  Transaction velocity  │
│      8 transfers in 6 hours          │
│      Contribution: +31% risk         │
│                                      │
│  [!] HIGH      New account           │
│      Account age: 3 days             │
│      Contribution: +22% risk         │
│                                      │
│  [✓] LOW       Transaction amount    │
│      $45,000 — within normal range   │
│      Contribution: -4% risk          │
└──────────────────────────────────────┘
```
- Each finding is a card with severity badge, plain-English description, and contribution bar
- Cards animate in with stagger (Motion `staggerChildren`)
- Severity: CRITICAL (red), HIGH (amber), MEDIUM (yellow), LOW (green)
- Data comes from `POST /api/explain` — backend returns pre-computed feature contributions

#### 4d. Counterfactual Explorer

This is the most technically original feature. After scoring, a collapsible section shows: **"What would make this transaction safe?"**

```
┌──────────────────────────────────────┐
│  COUNTERFACTUAL ANALYSIS             │
│  Minimum changes needed to lower     │
│  risk below threshold (50%)          │
│                                      │
│  Account age:     3 days → 45 days   │
│  Transaction vel: 8 → 2 per hour     │
│  Hour of day:     03:00 → 14:00      │
│                                      │
│  [Try this scenario →]               │
└──────────────────────────────────────┘
```

**Implementation (backend):**
```python
# POST /api/counterfactual
# Uses simple perturbation approach — no external library needed
def find_counterfactual(transaction_params, target_score=0.40):
    """
    Iteratively perturb features in order of their SHAP contribution
    (largest contributors first) until score drops below target.
    Return the minimal set of changes.
    """
    current = transaction_params.copy()
    changes = {}
    
    for feature in sorted_by_shap_contribution(transaction_params):
        if predict_score(current) < target_score:
            break
        # Try the "safe" direction for this feature
        safe_value = get_safe_value(feature, current[feature])
        current[feature] = safe_value
        changes[feature] = (transaction_params[feature], safe_value)
    
    return changes
```

The "Try this scenario" button loads the counterfactual parameters into the Scenario Composer — closing the loop between insight and exploration.

---

### Feature 5 — Analyst Dashboard (`/analyst`)

**What it is:** A second view that looks like a fraud analyst's workstation. A table of synthetic transactions that refreshes periodically, sortable, filterable, with a slide-out case file when you click a row.

**Components:**
- shadcn `DataTable` with TanStack Table v8 underneath
- `Sheet` component for the slide-out case file panel
- `Badge` for risk level in each row
- `Progress` bar for fraud score column
- Filter bar: risk level filter, date range, amount range, country pair
- Sort by: amount, score, timestamp, account age
- "Flagged only" toggle

**Real-time feel:**
```typescript
// New transaction appears every 3s in demo mode
const { data } = useQuery({
  queryKey: ['transactions'],
  queryFn: fetchRecentTransactions,
  refetchInterval: 3000,
})

// New rows animate in from top using Motion AnimatePresence
<AnimatePresence>
  {transactions.map(tx => (
    <motion.tr
      key={tx.id}
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
    />
  ))}
</AnimatePresence>
```

---

### Feature 6 — Risk DNA Component (Standalone / Reusable)

This component is important enough to spec separately because it appears in multiple places (Case File, Analyst Dashboard table rows, LinkedIn screenshot).

```typescript
type RiskDNAProps = {
  features: {
    name: string
    value: number      // actual feature value (normalized 0-1)
    contribution: number  // SHAP value (-1 to +1)
  }[]
  width?: number
  height?: number
  showLabels?: boolean
  animated?: boolean
}
```

**Rendering logic:**
```typescript
// Each segment:
// - width: equal (total_width / n_features)
// - height: Math.abs(contribution) * max_height  (min 4px)
// - color: contribution > 0 ? fraudColor : safeColor
// - opacity: 0.3 + (0.7 * Math.abs(contribution))  (dim low-contrib features)
// Baseline is vertically centered — positive contributions go up, negative go down

const RiskDNA: React.FC<RiskDNAProps> = ({ features, width = 240, height = 48 }) => {
  const segmentWidth = width / features.length
  const maxH = height / 2
  
  return (
    <svg width={width} height={height}>
      {features.map((f, i) => {
        const barH = Math.max(4, Math.abs(f.contribution) * maxH)
        const y = f.contribution > 0 
          ? maxH - barH 
          : maxH
        const color = f.contribution > 0 ? '#EF4444' : '#22C55E'
        return (
          <rect
            key={f.name}
            x={i * segmentWidth + 1}
            y={y}
            width={segmentWidth - 2}
            height={barH}
            fill={color}
            opacity={0.3 + 0.7 * Math.abs(f.contribution)}
            rx={1}
          />
        )
      })}
      <line x1={0} y1={maxH} x2={width} y2={maxH} stroke="#2A2A3A" strokeWidth={1}/>
    </svg>
  )
}
```

---

## 7. Backend API Changes

All new endpoints. Nothing existing is modified.

### `POST /api/explain`
```
Input:  { transaction: TransactionParams }
Output: {
  score: float,
  verdict: 'clean' | 'suspicious' | 'fraud',
  features: [
    {
      name: string,
      humanLabel: string,
      value: any,
      normalizedValue: float,   // 0-1
      contribution: float,       // signed, sum ≈ score
      severity: 'critical' | 'high' | 'medium' | 'low'
    }
  ]
}
```

Implement using permutation importance if SHAP is too heavy. Approximate SHAP values by computing `predict(x) - predict(x_with_feature_i_zeroed)` for each feature. Fast, no library needed.

### `POST /api/graph`
```
Input:  { transaction: TransactionParams, archetype: string }
Output: {
  nodes: NetworkNode[],
  edges: NetworkEdge[]
}
```

Generate deterministically from the archetype + a transaction hash (so the same params always produce the same graph). Keep it synthetic — 8-15 nodes is enough for a clear visual.

### `POST /api/counterfactual`
```
Input:  { transaction: TransactionParams, targetScore: float }
Output: {
  originalScore: float,
  achievedScore: float,
  changes: { [feature]: { from: any, to: any, impact: float } }
}
```

### `GET /api/scenarios`
```
Output: FraudArchetype[]  (the 6 archetypes with their parameter defaults)
```

---

## 8. Implementation Phases

Strict order. Each phase is independently deployable.

---

### Phase 1 — Foundation (Days 1-2)
**Goal:** Next.js app running with correct stack, talking to Flask backend.

Tasks:
- [ ] `npx create-next-app@latest fraud-ui --typescript --tailwind --app`
- [ ] Install shadcn CLI: `npx shadcn@latest init`
- [ ] Install components: `npx shadcn@latest add card badge button slider select switch sheet data-table progress`
- [ ] Install Motion: `npm install motion`
- [ ] Install Zustand: `npm install zustand`
- [ ] Install TanStack Query: `npm install @tanstack/react-query`
- [ ] Install react-force-graph: `npm install react-force-graph-2d`
- [ ] Install lucide-react: already included with shadcn
- [ ] Set up Geist font in `layout.tsx`
- [ ] Configure Tailwind with custom color tokens (exact hex values from Section 4)
- [ ] Set up global store with Zustand: `store/investigation.ts`
- [ ] Add Flask CORS: `pip install flask-cors` + `CORS(app, origins=["http://localhost:3000"])`
- [ ] Verify `POST /api/predict` returns valid response from Next.js dev server

**Deliverable:** Next.js app at `localhost:3000`, can score a hardcoded transaction, result appears in browser console.

---

### Phase 2 — Case File + Risk DNA (Days 3-4)
**Goal:** The right panel is fully functional and visually complete.

Tasks:
- [ ] Implement `POST /api/explain` on Flask (approximate SHAP via permutation)
- [ ] Build `<RiskDNA />` component (spec in Section 6)
- [ ] Build `<VerdictHeader />` — score arc gauge using Recharts RadialBarChart, animated with Motion
- [ ] Build `<FindingsPanel />` — feature cards with severity badges, staggered Motion entry
- [ ] Build `<CaseFilePanel />` — assembles Verdict + RiskDNA + Findings
- [ ] Wire to Zustand store: `selectedTransaction → CaseFilePanel` re-renders
- [ ] Dark theme: panel background `#111118`, border `#2A2A3A`
- [ ] Test: change transaction params manually in store → panel updates correctly

**Deliverable:** Right panel showing a live, visually complete case file for a hardcoded transaction.

---

### Phase 3 — Scenario Composer (Days 5-6)
**Goal:** Left panel is fully interactive. Changing archetype or parameters re-scores in real time.

Tasks:
- [ ] Implement `GET /api/scenarios` on Flask (return hardcoded archetype list)
- [ ] Build `<ArchetypeGrid />` — 6 cards, click selects archetype, loads parameters into store
- [ ] Build `<ParameterControls />` — sliders, selects, toggles for each parameter
- [ ] Hook up debounced scoring: parameter change → 300ms debounce → `POST /api/explain` → update store
- [ ] Animate archetype switch: old parameters fade out, new fade in (Motion `AnimatePresence`)
- [ ] Each parameter shows its current contribution tag (requires `/api/explain` response)

**Deliverable:** Left panel fully interactive. Select "BEC" archetype → parameters load → score updates → case file updates.

---

### Phase 4 — Relationship Map (Days 7-9)
**Goal:** Center panel shows force-directed graph. Clicking nodes updates case file.

Tasks:
- [ ] Implement `POST /api/graph` on Flask (6 synthetic graph generators, one per archetype)
- [ ] Build `<RelationshipMap />` wrapper around `ForceGraph2D`
- [ ] Implement custom `nodeCanvasObject` renderer (risk color, glow on selected)
- [ ] Implement custom `linkCanvasObject` renderer (dashed animation for fraud edges)
- [ ] Handle `onNodeClick` → update `focusedNode` in Zustand store
- [ ] Handle archetype change → fetch new graph → re-render (with transition: fade out old, animate in new)
- [ ] Graph container: `background: transparent`, border `1px solid #2A2A3A`, border-radius 12px
- [ ] Add zoom controls (+ / - buttons, re-center button) in top-right corner of panel
- [ ] Tooltip on hover: entity name, country flag emoji, total transaction volume

**Deliverable:** Three-panel Investigation Room fully functional. This is the hero feature.

---

### Phase 5 — Counterfactual Explorer (Days 10-11)
**Goal:** Case file shows "what would make this safe?" section with working "Try this" button.

Tasks:
- [ ] Implement `POST /api/counterfactual` on Flask
- [ ] Build `<CounterfactualPanel />` — collapsible section in Case File
- [ ] Each change shown as: `feature name: old value → new value [impact badge]`
- [ ] "Try this scenario" button → loads counterfactual parameters into Scenario Composer
- [ ] Animate the parameter change in Scenario Composer (highlight changed sliders briefly)

**Deliverable:** Full feedback loop: score → explanation → counterfactual → try it → new score.

---

### Phase 6 — Analyst Dashboard (Days 12-13)
**Goal:** `/analyst` route with full-page transaction table and slide-out case file.

Tasks:
- [ ] Build `GET /api/transactions` Flask endpoint (returns 20 synthetic transactions, weighted toward 3-5 fraud)
- [ ] Set up TanStack Table with shadcn DataTable
- [ ] Columns: timestamp, amount, sender country, receiver country, message type, risk DNA strip, score badge, status
- [ ] Implement filter bar (risk level, date range)
- [ ] Implement `Sheet` slide-out: click row → slide-out panel shows full `<CaseFilePanel />`
- [ ] Implement refetch every 3 seconds + `AnimatePresence` for new row animation
- [ ] "Flagged only" toggle using Zustand

**Deliverable:** `/analyst` route, fully functional dashboard view.

---

### Phase 7 — Polish + Deploy (Days 14-15)
**Goal:** Production-ready, deployed, LinkedIn-ready.

Tasks:
- [ ] Responsive layout: mobile breakpoints for all panels (stacked on < 768px)
- [ ] Loading skeletons: all panels show Skeleton (shadcn) while data loads
- [ ] Error states: if API fails, show graceful error message not blank panel
- [ ] Page transitions: route changes use Motion `AnimatePresence`
- [ ] Navigation: minimal top bar with `Fraud · Case Room` wordmark and links to `/analyst`, `/network`
- [ ] SEO: `metadata` in `layout.tsx` with correct title/description/og:image
- [ ] Deploy frontend to Vercel
- [ ] Update Flask CORS to allow Vercel production URL
- [ ] Smoke test entire flow on deployed URL
- [ ] Record a 30-second screen capture for LinkedIn

---

## 9. File Structure (Target)

```
fraud-ui/
├── app/
│   ├── layout.tsx              # Root layout, Geist font, TanStack Query Provider
│   ├── page.tsx                # Investigation Room (/)
│   ├── analyst/
│   │   └── page.tsx            # Analyst Dashboard
│   └── globals.css             # Tailwind base + custom color tokens
│
├── components/
│   ├── investigation/
│   │   ├── InvestigationRoom.tsx      # Three-panel layout
│   │   ├── ScenarioComposer.tsx       # Left panel
│   │   ├── RelationshipMap.tsx        # Center panel
│   │   └── CaseFilePanel.tsx          # Right panel
│   │
│   ├── case-file/
│   │   ├── VerdictHeader.tsx          # Score arc + verdict badge
│   │   ├── RiskDNA.tsx                # Barcode signature component
│   │   ├── FindingsPanel.tsx          # Feature contribution cards
│   │   └── CounterfactualPanel.tsx    # "What would make this safe?"
│   │
│   ├── graph/
│   │   └── NetworkGraph.tsx           # ForceGraph2D wrapper
│   │
│   ├── analyst/
│   │   ├── TransactionTable.tsx       # TanStack Table
│   │   └── TransactionRow.tsx         # Row with inline RiskDNA
│   │
│   └── ui/                            # shadcn components (auto-generated)
│
├── store/
│   └── investigation.ts               # Zustand store
│
├── lib/
│   ├── api.ts                         # All API calls (typed)
│   ├── archetypes.ts                  # Archetype definitions
│   └── utils.ts                       # shadcn utils + custom helpers
│
└── types/
    └── index.ts                       # All shared TypeScript types
```

---

## 10. Zustand Store Shape

```typescript
// store/investigation.ts

type InvestigationStore = {
  // Current transaction being investigated
  parameters: TransactionParameters
  setParameters: (params: Partial<TransactionParameters>) => void

  // Selected archetype
  archetypeId: string
  setArchetype: (id: string) => void

  // Scoring result
  scoreResult: ScoreResult | null
  setScoreResult: (result: ScoreResult) => void

  // Currently focused node in graph
  focusedNodeId: string | null
  setFocusedNode: (id: string | null) => void

  // Analyst dashboard
  showFlaggedOnly: boolean
  toggleFlaggedOnly: () => void
}
```

---

## 11. Key Cursor Prompting Notes

When using Cursor to implement this, structure your prompts like this:

**For new components:**
> "Implement `<ComponentName />` exactly as specified in the PRD Section X. Use the color tokens from Section 4. All animations use Motion (from 'motion/react'). TypeScript strict mode. No any types."

**For backend endpoints:**
> "Implement the `POST /api/[endpoint]` endpoint in `app.py` exactly as specified in PRD Section 7. Input/output types are defined there. Use the existing model at `models/fraud_detector.pkl`."

**For the graph:**
> "Implement `<NetworkGraph />` using `react-force-graph-2d`. Use a custom `nodeCanvasObject` renderer. Do NOT use the default node rendering. The node colors come from the riskLevel field. Spec is in PRD Section 6, Feature 3."

**Golden rule:** Give Cursor one component at a time. Never ask it to implement an entire panel in one prompt. Build bottom-up: primitive components first, then composites.

---

## 12. Design Inspiration Notes

### Aesthetic reference points
- **Vercel's dashboard** — sparse, dark, precise typography, no decoration
- **Linear's issue tracker** — the "case management" interaction model
- **Stripe Radar** — how fraud findings are presented to analysts (clinical, not alarming)
- **Dune Analytics** — dark data visualization, colored signal against dark background

### What to avoid
- Glassmorphism (overdone, distracting)
- Gradient backgrounds (looks cheap in 2025)
- Unnecessary 3D effects
- Modal-heavy interactions (everything should be inline or slide-out)
- Any animation that doesn't communicate meaning

### Typography rule
Numbers (scores, amounts, timestamps) are always `font-mono` (Geist Mono). Everything else is `font-sans` (Geist). This single rule makes the UI feel immediately more professional.

---

## 13. LinkedIn Post Strategy (Post-Deploy)

Once deployed, the post structure that works:

1. **Opening line:** One specific, concrete observation (not "I built a fraud detection system")
   > "Most fraud tools make you fill in a form. This one starts with a case already open."

2. **Screen recording:** 30 seconds. Show: archetype selection → graph animates → case file updates → counterfactual. No narration needed.

3. **Technical callout:** One genuinely interesting thing (the counterfactual explorer or the Risk DNA). Link to GitHub + live demo.

4. **Tags:** #fintech #machinelearning #iso20022 #payments #react — in that order.

Post on a Tuesday or Wednesday morning. Do not post on Monday.

---

*End of PRD. Start with Phase 1.*
