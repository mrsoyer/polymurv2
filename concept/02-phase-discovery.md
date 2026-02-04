# Phase 2: Discovery - Identification Events Pertinents

> Surveiller les positions des top traders pour découvrir les markets à analyser

---

## Objectif

Identifier automatiquement les events (markets) sur lesquels les traders de confiance se positionnent, afin de constituer une watchlist dynamique.

**Input**: Traders suivis (Phase 1) + leurs positions actives
**Output**: Liste events à enrichir (Phase 3)
**Fréquence**: Temps réel (WebSocket) + Batch quotidien
**Coût**: Gratuit

---

## Stratégie: Polling Data API

⚠️ **Note** : Le WebSocket CLOB public ne fournit PAS les adresses maker/taker dans les events OrderFilled. On doit utiliser le polling via Data API.

### Polling Positions Traders (Recommandé)

```
Cron 1×/jour
    │
    ▼
Data API /positions
    │
    ├─ Pour chaque trader suivi
    │  └─ Récupérer positions actives
    │     └─ Extraire market_ids
    │        └─ Upsert dans events table
    │
    └─ Résultat: Liste complète events
```

**Avantage**: Coverage complète, pas de limite
**Inconvénient**: Latence 1 jour (acceptable)

---

## Spécifications Techniques

### Table: `events`

```sql
CREATE TABLE events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  polymarket_id VARCHAR(100) UNIQUE NOT NULL,

  -- Metadata
  title TEXT NOT NULL,
  category VARCHAR(50),  -- crypto, politics, sports, etc.
  status VARCHAR(20) DEFAULT 'active',  -- active, resolved, cancelled

  -- Timing
  created_at TIMESTAMPTZ,
  resolution_date TIMESTAMPTZ,
  discovered_at TIMESTAMPTZ DEFAULT NOW(),

  -- Metrics
  volume_total DECIMAL(15,2) DEFAULT 0,
  holders_count INTEGER DEFAULT 0,
  top_traders_count INTEGER DEFAULT 0,  -- Combien de nos traders suivis

  -- Indexes
  INDEX idx_events_status (status),
  INDEX idx_events_category (category),
  INDEX idx_events_top_traders (top_traders_count DESC)
);
```

---

### Table de Liaison: `trader_events`

```sql
CREATE TABLE trader_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  trader_id UUID REFERENCES traders(id) ON DELETE CASCADE,
  event_id UUID REFERENCES events(id) ON DELETE CASCADE,

  -- Position info
  side VARCHAR(3),  -- YES, NO, ou NULL si pas encore connu
  first_position_at TIMESTAMPTZ DEFAULT NOW(),

  -- Top holder flags (mis à jour Phase 3)
  is_top_holder_yes BOOLEAN DEFAULT FALSE,
  is_top_holder_no BOOLEAN DEFAULT FALSE,

  -- Unique constraint
  UNIQUE(trader_id, event_id),

  -- Indexes
  INDEX idx_trader_events_trader (trader_id),
  INDEX idx_trader_events_event (event_id)
);
```

---

## Worker: Monitoring Temps Réel

### Implementation WebSocket

```typescript
// supabase/functions/monitor-trader-positions/index.ts

import { WebSocket } from "https://deno.land/x/websocket@v0.1.4/mod.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/"

async function monitorPositions() {
  const supabase = createClient(
    Deno.env.get('SUPABASE_URL')!,
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
  )

  // Fetch traders à surveiller
  const { data: traders } = await supabase
    .from('traders')
    .select('address')
    .eq('is_seeded', true)

  const trackedAddresses = new Set(traders.map(t => t.address.toLowerCase()))

  // Connect WebSocket
  const ws = new WebSocket(CLOB_WS_URL)

  ws.onopen = () => {
    console.log('Connected to CLOB WebSocket')

    // Subscribe market channel (public)
    ws.send(JSON.stringify({
      type: "subscribe",
      channel: "market",
      markets: ["*"]  // All markets (or liste spécifique)
    }))
  }

  ws.onmessage = async (event) => {
    const data = JSON.parse(event.data)

    // Event: Trade
    if (data.type === "trade") {
      const maker = data.data.maker?.toLowerCase()
      const taker = data.data.taker?.toLowerCase()

      // Check si un top trader a tradé
      if (trackedAddresses.has(maker) || trackedAddresses.has(taker)) {
        const trader_address = trackedAddresses.has(maker) ? maker : taker
        const market_id = data.data.market_id

        // Enregistrer event découvert
        await discoverEvent(supabase, market_id, trader_address)
      }
    }
  }

  ws.onerror = (error) => {
    console.error('WebSocket error:', error)
  }

  ws.onclose = () => {
    console.log('Connection closed, reconnecting in 5s...')
    setTimeout(() => monitorPositions(), 5000)
  }
}

async function discoverEvent(supabase, market_id, trader_address) {
  // 1. Fetch market details from Data API
  const marketResponse = await fetch(
    `https://data-api.polymarket.com/markets/${market_id}`
  )
  const market = await marketResponse.json()

  // 2. Upsert event
  const { data: event } = await supabase
    .from('events')
    .upsert({
      polymarket_id: market_id,
      title: market.title,
      category: market.category,
      status: market.status,
      resolution_date: market.resolution_date,
      volume_total: market.volume
    }, {
      onConflict: 'polymarket_id'
    })
    .select()
    .single()

  // 3. Link trader <-> event
  const { data: trader } = await supabase
    .from('traders')
    .select('id')
    .eq('address', trader_address)
    .single()

  await supabase
    .from('trader_events')
    .upsert({
      trader_id: trader.id,
      event_id: event.id
    }, {
      onConflict: 'trader_id,event_id'
    })

  // 4. Increment top_traders_count
  await supabase.rpc('increment_top_traders_count', {
    event_id_param: event.id
  })
}

// Launch monitoring
monitorPositions()
```

---

### RPC Function: `increment_top_traders_count`

```sql
CREATE OR REPLACE FUNCTION increment_top_traders_count(event_id_param UUID)
RETURNS VOID AS $$
BEGIN
  UPDATE events
  SET top_traders_count = (
    SELECT COUNT(DISTINCT trader_id)
    FROM trader_events
    WHERE event_id = event_id_param
  )
  WHERE id = event_id_param;
END;
$$ LANGUAGE plpgsql;
```

---

## Worker: Polling Batch Quotidien

### Implementation Batch

```typescript
// supabase/functions/discover-events-batch/index.ts

import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

serve(async (req) => {
  const supabase = createClient(
    Deno.env.get('SUPABASE_URL')!,
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
  )

  // Fetch tous les traders suivis
  const { data: traders } = await supabase
    .from('traders')
    .select('id, address')

  let discovered_events = 0

  for (const trader of traders) {
    // Fetch positions actives
    const response = await fetch(
      `https://data-api.polymarket.com/positions?user=${trader.address}&status=active`
    )
    const data = await response.json()
    const positions = data.positions

    for (const position of positions) {
      const market_id = position.market_id

      // Fetch market details
      const marketResp = await fetch(
        `https://data-api.polymarket.com/markets/${market_id}`
      )
      const market = await marketResp.json()

      // Upsert event
      const { data: event } = await supabase
        .from('events')
        .upsert({
          polymarket_id: market_id,
          title: market.title,
          category: market.category,
          status: 'active',
          resolution_date: market.resolution_date,
          volume_total: market.volume
        }, {
          onConflict: 'polymarket_id'
        })
        .select()
        .single()

      // Link trader <-> event
      await supabase
        .from('trader_events')
        .upsert({
          trader_id: trader.id,
          event_id: event.id,
          side: position.side  // YES or NO
        }, {
          onConflict: 'trader_id,event_id'
        })

      discovered_events++
    }

    // Rate limiting: pause 100ms entre traders
    await new Promise(resolve => setTimeout(resolve, 100))
  }

  // Update top_traders_count pour tous events
  await supabase.rpc('refresh_all_top_traders_counts')

  return new Response(JSON.stringify({
    success: true,
    discovered_events,
    timestamp: new Date().toISOString()
  }), {
    headers: { 'Content-Type': 'application/json' }
  })
})
```

---

### Schedule Cron Batch

```sql
SELECT cron.schedule(
  'discover-events-batch-daily',
  '0 1 * * *',  -- Every day at 01:00 UTC (après seeding)
  $$
    SELECT net.http_post(
      url := 'https://[project-id].supabase.co/functions/v1/discover-events-batch',
      headers := '{"Authorization": "Bearer [anon-key]"}',
      body := '{}'
    );
  $$
);
```

---

## Filtres de Qualité Events

Ne pas surveiller un event si:

```typescript
function isValidEvent(event: Event): boolean {
  // Volume minimum 10k USD (exclure petits markets)
  if (event.volume_total < 10000) {
    return false
  }

  // Minimum 3 top traders positionnés (sinon pas de signal)
  if (event.top_traders_count < 3) {
    return false
  }

  // Status actif uniquement
  if (event.status !== 'active') {
    return false
  }

  // Pas trop proche de la résolution (< 24h)
  const hoursUntilResolution = (new Date(event.resolution_date) - Date.now()) / 3600000
  if (hoursUntilResolution < 24) {
    return false  // Trop tard pour entrer
  }

  return true
}
```

---

## Prioritisation Watchlist

Events à surveiller en priorité:

```sql
SELECT
  e.id,
  e.title,
  e.category,
  e.top_traders_count,
  e.volume_total,
  (e.resolution_date - NOW()) as time_until_resolution
FROM events e
WHERE
  e.status = 'active'
  AND e.top_traders_count >= 3
  AND e.volume_total >= 10000
  AND e.resolution_date > NOW() + INTERVAL '24 hours'
ORDER BY
  e.top_traders_count DESC,  -- Plus de top traders = priorité
  e.volume_total DESC         -- Puis volume
LIMIT 100;  -- Top 100 events à enrichir (Phase 3)
```

---

## Métriques de Succès

### KPIs Phase 2

| Métrique | Objectif | Mesure |
|----------|----------|--------|
| **Events découverts/jour** | 20-50 | COUNT(*) WHERE discovered_at > NOW() - INTERVAL '1 day' |
| **Events actifs watchlist** | 50-100 | COUNT(*) WHERE status = 'active' AND top_traders_count >= 3 |
| **Top traders moyen/event** | > 5 | AVG(top_traders_count) |
| **Volume moyen/event** | > 50k USD | AVG(volume_total) |
| **Latence détection** | < 5 min | timestamp_detected - timestamp_trade |

---

## Prochaines Étapes

Une fois watchlist constituée (50+ events actifs):

→ **Phase 3: Enrichment** ([03-phase-enrichment.md](03-phase-enrichment.md))

Analyser TOUS les holders de chaque event pour calculer scores ROI + confiance.

---

## Ressources

- [WebSocket API](../docapi/polymarket/websocket-api.md) - CLOB WebSocket
- [Data API](../docapi/polymarket/data-api.md) - Endpoint `/positions`
- [Worker Discovery](../backend/workers/worker-discovery.md) - Implémentation
- [Schema DB](../backend/database/schema.md) - Tables `events` et `trader_events`

---

**Version**: 1.0
**Date**: 2026-02-04
**Auteur**: SYM Framework - Orchestrator v4
**Status**: ✅ Spécifications complètes
