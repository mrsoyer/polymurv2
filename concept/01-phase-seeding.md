# Phase 1: Seeding - Construction Base Traders Fiables

> Initialisation de la base de données avec les meilleurs traders par secteur

---

## Objectif

Construire la base initiale de "traders de confiance" qui servira de noyau pour découvrir les events pertinents.

**Input**: Leaderboard API Polymarket
**Output**: Base de données traders avec métadonnées
**Fréquence**: 1×/jour (refresh quotidien)
**Coût**: Gratuit

---

## Flux de Données

```
Polymarket Data API /leaderboard
         │
         ├─ Secteur: crypto (top 50)
         ├─ Secteur: politics (top 50)
         ├─ Secteur: sports (top 50)
         ├─ Secteur: economics (top 50)
         └─ Secteur: all (top 100)
         │
         ▼
   DB Table: traders
   (250-300 traders initiaux)
```

---

## Spécifications Techniques

### API Call

```python
import requests

def seed_top_traders(category, limit=50):
    """
    Récupère et stocke les top traders d'une catégorie
    """
    url = "https://data-api.polymarket.com/leaderboard"

    params = {
        "metric": "roi",          # Trier par ROI
        "category": category,     # crypto, politics, sports, economics, all
        "period": "30d",          # Historique 30 jours
        "limit": limit            # Nombre de traders
    }

    response = requests.get(url, params=params)
    leaderboard = response.json()["leaderboard"]

    return leaderboard

# Exemple réponse
[
  {
    "rank": 1,
    "address": "0x1234...",
    "display_name": "CryptoPro",
    "roi": 45.8,
    "volume": 125000.0,
    "win_rate": 0.68,
    "trades_count": 342,
    "category_affinity": "crypto"
  }
]
```

---

## Critères de Sélection

### Par Secteur

| Secteur | Top N | Justification |
|---------|-------|---------------|
| **crypto** | 50 | Secteur principal, haute volatilité |
| **politics** | 50 | Gros volumes, événements prévisibles |
| **sports** | 50 | Spécialistes reconnus |
| **economics** | 50 | Données macro importantes |
| **all** (overlap) | 100 | Traders généralistes performants |

**Total unique**: ~250-300 traders après dédoublonnage

---

### Filtres de Qualité

Ne pas seeder un trader si:

```python
def is_valid_trader(trader):
    """
    Filtres de qualité pour éviter faux positifs
    """
    # Minimum 50 trades (éviter survivorship bias)
    if trader["trades_count"] < 50:
        return False

    # Minimum 10k USD volume (exclure petits joueurs)
    if trader["volume"] < 10000:
        return False

    # Win rate raisonnable (éviter anomalies)
    if trader["win_rate"] < 0.5 or trader["win_rate"] > 0.95:
        return False  # Trop bas ou suspicieusement parfait

    # ROI positif minimum
    if trader["roi"] < 5.0:
        return False

    return True
```

**Résultat attendu**: ~80% des top 50 passent les filtres = ~200 traders valides seedés

---

## Structure Base de Données

### Table: `traders`

```sql
CREATE TABLE traders (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  address VARCHAR(42) UNIQUE NOT NULL,
  display_name VARCHAR(100),

  -- Metrics initiaux (leaderboard)
  roi_30d DECIMAL(10,2) NOT NULL,
  volume_total DECIMAL(15,2) NOT NULL,
  win_rate DECIMAL(5,4) NOT NULL,
  trades_count INTEGER NOT NULL,

  -- Catégorie affinity
  category_affinity VARCHAR(50),

  -- Tracking
  is_seeded BOOLEAN DEFAULT TRUE,
  first_seen_at TIMESTAMPTZ DEFAULT NOW(),
  last_updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes (créés séparément)
CREATE INDEX idx_traders_roi ON traders(roi_30d DESC);
CREATE INDEX idx_traders_category ON traders(category_affinity);
CREATE INDEX idx_traders_seeded ON traders(is_seeded);
```

---

## Implémentation Worker

### Worker: `seed_top_traders`

**Technologie**: Supabase Edge Function (Deno)
**Schedule**: Cron daily @ 00:00 UTC via pg_cron

```typescript
// supabase/functions/seed-top-traders/index.ts

import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const CATEGORIES = ['crypto', 'politics', 'sports', 'economics', 'all']

serve(async (req) => {
  const supabase = createClient(
    Deno.env.get('SUPABASE_URL')!,
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
  )

  let seeded_count = 0
  let skipped_count = 0

  for (const category of CATEGORIES) {
    // Fetch leaderboard
    const response = await fetch(
      `https://data-api.polymarket.com/leaderboard?metric=roi&category=${category}&period=30d&limit=50`
    )
    const data = await response.json()
    const leaderboard = data.leaderboard

    for (const trader of leaderboard) {
      // Apply filters
      if (!isValidTrader(trader)) {
        skipped_count++
        continue
      }

      // Upsert trader
      const { error } = await supabase
        .from('traders')
        .upsert({
          address: trader.address,
          display_name: trader.display_name,
          roi_30d: trader.roi,
          volume_total: trader.volume,
          win_rate: trader.win_rate,
          trades_count: trader.trades_count,
          category_affinity: category,
          is_seeded: true,
          last_updated_at: new Date().toISOString()
        }, {
          onConflict: 'address'  // Update if exists
        })

      if (!error) seeded_count++
    }
  }

  return new Response(JSON.stringify({
    success: true,
    seeded_count,
    skipped_count,
    timestamp: new Date().toISOString()
  }), {
    headers: { 'Content-Type': 'application/json' }
  })
})

function isValidTrader(trader: any): boolean {
  return (
    trader.trades_count >= 50 &&
    trader.volume >= 10000 &&
    trader.win_rate >= 0.5 && trader.win_rate <= 0.95 &&
    trader.roi >= 5.0
  )
}
```

---

## Schedule Cron

```sql
-- Supabase: pg_cron extension
SELECT cron.schedule(
  'seed-top-traders-daily',
  '0 0 * * *',  -- Every day at 00:00 UTC
  $$
    SELECT net.http_post(
      url := 'https://[project-id].supabase.co/functions/v1/seed-top-traders',
      headers := '{"Authorization": "Bearer [anon-key]", "Content-Type": "application/json"}',
      body := '{}'
    );
  $$
);
```

---

## Métriques de Succès

### KPIs Phase 1

| Métrique | Objectif | Mesure |
|----------|----------|--------|
| **Traders seedés/jour** | 200-300 | COUNT(*) WHERE is_seeded = TRUE |
| **Win rate moyen** | > 0.60 | AVG(win_rate) |
| **ROI moyen** | > 15% | AVG(roi_30d) |
| **Volume moyen** | > 50k USD | AVG(volume_total) |
| **Coverage secteurs** | 4+ secteurs | COUNT(DISTINCT category_affinity) |

### Dashboard Query

```sql
SELECT
  category_affinity,
  COUNT(*) as traders_count,
  AVG(roi_30d) as avg_roi,
  AVG(win_rate) as avg_win_rate,
  SUM(volume_total) as total_volume
FROM traders
WHERE is_seeded = TRUE
GROUP BY category_affinity
ORDER BY traders_count DESC;
```

---

## Gestion des Duplicates

### Problème

Un trader peut apparaître dans plusieurs catégories (ex: bon en crypto ET politics).

### Solution: Merge avec Priorité

```typescript
function resolveDuplicates(traders: Trader[]): Trader[] {
  const byAddress = new Map<string, Trader>()

  for (const trader of traders) {
    const existing = byAddress.get(trader.address)

    if (!existing) {
      byAddress.set(trader.address, trader)
      continue
    }

    // Keep trader with better ROI
    if (trader.roi > existing.roi) {
      byAddress.set(trader.address, {
        ...trader,
        // Merge categories
        category_affinity: `${existing.category_affinity},${trader.category_affinity}`
      })
    }
  }

  return Array.from(byAddress.values())
}
```

**Résultat**: Trader marqué avec multiple categories (ex: "crypto,politics")

---

## Évolution de la Base

### Croissance Organique

Au fil du temps, la base s'enrichit via Phase 2 (Discovery):

```
Jour 1   : 250 traders (seeded)
Jour 7   : 350 traders (+100 découverts via positions)
Jour 30  : 500 traders (+250 découverts)
Jour 90+ : 1000+ traders (maturité)
```

**Marqueur**: Trader découverts ont `is_seeded = FALSE` (différenciation)

---

## Refresh Strategy

### Daily Refresh

Chaque jour, mettre à jour les métriques des traders seedés:

```sql
UPDATE traders
SET
  roi_30d = (SELECT new_roi FROM leaderboard WHERE address = traders.address),
  win_rate = (SELECT new_win_rate FROM leaderboard WHERE address = traders.address),
  trades_count = (SELECT new_count FROM leaderboard WHERE address = traders.address),
  last_updated_at = NOW()
WHERE is_seeded = TRUE;
```

### Pruning

Retirer traders qui ne performent plus:

```sql
DELETE FROM traders
WHERE
  is_seeded = TRUE
  AND roi_30d < 5.0  -- ROI tombé sous seuil
  AND last_updated_at < NOW() - INTERVAL '30 days';  -- Pas vu depuis 30j
```

---

## Monitoring & Alertes

### Alertes à Configurer

| Condition | Action |
|-----------|--------|
| `seeded_count < 150` | Email admin: "Seeding failed" |
| `AVG(roi_30d) < 10` | Email admin: "Poor trader quality" |
| `Cron job failed` | Retry + alert |
| `API rate limit` | Backoff + retry |

---

## Prochaines Étapes

Une fois Phase 1 complétée (250+ traders seedés):

→ **Phase 2: Discovery** ([02-phase-discovery.md](02-phase-discovery.md))

Monitor les positions de ces traders pour identifier les events pertinents.

---

## Ressources

- [Data API Documentation](../docapi/polymarket/data-api.md) - Endpoint `/leaderboard`
- [Architecture Workers](../backend/workers/worker-seeding.md) - Implémentation complète
- [Schema DB](../backend/database/schema.md) - Table `traders` détaillée

---

**Version**: 1.0
**Date**: 2026-02-04
**Auteur**: SYM Framework - Orchestrator v4
**Status**: ✅ Spécifications complètes
