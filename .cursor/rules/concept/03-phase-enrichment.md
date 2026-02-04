# Phase 3: Enrichment - Analyse Complète Holders + Calcul Scores

> Analyser TOUS les holders d'un market et calculer ROI + confiance

---

## Objectif

Pour chaque event de la watchlist, récupérer TOUS les holders (via Bitquery, pas limité à 20) et calculer un score de fiabilité global.

**Input**: Events watchlist (Phase 2)
**Output**: Scores agrégés (ROI moyen, confiance moyenne) par event
**Fréquence**: Batch 15 min (cache)
**Coût**: 149 USD/mois (Bitquery Startup)

---

## Flux de Données

```
Events watchlist (50-100 events)
         │
         ▼
Bitquery GraphQL → TOUS holders YES
Bitquery GraphQL → TOUS holders NO
         │
         ├─ Croiser avec DB traders (connus vs inconnus)
         │  ├─ Traders connus: utiliser ROI/confiance en DB
         │  └─ Traders inconnus: fetch historique via Bitquery
         │
         ▼
Calculer scores agrégés par event:
├─ ROI moyen YES (pondéré par taille position)
├─ ROI moyen NO (pondéré par taille position)
├─ Confiance moyenne YES
└─ Confiance moyenne NO
         │
         ▼
DB Table: event_scores (cache 15 min)
```

---

## Table: `event_scores`

```sql
CREATE TABLE event_scores (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  event_id UUID REFERENCES events(id) ON DELETE CASCADE,
  
  -- Scores YES side
  roi_avg_yes DECIMAL(10,2),
  confidence_avg_yes DECIMAL(5,4),
  holders_count_yes INTEGER,
  holders_known_yes INTEGER,  -- Déjà en DB
  
  -- Scores NO side
  roi_avg_no DECIMAL(10,2),
  confidence_avg_no DECIMAL(5,4),
  holders_count_no INTEGER,
  holders_known_no INTEGER,
  
  -- Metadata
  calculated_at TIMESTAMPTZ DEFAULT NOW(),
  cache_ttl INTEGER DEFAULT 900,  -- 15 min en secondes
  
  -- Indexes
  INDEX idx_event_scores_event (event_id),
  INDEX idx_event_scores_calculated (calculated_at)
);
```

---

## Calcul Score Confiance

```python
def calculate_confidence_score(trader):
    roi = trader["roi"]
    win_rate = trader["win_rate"]
    trades_count = trader["trades_count"]
    volume = trader["total_volume"]
    
    # Pénaliser faible volume (survivorship bias)
    if trades_count < 50:
        volume_penalty = trades_count / 50.0
    else:
        volume_penalty = 1.0
    
    # Score base
    base_score = (roi / 100) * win_rate
    
    # Appliquer pénalité
    confidence = base_score * volume_penalty
    
    return min(max(confidence, 0), 1.0)
```

---

## Worker: Enrichment Batch

```typescript
// Pseudo-code simplifié
for (const event of watchlist) {
  // Fetch YES holders via Bitquery
  const yes_holders = await bitquery.getAllHolders(event.yes_token)
  
  // Fetch NO holders
  const no_holders = await bitquery.getAllHolders(event.no_token)
  
  // Calculer scores YES
  let roi_sum_yes = 0, confidence_sum_yes = 0
  for (const holder of yes_holders) {
    const trader = await getOrFetchTrader(holder.address)
    const confidence = calculateConfidence(trader)
    
    roi_sum_yes += trader.roi * holder.position_size
    confidence_sum_yes += confidence * holder.position_size
  }
  
  // Scores pondérés
  const roi_avg_yes = roi_sum_yes / total_volume_yes
  const confidence_avg_yes = confidence_sum_yes / total_volume_yes
  
  // Idem pour NO...
  
  // Store scores
  await upsertEventScore(event.id, {
    roi_avg_yes,
    confidence_avg_yes,
    roi_avg_no,
    confidence_avg_no
  })
}
```

---

## Prochaines Étapes

→ **Phase 4: Buy Signal** ([04-phase-buy-signal.md](04-phase-buy-signal.md))

Utiliser scores pour générer signaux d'achat.

---

**Status**: ✅ Spécifications complètes
