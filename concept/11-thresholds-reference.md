# Référence Centralisée des Seuils

> Tous les paramètres et seuils du système en un seul endroit

---

## Objectif

Centraliser TOUS les seuils configurables pour éviter les incohérences entre fichiers et faciliter l'ajustement des paramètres.

---

## Profils de Risque Standards

### Profil Conservateur

```yaml
name: "Conservative"
use_case: "Cold start (jours 1-30), risk-averse"

filters:
  sectors: ["all"]
  min_volume: 50000  # USD
  min_traders_count: 10
  min_holders_count: 400

signals:
  # Seuils absolus
  min_roi_absolute: 15.0  # %
  min_confidence_absolute: 0.8

  # Seuils différentiels YES vs NO
  min_roi_diff: 10.0  # %
  min_confidence_diff: 0.2

  # Exit
  stop_loss: -8.0  # %
  profit_target: 18.0  # %
  exit_mode: "hybrid"
  exit_threshold: 0.25  # 25% holders fiables sortent

risk:
  max_positions: 5
  max_position_size: 50.0  # USD
  max_total_exposure: 250.0
  max_positions_per_sector: 3
```

---

### Profil Équilibré

```yaml
name: "Balanced"
use_case: "Régime nominal (jour 30+), compromis risque/rendement"

filters:
  sectors: ["politics", "crypto"]
  min_volume: 30000
  min_traders_count: 7
  min_holders_count: 300

signals:
  # Seuils absolus
  min_roi_absolute: 10.0
  min_confidence_absolute: 0.7

  # Seuils différentiels
  min_roi_diff: 7.0
  min_confidence_diff: 0.15

  # Exit
  stop_loss: -10.0
  profit_target: 20.0
  exit_mode: "hybrid"
  exit_threshold: 0.30

risk:
  max_positions: 10
  max_position_size: 100.0
  max_total_exposure: 1000.0
  max_positions_per_sector: 5
```

---

### Profil Agressif

```yaml
name: "Aggressive"
use_case: "Expérimental, high volume, requires backtest validation"

filters:
  sectors: ["all"]
  min_volume: 15000
  min_traders_count: 5
  min_holders_count: 200

signals:
  # Seuils absolus
  min_roi_absolute: 5.0
  min_confidence_absolute: 0.6

  # Seuils différentiels
  min_roi_diff: 5.0
  min_confidence_diff: 0.10

  # Exit
  stop_loss: -12.0
  profit_target: 25.0
  exit_mode: "hybrid"
  exit_threshold: 0.35

risk:
  max_positions: 15
  max_position_size: 200.0
  max_total_exposure: 3000.0
  max_positions_per_sector: 8
```

---

## Évolution Cold Start

### Phase 0 : Ultra-Conservateur (Jours 1-7)

```yaml
name: "Cold-Start-Phase-0"
filters:
  sectors: ["all"]
  min_volume: 75000  # Gros markets seulement
  min_traders_count: 15  # Très sélectif

signals:
  min_roi_absolute: 20.0  # Très élevé
  min_confidence_absolute: 0.9
  min_roi_diff: 12.0
  min_confidence_diff: 0.25

  stop_loss: -6.0  # Strict
  profit_target: 15.0
  exit_mode: "target"  # Exit fixe uniquement

risk:
  max_positions: 3  # Très limité
  max_position_size: 30.0
  max_total_exposure: 100.0
```

**Objectif** : Accumuler historique sans pertes
**Expected ROI** : +5-8%/mois

---

### Phase 1 : Conservateur (Jours 8-30)

```yaml
name: "Cold-Start-Phase-1"
# = Profil Conservateur standard (voir ci-dessus)
```

**Objectif** : Enrichir base traders progressivement
**Expected ROI** : +10-12%/mois

---

### Phase 2 : Équilibré/Agressif (Jour 30+)

```yaml
name: "Nominal-Regime"
# = Profil Équilibré ou Agressif (configurable)
```

**Objectif** : Edge complet, signaux fiables
**Expected ROI** : +15-20%/mois

---

## Timing et Cache

| Phase | Cache TTL | Fréquence Refresh | Justification |
|-------|-----------|-------------------|---------------|
| **Phase 1: Seeding** | N/A | 1×/jour | Leaderboard change lentement |
| **Phase 2: Discovery** | Temps réel | WebSocket | Détection immédiate |
| **Phase 3: Enrichment** | 900s (15 min) | Batch 15 min | Compromis fraîcheur/coût |
| **Phase 4: Buy Signal** | N/A | Temps réel | Latence critique |
| **Phase 5: Monitoring** | 300s (5 min) | Batch 5 min | Plus fréquent (surveillance) |
| **Phase 6: Sell Signal** | N/A | Temps réel | Exit rapide requis |

---

## Filtres Qualité Traders

```yaml
trader_filters:
  min_trades_count: 50  # Éviter survivorship bias
  min_volume_total: 10000  # USD
  min_win_rate: 0.50
  max_win_rate: 0.95  # Exclure anomalies
  min_roi: 5.0  # %
```

---

## Filtres Qualité Events

```yaml
event_filters:
  min_volume: 10000  # USD (varie par profil)
  min_top_traders_count: 3  # Minimum signal
  min_holders_count: 100
  min_days_until_resolution: 2
  max_days_until_resolution: 30
  exclude_resolved: true
```

---

## Rate Limits et Contraintes

| Contrainte | Valeur | Impact |
|------------|--------|--------|
| **Bitquery req/min** | 120 (Startup plan) | Max 120 enrichments/min |
| **Polymarket Data API** | 1000/min | OK pour polling |
| **WebSocket connections** | 5 | Max 250 tokens (5×50) |
| **CLOB orders/sec** | 10 | Goulot si multi-signaux |
| **Max trades/jour** | 50 | Cap pour contrôler fees |

---

## Trading Fees Cap

```yaml
fees_management:
  max_trades_per_day: 50  # Hard cap
  max_trades_per_hour: 10  # Burst protection

  # Fee estimation
  avg_trade_size: 100  # USD
  avg_fee_rate: 0.15  # % (moyenne maker 0.1% + taker 0.2%)

  max_fees_per_day: 50 × 100 × 0.0015 = 7.50 USD/jour
  max_fees_per_month: 7.50 × 30 = 225 USD/mois
```

⚠️ **ATTENTION** : 225 USD/mois fees dépasse notre budget total (238 USD) !

**Solution** : Réduire max_trades_per_day ou augmenter budget.

---

## Latences Réalistes

| Opération | Latence | Composants |
|-----------|---------|------------|
| **Buy signal → Ordre placé** | 1000-1500ms | Score read (50ms) + Prix fetch (200ms) + CLOB POST (300ms) + Network (450ms) |
| **Sell signal → Ordre placé** | 800-1200ms | Detection (100ms) + CLOB POST (300ms) + Network (400ms) |
| **Enrichment complet** | 2-5 min | Bitquery 100 markets × 2 sec/market |
| **Monitoring complet** | 1-3 min | Bitquery positions ouvertes |

---

## Score Confiance (Formule)

```python
def calculate_confidence_score(trader):
    roi = trader.roi / 100  # Normaliser
    win_rate = trader.win_rate
    trades_count = trader.trades_count

    # Pénalité volume (survivorship bias)
    if trades_count < 50:
        volume_penalty = trades_count / 50.0
    else:
        volume_penalty = 1.0

    # Score base
    base_score = roi * win_rate

    # Appliquer pénalité
    confidence = base_score * volume_penalty

    # Clamp [0, 1]
    return min(max(confidence, 0.0), 1.0)
```

**Exemple** :
- Trader A : ROI 45%, win rate 90%, 10 trades → confidence = 0.081
- Trader B : ROI 38%, win rate 65%, 500 trades → confidence = 0.247

---

## Métriques KPIs

### Performance Algorithm

| Métrique | Conservateur | Équilibré | Agressif |
|----------|--------------|-----------|----------|
| **Target ROI/mois** | +12% | +15% | +20% |
| **Target Win Rate** | 70% | 65% | 55% |
| **Target Sharpe** | 1.5 | 1.8 | 2.0 |
| **Max Drawdown** | 8% | 12% | 18% |

### Performance Opérationnelle

| Métrique | Target | Alerte si |
|----------|--------|-----------|
| **Latence signal→exec** | < 1500ms | > 3000ms |
| **Taux erreur APIs** | < 1% | > 5% |
| **Couverture holders** | > 95% | < 80% |
| **Uptime workers** | > 99% | < 95% |

---

## Résumé Graphique

```
╔═══════════════════════════════════════════════════════════════════╗
║  COMPARAISON PROFILS                                               ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Profil         │ Trades/mois │ ROI Target │ Max Drawdown        ║
║  ──────────────┼─────────────┼────────────┼─────────────────── ║
║  Conservateur  │ 15-25       │ +12%       │ -8%                 ║
║  Équilibré     │ 40-60       │ +15%       │ -12%                ║
║  Agressif      │ 80-120      │ +20%       │ -18%                ║
║                                                                   ║
║  Cold Start :                                                     ║
║  Jours 1-7   → Ultra-conservateur (ROI +5-8%)                    ║
║  Jours 8-30  → Conservateur (ROI +10-12%)                        ║
║  Jour 30+    → Équilibré/Agressif (ROI +15-20%)                  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

**Version**: 1.0
**Date**: 2026-02-04
**Auteur**: SYM Framework - Corrections sym-opus
**Status**: ✅ Référence centralisée complète
