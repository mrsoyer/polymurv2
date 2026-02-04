# Algorithme de Trading Polymarket - Vue d'Ensemble

> Cahier des charges fonctionnel pour un système de trading automatisé basé sur l'analyse des holders

---

## Vision Globale

Créer un algorithme de trading pour Polymarket qui :

1. **Suit les meilleurs traders** par secteur (crypto, politics, sports, etc.)
2. **Analyse TOUS les holders** d'un market (pas seulement les 20 premiers)
3. **Calcule un score de fiabilité** pour chaque holder (ROI historique + indice de confiance)
4. **Génère des signaux d'achat/vente** automatiques basés sur l'intelligence collective pondérée
5. **Surveille continuellement** les positions post-achat pour détecter les sorties

---

## Différenciation vs Copy Trading Classique

### Copy Trading Classique

```
┌─────────────────────────────────────────────────────────────┐
│  Approche Naïve                                             │
├─────────────────────────────────────────────────────────────┤
│  1. Suivre top 10 leaderboard                               │
│  2. Copier leurs trades immédiatement                       │
│  3. Problème: tout le monde fait pareil                     │
│  4. Résultat: prix déjà mové, edge perdu                    │
└─────────────────────────────────────────────────────────────┘
```

**Edge perdu**: 3-5% en moyenne (latence + consensus market)

### Notre Approche: Intelligence Collective Pondérée

```
┌─────────────────────────────────────────────────────────────┐
│  Approche Sophistiquée                                      │
├─────────────────────────────────────────────────────────────┤
│  1. Analyser TOUS les holders (pas juste top 20)            │
│  2. Pondérer par ROI historique + confiance                 │
│  3. Détecter consensus avant que market réagisse            │
│  4. Seuils configurables (conservateur/équilibré/agressif)  │
│  5. Monitoring post-achat (tous holders, pas juste 20)      │
└─────────────────────────────────────────────────────────────┘
```

**Edge préservé**: Détection consensus avant mouvement prix

---

## Architecture 6 Phases

### Phase 1: Seeding 🌱
**Objectif**: Construire base initiale de traders fiables

**Input**: Leaderboard Polymarket par secteur
**Output**: 50 top traders/secteur dans DB

**Fréquence**: 1×/jour

**Détails**: Voir [01-phase-seeding.md](01-phase-seeding.md)

---

### Phase 2: Discovery 🔍
**Objectif**: Identifier events où les top traders se positionnent

**Input**: Positions des traders suivis
**Output**: Liste events à surveiller

**Fréquence**: Temps réel (WebSocket)

**Détails**: Voir [02-phase-discovery.md](02-phase-discovery.md)

---

### Phase 3: Enrichment 📊
**Objectif**: Analyser TOUS les holders et calculer scores

**Input**: Market IDs de la watchlist
**Output**: Scores ROI + confiance par market

**Fréquence**: Batch 15 min (cache)

**Détails**: Voir [03-phase-enrichment.md](03-phase-enrichment.md)

---

### Phase 4: Buy Signal 💰
**Objectif**: Décider si acheter basé sur scores agrégés

**Input**: Scores market + prix actuel
**Output**: Ordre d'achat automatique

**Latence**: < 500ms

**Détails**: Voir [04-phase-buy-signal.md](04-phase-buy-signal.md)

---

### Phase 5: Monitoring 👁️
**Objectif**: Surveiller TOUS les holders post-achat

**Input**: Nos positions ouvertes
**Output**: Détection vagues de vente

**Fréquence**: Batch 5 min (cache plus fréquent)

**Détails**: Voir [05-phase-monitoring.md](05-phase-monitoring.md)

---

### Phase 6: Sell Signal 🚨
**Objectif**: Vendre quand holders fiables sortent

**Input**: Vagues de vente détectées
**Output**: Ordre de vente automatique

**Latence**: < 500ms

**Détails**: Voir [06-phase-sell-signal.md](06-phase-sell-signal.md)

---

## Flux de Données Complet

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: SEEDING                         │
│  Leaderboard API → Top Traders → DB (traders)               │
│  Fréquence: 1×/jour                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 2: DISCOVERY                        │
│  WebSocket → Positions Traders → DB (events watchlist)      │
│  Fréquence: Temps réel                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 3: ENRICHMENT                        │
│  Bitquery → TOUS Holders → ROI + Confiance → DB (scores)    │
│  Fréquence: Batch 15 min                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 4: BUY SIGNAL                        │
│  Scores Agrégés → Seuils → CLOB API POST /order             │
│  Latence: < 500ms                                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 5: MONITORING                        │
│  Bitquery → TOUS Holders → Détection Ventes → DB (alerts)   │
│  Fréquence: Batch 5 min                                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 6: SELL SIGNAL                       │
│  Vagues Vente → Analyse Confiance → CLOB API POST /order    │
│  Latence: < 500ms                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Seuils Configurables

L'algorithme supporte 3 profils de risque configurables:

### Conservateur (Cold Start Recommandé)
```json
{
  "min_roi_absolute": 15.0,
  "min_confidence_absolute": 0.8,
  "min_roi_diff": 10.0,
  "min_confidence_diff": 0.2,
  "min_trades_count": 100,
  "max_position_size_usd": 50.0,
  "stop_loss": -8.0,
  "profit_target": 18.0
}
```

**Note** : Voir [11-thresholds-reference.md](11-thresholds-reference.md) pour référence complète des seuils.

**Caractéristiques**:
- Très sélectif (peu de signaux)
- Haute fiabilité attendue
- Petites positions (risque limité)
- Recommandé: Jours 1-30

---

### Équilibré (Régime Nominal)
```json
{
  "min_roi_absolute": 10.0,
  "min_confidence_absolute": 0.7,
  "min_roi_diff": 7.0,
  "min_confidence_diff": 0.15,
  "min_trades_count": 50,
  "max_position_size_usd": 100.0,
  "stop_loss": -10.0,
  "profit_target": 20.0
}
```

**Caractéristiques**:
- Bon compromis volume/qualité
- ROI positif attendu
- Positions moyennes
- Recommandé: Après jour 30

---

### Agressif (Expérimental)
```json
{
  "min_roi_absolute": 5.0,
  "min_confidence_absolute": 0.6,
  "min_roi_diff": 5.0,
  "min_confidence_diff": 0.10,
  "min_trades_count": 25,
  "max_position_size_usd": 200.0,
  "stop_loss": -12.0,
  "profit_target": 25.0
}
```

**Caractéristiques**:
- Plus de signaux, plus de risque
- Nécessite backtest approfondi
- Grosses positions
- Recommandé: Avec monitoring strict

---

## KPIs à Tracker

### Métriques Algorithme

| Métrique | Formule | Objectif |
|----------|---------|----------|
| **ROI Global** | (Returns - Invested) / Invested | > 15%/mois |
| **Win Rate** | Trades gagnants / Total trades | > 60% |
| **Sharpe Ratio** | (ROI - Risk-free) / Volatilité | > 1.5 |
| **Max Drawdown** | Plus grosse perte série | < 20% |
| **Temps moyen position** | Avg(date_close - date_open) | < 7 jours |

### Métriques Opérationnelles

| Métrique | Objectif | Alerte si |
|----------|----------|-----------|
| **Latence signal→exécution** | < 1500ms | > 3000ms |
| **Taux erreur APIs** | < 1% | > 5% |
| **Couverture holders** | > 95% markets | < 80% |
| **Coût API mensuel** | < 200 USD | > 300 USD |
| **Trading fees mensuel** | < 100 USD | > 200 USD |

---

## Évolution Cold Start

```
┌─────────────────────────────────────────────────────────────┐
│  JOURS 1-7: Phase 0 (Copy Trading Prudent)                 │
├─────────────────────────────────────────────────────────────┤
│  Base traders: 50 (leaderboard)                             │
│  Profil: Conservateur++                                     │
│  Objectif: Accumuler historique sans pertes                 │
│  Expected ROI: +5-8%                                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  JOURS 8-30: Phase 1 (Enrichissement Progressif)           │
├─────────────────────────────────────────────────────────────┤
│  Base traders: 200 (découverte organique)                   │
│  Profil: Conservateur                                       │
│  Objectif: Affiner scores, élargir base                     │
│  Expected ROI: +10-12%                                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  JOUR 30+: Phase 2 (Régime Nominal)                        │
├─────────────────────────────────────────────────────────────┤
│  Base traders: 500-1000 (maturité)                          │
│  Profil: Équilibré (configurable)                           │
│  Objectif: Edge complet, signaux fiables                    │
│  Expected ROI: +15-20%                                      │
└─────────────────────────────────────────────────────────────┘
```

**Détails**: Voir [07-cold-start.md](07-cold-start.md)

---

## Risques Identifiés

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| **Survivorship bias** | Haute | Élevé | Pondération volume historique ([08-anti-bias.md](08-anti-bias.md)) |
| **Latence exécution** | Moyenne | Moyen | WebSocket + ordre FOK < 500ms |
| **Rate limiting APIs** | Moyenne | Élevé | Cache agressif + fallback RPC |
| **Budget dépassé** | Faible | Moyen | Monitoring quotidien + alertes |
| **Flash crashes** | Faible | Élevé | Stop-loss automatiques |
| **Corrélation market** | Haute | Moyen | Diversification secteurs |

---

## Prochaines Étapes

1. ✅ **Wave 1 complète**: Documentation APIs ([../docapi/polymarket/](../docapi/polymarket/))
2. 🔄 **Wave 2 en cours**: Spécifications fonctionnelles par phase (ce dossier)
3. ⏳ **Wave 3**: Architecture technique ([../architecture/](../architecture/))
4. ⏳ **Wave 4**: Schema base de données ([../backend/database/](../backend/database/))
5. ⏳ **Wave 5**: Spécifications workers ([../backend/workers/](../backend/workers/))
6. ⏳ **Wave 6**: Organisation & index final

---

## Ressources

- [APIs Documentation](../docapi/polymarket/) - Documentation complète des APIs
- [Limitations & Stratégie](../docapi/polymarket/limitations.md) - Approche hybride multi-sources

---

**Version**: 1.0
**Date**: 2026-02-04
**Auteur**: SYM Framework - Orchestrator v4
**Status**: ✅ Overview complété, détails par phase à suivre
