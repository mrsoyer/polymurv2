# Cahier des Charges - Algorithme Trading Polymarket

> Spécifications fonctionnelles complètes par phase

---

## Vue d'Ensemble

Documentation complète des 6 phases de l'algorithme de trading Polymarket basé sur l'analyse intelligente des holders.

**Date**: 2026-02-04
**Version**: 1.0
**Auteur**: SYM Framework - Orchestrator v4

---

## Structure Documentation

```
concept/
├── _index.md (ce fichier)
├── 00-overview.md           # Vision globale & architecture 7 phases
├── 01-phase-seeding.md      # Construction base traders fiables
├── 02-phase-discovery.md    # Identification events pertinents
├── 03-phase-enrichment.md   # Analyse TOUS holders + scores
├── 04-phase-buy-signal.md   # Génération signaux d'achat
├── 05-phase-monitoring.md   # Surveillance post-achat
├── 06-phase-sell-signal.md  # Génération signaux de vente
├── 07-phase-resolution.md   # ⚠️ Gestion résolution markets
├── 08-cold-start.md         # Stratégie démarrage progressif
├── 09-anti-bias.md          # Mitigation survivorship bias
├── 10-simulateur.md         # ⭐ Simulateur backtest & optimisation
├── 11-thresholds-reference.md # 📋 Référence centralisée seuils
├── 12-error-handling.md     # 🛡️ Résilience & gestion erreurs
├── 13-security.md           # 🔒 Sécurité & credentials
├── 14-testing.md            # ✅ Tests & validation
└── 15-alerting.md           # 🔔 Notifications & rapports
```

---

## Parcours Lecture Recommandé

### 1️⃣ Comprendre la Vision Globale

Commencer par [00-overview.md](00-overview.md) pour:
- Comprendre les 6 phases
- Voir le flux de données complet
- Comprendre la différenciation vs copy trading classique

### 2️⃣ Explorer les Phases (Ordre Chronologique)

1. [01-phase-seeding.md](01-phase-seeding.md) - Construction base traders
2. [02-phase-discovery.md](02-phase-discovery.md) - Surveillance positions
3. [03-phase-enrichment.md](03-phase-enrichment.md) - Analyse holders complets
4. [04-phase-buy-signal.md](04-phase-buy-signal.md) - Signaux d'achat
5. [05-phase-monitoring.md](05-phase-monitoring.md) - Surveillance post-achat
6. [06-phase-sell-signal.md](06-phase-sell-signal.md) - Signaux de vente

### 3️⃣ Comprendre les Défis

- [07-phase-resolution.md](07-phase-resolution.md) - Gestion clôture markets
- [08-cold-start.md](08-cold-start.md) - Démarrage sans historique
- [09-anti-bias.md](09-anti-bias.md) - Éviter faux positifs

### 4️⃣ ⭐ Simulateur & Optimisation (CRUCIAL)

- [10-simulateur.md](10-simulateur.md) - Framework d'expérimentation pour découvrir la stratégie optimale

**Pourquoi crucial** : Au lieu de deviner les paramètres (secteurs, seuils, stop-loss), le simulateur permet de backtester plusieurs stratégies et découvrir empiriquement ce qui fonctionne.

### 5️⃣ 📋 Références & Résilience

- [11-thresholds-reference.md](11-thresholds-reference.md) - Centralisation de TOUS les seuils
- [12-error-handling.md](12-error-handling.md) - Retry, fallback, circuit breaker

---

## Résumé par Phase

### Phase 1: Seeding 🌱
**Input**: Leaderboard API
**Output**: 250+ top traders en DB
**Fréquence**: 1×/jour
**Coût**: Gratuit

**Objectif**: Construire base initiale traders de confiance

[Détails →](01-phase-seeding.md)

---

### Phase 2: Discovery 🔍
**Input**: Positions traders suivis
**Output**: 50-100 events watchlist
**Fréquence**: Temps réel + batch quotidien
**Coût**: Gratuit

**Objectif**: Identifier events où top traders se positionnent

[Détails →](02-phase-discovery.md)

---

### Phase 3: Enrichment 📊
**Input**: Events watchlist
**Output**: Scores ROI + confiance par event
**Fréquence**: Batch 15 min (cache)
**Coût**: 149 USD/mois (Bitquery)

**Objectif**: Analyser TOUS les holders (pas juste 20) et calculer scores

⭐ **PHASE CRITIQUE**: Utilise Bitquery GraphQL pour holders complets

[Détails →](03-phase-enrichment.md)

---

### Phase 4: Buy Signal 💰
**Input**: Scores event + seuils
**Output**: Ordres d'achat automatiques
**Latence**: < 500ms
**Coût**: Trading fees (0.1-0.2%)

**Objectif**: Décider si acheter basé sur consensus holders fiables

[Détails →](04-phase-buy-signal.md)

---

### Phase 5: Monitoring 👁️
**Input**: Nos positions ouvertes
**Output**: Détection vagues de vente
**Fréquence**: Batch 5 min (cache fréquent)
**Coût**: Même budget Bitquery

**Objectif**: Surveiller TOUS holders pour détecter sorties

⭐ **PHASE CRITIQUE**: Monitoring complet tous holders (pas juste 20)

[Détails →](05-phase-monitoring.md)

---

### Phase 6: Sell Signal 🚨
**Input**: Vagues vente + confiance holders
**Output**: Ordres de vente automatiques
**Latence**: < 500ms
**Coût**: Trading fees

**Objectif**: Sortir quand holders fiables vendent

[Détails →](06-phase-sell-signal.md)

---

## Défis Majeurs

### Cold Start Problem

**Problème**: Jour 1 = pas d'historique → impossible calculer ROI traders

**Solution**: Profils progressifs (voir [07-cold-start.md](07-cold-start.md))

```
Jours 1-7   : Conservateur++ (copie leaderboard prudent)
Jours 8-30  : Conservateur (enrichissement progressif)
Jour 30+    : Équilibré/Agressif (régime nominal)
```

---

### Survivorship Bias

**Problème**: Trader avec 10 trades à 90% win rate = chance, pas skill

**Solution**: Pondération volume (voir [08-anti-bias.md](08-anti-bias.md))

```python
confidence = (roi * win_rate) * min(trades_count / 50, 1.0)
```

**Résultat**: Pénaliser traders avec faible historique

---

## Seuils Configurables

L'algorithme supporte 3 profils:

| Profil | ROI min | Confiance min | Position max | Use Case |
|--------|---------|---------------|--------------|----------|
| **Conservateur** | 15% | 0.8 | 50 USD | Cold start, prudent |
| **Équilibré** | 10% | 0.7 | 100 USD | Régime nominal |
| **Agressif** | 5% | 0.6 | 200 USD | Expérimental |

**Configuration**: Voir [00-overview.md](00-overview.md) section "Seuils Configurables"

---

## KPIs Globaux

### Métriques Algorithme

| Métrique | Objectif | Formule |
|----------|----------|---------|
| **ROI Global** | > 15%/mois | (Returns - Invested) / Invested |
| **Win Rate** | > 60% | Trades gagnants / Total |
| **Sharpe Ratio** | > 1.5 | (ROI - Risk-free) / Volatilité |
| **Max Drawdown** | < 20% | Plus grosse perte série |

### Métriques Opérationnelles

| Métrique | Objectif |
|----------|----------|
| **Latence signal→exécution** | < 500ms |
| **Taux erreur APIs** | < 1% |
| **Couverture holders** | > 95% markets |
| **Coût API mensuel** | < 200 USD |

---

## Budget Estimé

**Budget utilisateur**: 200-500 USD/mois

| Composant | Service | Coût/mois |
|-----------|---------|-----------|
| Holders complets | Bitquery Startup | 149 USD |
| RPC fallback | Alchemy Growth | 49 USD |
| Database + Workers | Supabase Pro | 25 USD |
| Trading fees | Polymarket (0.1-0.2%) | ~15 USD |
| **TOTAL** | | **238 USD** ✅ |

**Marge restante**: 262 USD pour scaling

---

## Prochaines Étapes

1. ✅ **Wave 1 complète**: Documentation APIs
2. 🔄 **Wave 2 en cours**: Cahier des charges (ce dossier)
3. ⏳ **Wave 3**: Architecture technique
4. ⏳ **Wave 4**: Schema base de données
5. ⏳ **Wave 5**: Spécifications workers
6. ⏳ **Wave 6**: Organisation & index

---

## Ressources

- [APIs Documentation](../docapi/polymarket/) - Documentation complète APIs
- [Limitations & Stratégie](../docapi/polymarket/limitations.md) - Approche hybride
- [Architecture Technique](../architecture/) - Designs système (Wave 3)
- [Schema DB](../backend/database/) - Tables & relations (Wave 4)
- [Workers](../backend/workers/) - Edge Functions specs (Wave 5)

---

**Version**: 1.3
**Complétude**: 16/16 fichiers ✅
**Corrections**: Seuils alignés, latence réaliste, SQL corrigé, erreurs gérées
**Ajouts**: Résolution, thresholds, error-handling, security, testing, alerting
**Status**: ✅ Concept production-ready
