# Anti-Bias - Mitigation Survivorship Bias

> Éviter de suivre traders chanceux vs traders skillés

---

## Problème

**Trader A**: 10 trades, 90% win rate, +45% ROI → Chance
**Trader B**: 500 trades, 65% win rate, +38% ROI → Skill

Naïf: Suivre Trader A (meilleur ROI)
Réalité: Trader B est plus fiable

---

## Solution: Pondération Volume

```python
def calculate_confidence_score(trader):
    roi = trader["roi"]
    win_rate = trader["win_rate"]
    trades_count = trader["trades_count"]
    
    # Pénaliser faible volume
    if trades_count < 50:
        volume_penalty = trades_count / 50.0
    else:
        volume_penalty = 1.0
    
    # Score base
    base_score = (roi / 100) * win_rate
    
    # Appliquer pénalité
    confidence = base_score * volume_penalty
    
    return min(confidence, 1.0)

# Résultats:
# Trader A: (0.45 * 0.9) * (10/50) = 0.081
# Trader B: (0.38 * 0.65) * 1.0 = 0.247
# → Trader B préféré ✓
```

---

## Filtres Additionnels

- Minimum 50 trades requis pour être pris au sérieux
- Minimum 10k USD volume total
- Win rate entre 50% et 95% (anomalies exclues)
- ROI positif minimum 5%

---

**Status**: ✅ Spécifications complètes
