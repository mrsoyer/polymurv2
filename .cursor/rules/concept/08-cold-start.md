# Cold Start Strategy - Démarrage Progressif

> Évolution des seuils pendant accumulation historique

---

## Problème

Jour 1 = pas d'historique traders → impossible calculer ROI précis

---

## Solution: 3 Phases

### Phase 0 (Jours 1-7)

**Profil**: Ultra-conservateur
- min_roi: 20%
- min_confidence: 0.9
- max_position: 50 USD
- **Objectif**: Accumuler historique sans pertes

### Phase 1 (Jours 8-30)

**Profil**: Conservateur
- min_roi: 15%
- min_confidence: 0.8
- max_position: 100 USD
- **Objectif**: Enrichir base traders progressivement

### Phase 2 (Jour 30+)

**Profil**: Équilibré (configurable)
- min_roi: 10%
- min_confidence: 0.7
- max_position: 200 USD
- **Objectif**: Régime nominal avec edge complet

---

## ROI Attendu par Phase

| Phase | Days | Expected ROI/month | Risk Level |
|-------|------|-------------------|------------|
| Phase 0 | 1-7 | +5-8% | Très faible |
| Phase 1 | 8-30 | +10-12% | Faible |
| Phase 2 | 30+ | +15-20% | Moyen |
