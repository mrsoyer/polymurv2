# Sécurité & Credentials Management

> Stockage sécurisé des clés privées et protection des accès

---

## Objectif

Documenter la gestion sécurisée des credentials, permissions et best practices sécurité.

---

## Credentials à Protéger

| Type | Usage | Sensibilité |
|------|-------|-------------|
| **Private Key CLOB** | Signature ordres trading | 🔴 CRITIQUE |
| **Bitquery API Key** | On-chain data | 🟡 HAUTE |
| **Alchemy API Key** | Fallback RPC | 🟡 HAUTE |
| **Supabase Service Role Key** | DB admin | 🔴 CRITIQUE |

---

## Stockage Sécurisé

### Variables d'Environnement

```bash
# .env (JAMAIS commiter)
POLYMARKET_PRIVATE_KEY=0x...
BITQUERY_API_KEY=bqy_...
ALCHEMY_API_KEY=alch_...
SUPABASE_URL=https://...
SUPABASE_SERVICE_ROLE_KEY=eyJ...

# .gitignore
.env
.env.local
secrets/
```

### Supabase Secrets (Recommandé)

```bash
# Stocker dans Supabase Vault
supabase secrets set POLYMARKET_PRIVATE_KEY=0x...
supabase secrets set BITQUERY_API_KEY=bqy_...

# Accès dans Edge Functions
const privateKey = Deno.env.get('POLYMARKET_PRIVATE_KEY')
```

---

## Permissions RLS

```sql
-- Table traders: Lecture seule publique
ALTER TABLE traders ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public read traders"
  ON traders FOR SELECT
  USING (true);

-- Table our_positions: Admin seulement
CREATE POLICY "Admin only positions"
  ON our_positions FOR ALL
  USING (auth.jwt() ->> 'role' = 'admin');
```

---

## Rotation des Clés

```yaml
rotation_schedule:
  bitquery_api_key: 90 jours
  alchemy_api_key: 90 jours
  supabase_keys: 180 jours
  
  # Private key CLOB: NE PAS ROTATER
  # (change = nouvelle adresse wallet)
```

---

**Status**: ✅ Sécurité documentée
