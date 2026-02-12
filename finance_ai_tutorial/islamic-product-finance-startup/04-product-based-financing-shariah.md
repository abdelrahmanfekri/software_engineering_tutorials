# Module 04: Product-Based Financing & Shariah

## 4.1 Introduction

Product-based financing prices risk by the underlying asset or product rather than a generic interest rate. Islamic finance is inherently product- and asset-backed. This module covers trade finance, commodity Murabaha, product/asset risk taxonomy, and mapping to Shariah structures.

**References**: AAOIFI SS 8 (Murabaha); ICC Uniform Rules for Demand Guarantees; Islamic trade finance practice.

---

## 4.2 Trade Finance in Islamic Context

### 4.2.1 Structures

| Structure | Use case | Flow |
|-----------|----------|------|
| Murabaha | Import/export financing | Bank buys goods, sells to customer at cost + margin |
| Wakalah | Agency for purchase | Bank appoints customer as agent to buy; bank pays supplier |
| Musharakah | Joint venture trade | Bank and trader fund purchase; share profit on sale |
| Letter of Credit (LC) | Documentary trade | Islamic banks use LC with Murabaha or Wakalah |

### 4.2.2 Commodity Murabaha (for liquidity)

Bank buys commodity (e.g. metal on LME, palm oil) and sells to customer on deferred payment; customer sells for cash. Used for working capital and liquidity. **Controversial**—SSB approval required. Some scholars restrict institutional tawarruq.

---

## 4.3 Product/Asset Risk Taxonomy

### 4.3.1 Dimensions

| Dimension | Low risk | Medium risk | High risk |
|-----------|----------|-------------|-----------|
| **Liquidity** | High (e.g. food staples) | Medium (machinery) | Low (specialized equipment) |
| **Volatility** | Stable price | Moderate | High (electronics) |
| **Obsolescence** | Low | Medium | High (tech hardware) |
| **Default correlation** | Low | Medium | High (cyclical) |
| **Shariah sensitivity** | Clearly halal | Mixed (e.g. tobacco-adjacent) | Borderline |

### 4.3.2 Example Classification

| Product category | Liquidity | Volatility | Typical structure | Margin calibration |
|------------------|-----------|------------|-------------------|---------------------|
| Food staples | High | Low | Murabaha | Lower margin |
| Electronics | Medium | High | Murabaha | Higher margin |
| Machinery | Low | Medium | Murabaha, Ijarah | Medium–higher |
| Textiles | Medium | Medium | Murabaha | Medium |
| Commodities (bulk) | High | Variable | Murabaha, commodity Murabaha | Benchmark + spread |
| Real estate | Low | Medium | Ijarah, Diminishing Musharakah | Rent/margin by asset |

### 4.3.3 Shariah Screening by Product

- **Halal goods**: Food (excluding pork, alcohol), textiles, machinery, electronics (general use)
- **Conditional**: Tobacco supply chain (some permit; many restrict)
- **Prohibited**: Alcohol, pork, gambling equipment, interest-based financial instruments, haram media

---

## 4.4 Margin Calibration (Murabaha)

### 4.4.1 Principles

- Margin is **disclosed profit** on sale, not interest.
- Margin may vary by product/asset risk, tenor, counterparty.
- Margin must be **fixed at contract inception** for the specific transaction.
- No floating rate pegged to interest benchmark without SSB approval and disclosure.

### 4.4.2 Factors Affecting Margin

| Factor | Impact |
|--------|--------|
| Product risk | Higher risk → higher margin |
| Tenor | Longer → typically higher margin |
| Counterparty risk | Higher default risk → higher margin |
| Asset liquidity | Lower liquidity → higher margin |
| Market conditions | Supply/demand for financing |

### 4.4.3 Benchmarking

Many institutions use a "profit rate benchmark" (e.g. cost of funds, interbank Islamic rate, or conventional benchmark with transformation). AAOIFI requires:

- Disclosure to customer
- SSB approval of benchmark
- Margin fixed for deal (benchmark movement affects new deals, not existing)

---

## 4.5 Profit-Sharing Calibration (Musharakah/Mudarabah)

- **Profit ratio**: Negotiated; may reflect risk, effort, capital contribution.
- **Higher risk**: Often higher mudarib/entrepreneur share to compensate for uncertainty.
- **Product type**: Affects expected return and volatility; influences ratio negotiation.

---

## 4.6 Data Requirements for Product-Based AI

### 4.6.1 Product/Asset Data

- Product category (HS code or internal taxonomy)
- Liquidity score, volatility, obsolescence
- Shariah classification (halal, conditional, haram)
- Historical default/loss by product type

### 4.6.2 Transaction Data

- Cost, margin, tenor
- Counterparty, industry
- Collateral/asset description

### 4.6.3 Halal-Only Constraint

For Shariah scoring and underwriting:

- Exclude haram-sector transaction data from features
- Exclude interest-based accounts from cash flow features
- Document data provenance for SSB audit

---

## 4.7 Implications for AI Design

1. **Product taxonomy**: Define hierarchy (category, subcategory, product type) with risk and Shariah attributes.
2. **Risk scoring**: Product risk score (liquidity, volatility, default correlation) as input to margin/profit-share recommendation.
3. **Structure selection**: Map product/asset and tenor to Murabaha, Musharakah, Ijarah.
4. **Margin engine**: Output recommended margin by product risk tier and counterparty; never "interest rate."
5. **Screening**: Integrate Shariah screening (sector, product) before risk scoring.
