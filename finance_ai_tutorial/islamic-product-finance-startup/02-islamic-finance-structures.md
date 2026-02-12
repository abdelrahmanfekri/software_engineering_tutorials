# Module 02: Islamic Finance Structures — Murabaha, Musharakah, Mudarabah, Ijarah

## 2.1 Introduction

Islamic finance substitutes interest-based lending with asset-backed and profit-sharing structures. This module covers the principal structures used in trade finance and product-based financing, with contract mechanics and mapping to product/asset types.

**References**: AAOIFI Shariah Standards (Murabaha 8, Musharakah 12, Mudarabah 13, Ijarah 9); Usmani (2002) *An Introduction to Islamic Finance*.

---

## 2.2 Murabaha (Cost-Plus Sale)

### 2.2.1 Definition

**Murabaha** is a sale where the seller discloses cost and adds a known profit margin. The buyer pays either cash or in installments. It is the dominant structure for trade finance and working capital in Islamic banks.

**Essential elements:**

- Real asset (commodity, equipment, inventory)
- Disclosure of cost to buyer
- Agreed profit margin (fixed, not variable with time)
- Transfer of ownership upon delivery

### 2.2.2 Contract Flow

1. Customer requests financing for specific asset
2. Bank purchases asset from supplier (or through agent)
3. Bank sells asset to customer at cost + margin
4. Customer pays in lump sum or installments

**Critical**: The bank must take constructive or actual possession (qaabd) before resale. Agency arrangements (wakalah) are permitted for purchase on behalf of bank.

### 2.2.3 Murabaha and Product-Based Pricing

| Product/Asset | Typical risk | Margin calibration |
|---------------|--------------|---------------------|
| Food staples | Lower | Lower margin (stable demand, lower default correlation) |
| Electronics / Hardware | Higher | Higher margin (volatility, obsolescence, liquidity risk) |
| Machinery | Medium | Medium margin (asset-backed but illiquid) |
| Commodities (bulk) | Variable | Depends on commodity; standard commodity Murabaha rules |

Margin must be disclosed at inception. It may vary by product/asset risk as long as it is fixed for the specific transaction—**not** a floating rate tied to an interest benchmark. Some institutions use a "profit rate" that is benchmarked but fixed at deal level; AAOIFI requires disclosure and Shariah compliance of the benchmark.

### 2.2.4 Commodity Murabaha (Tawarruq)

Bank buys commodity (e.g. metals on exchange), sells to customer on deferred payment, customer sells commodity for cash. Net effect: liquidity for customer. **Controversial**: Many scholars permit; some (e.g. OIC Fiqh Academy 2009) restrict or prohibit institutional tawarruq. Critical to obtain SSB approval per institution.

---

## 2.3 Musharakah (Partnership)

### 2.3.1 Definition

**Musharakah** is a partnership where two or more parties contribute capital and/or effort and share profit and loss according to pre-agreed ratios. Loss is shared in proportion to capital contribution; profit ratio may differ by agreement.

### 2.3.2 Contract Flow

1. Partners contribute capital (and/or effort)
2. Joint venture undertakes activity (e.g. purchase and sale of goods)
3. Profit distributed per agreed ratio; loss per capital ratio
4. Exit: buyout, liquidation, or diminishing Musharakah (partner gradually buys out financier)

### 2.3.3 Product-Based Application

- **Trade Musharakah**: Bank and trader jointly fund purchase of goods; share profit on sale. Product risk (e.g. hardware vs food) affects profit-sharing ratio and capital commitment.
- **Diminishing Musharakah**: Used for asset finance (e.g. property, equipment). Customer gradually purchases bank's share. Asset type drives risk and pricing.

---

## 2.4 Mudarabah (Profit-Sharing Partnership)

### 2.4.1 Definition

**Mudarabah** is a partnership where one party (rabb al-mal) provides capital and the other (mudarib) provides labor/expertise. Profit is shared per agreed ratio; loss is borne by capital provider (unless mudarib is negligent).

### 2.4.2 Contract Flow

1. Bank provides capital; entrepreneur manages business
2. Profit shared (e.g. 70/30 or 60/40)
3. Loss absorbed by bank; mudarib loses effort
4. No guaranteed return for bank

### 2.4.3 Product-Based Application

- Suitable for working capital and trade where entrepreneur has expertise. Product risk affects profit-sharing ratio negotiation (higher risk → higher mudarib share to compensate, or lower bank participation).

---

## 2.5 Ijarah (Lease)

### 2.5.1 Definition

**Ijarah** is a lease: lessor owns asset and grants use to lessee for rent. Ownership remains with lessor unless Ijarah wa Iqtina (lease-to-own) is used.

**Essential elements:**

- Identified asset
- Known rent (fixed or formula-based, disclosed)
- Lessor bears ownership risk (usufruct transfer only)

### 2.5.2 Ijarah vs Interest-Based Lease

| Aspect | Ijarah | Conventional lease |
|--------|--------|---------------------|
| Basis | Rent for use of asset | Often finance lease (effective purchase) |
| Ownership risk | Lessor | Often transferred to lessee |
| Default | Termination, claim for rent due | May include punitive charges |

### 2.5.3 Product-Based Application

- Equipment, vehicles, real estate. Rent calibrated by asset type, useful life, residual value risk.

---

## 2.6 Structure Selection by Product/Asset

| Product/Asset | Preferred structure(s) | Rationale |
|---------------|------------------------|-----------|
| Trade goods (import/export) | Murabaha | Standard for trade finance; asset exists |
| Working capital (inventory) | Murabaha, Musharakah | Asset-backed; Musharakah for partnership |
| Equipment / machinery | Ijarah, Murabaha | Ijarah for use; Murabaha for purchase |
| Real estate | Ijarah, Diminishing Musharakah | Long-term; ownership transfer |
| SME financing | Musharakah, Mudarabah | Profit-sharing fits venture nature |
| Commodities (bulk) | Murabaha (commodity Murabaha) | Standardized; exchange-traded |

---

## 2.7 Sukuk (Islamic Bonds)

Sukuk represent proportionate ownership in an underlying asset or project. Returns are rent, profit share, or trading gains—not interest. Sukuk structures (Ijarah, Murabaha, Musharakah, Wakalah) must comply with AAOIFI 17 (Investment Sukuk). For product-based financing AI, Sukuk are less central than Murabaha/Musharakah/Ijarah for trade and SME finance.

---

## 2.8 Implications for AI Design

1. **Structure selection logic**: Map product/asset type and tenor to recommended structure (Murabaha, Musharakah, Ijarah).
2. **Pricing logic**: Margin (Murabaha), profit share (Musharakah/Mudarabah), rent (Ijarah)—all tied to product/asset risk, not interest rate.
3. **Contract fields**: Asset description, cost, margin, delivery terms, payment schedule—no interest fields.
4. **Output language**: "Murabaha margin," "profit-sharing ratio," "Ijarah rent"—never "interest rate."
