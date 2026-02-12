# Module 05: Shariah Governance for AI

## 5.1 Introduction

AI systems used in Islamic finance must comply with Shariah and be auditable by SSBs. This module covers governance for generative AI, data provenance, algorithmic compliance, and SSB-auditable design.

**References**: Shariah Governance Standard on Generative AI for Islamic Financial Institutions (2025); AAOIFI GSIFI 1; FinRegLab (2023) Explainability and Fairness.

---

## 5.2 Shariah Governance Standard on Generative AI (2025)

### 5.2.1 Key Requirements

- **SSB oversight**: SSB must approve use of Gen AI in Shariah-sensitive applications (underwriting, screening, fatwa retrieval, customer-facing advice).
- **Human-in-the-loop**: Gen AI assists; final binding decisions remain with human (e.g. underwriter, SSB).
- **No fatwa issuance**: Gen AI may not issue binding fatwas; may support analysis and retrieval.
- **Transparency**: Disclosure of AI use to customers and regulators.
- **Auditability**: Traceability of inputs, model version, prompts, outputs.

### 5.2.2 Applications

| Application | SSB approval | Human-in-loop | Binding? |
|-------------|--------------|---------------|----------|
| Credit memo generation | Required | Yes (review) | No—human approves |
| Shariah screening | Required | Yes (referral for borderline) | No—SSB sets rules |
| Fatwa retrieval | Required | Yes (scholar interprets) | No—AI assists only |
| Contract analysis | Required | Yes (SSB/legal review) | No |
| Customer chatbot | Required | Varies | No—no binding advice |

---

## 5.3 Data Provenance and Halal-Only Constraint

### 5.3.1 Principles

- Data used for Shariah scoring and underwriting must be **halal-compliant**.
- Interest-based accounts, haram-sector transactions, and prohibited categories must be **excluded** from features that drive Shariah or financing decisions.
- Data sources and transformations must be **documented** for SSB audit.

### 5.3.2 Prohibited Data Categories

| Category | Treatment |
|----------|-----------|
| Interest income/expense | Exclude from cash flow aggregates |
| Haram-sector transactions | Exclude from behavioral features |
| Conventional insurance | Exclude |
| Gambling, alcohol, etc. | Exclude |
| Interest-bearing debt | May be used for screening ratios; not for pricing as interest |

### 5.3.3 Data Provenance Schema

For each feature or score:

- Source (internal system, bureau, third party)
- Halal status (halal-only, mixed, excluded)
- Transformation (aggregation, filtering)
- SSB approval reference (if applicable)

---

## 5.4 Algorithmic Compliance

### 5.4.1 No Riba in Logic or Output

- **Inputs**: No interest rate as direct input for pricing.
- **Outputs**: No "interest rate," "APR," "interest payment" in adverse action or pricing.
- **Labels**: Use "margin," "profit share," "rent," "installment."

### 5.4.2 Gharar Minimization

- **Explainability**: Adverse action reasons must reference concrete factors (payment capacity, asset quality, Shariah screening).
- **Confidence**: Low-confidence predictions → human referral.
- **Gharar score**: Quantify prediction uncertainty; use for referral logic and SSB reporting.

### 5.4.3 Maysir Avoidance

- Financing tied to real assets; no speculative or purely financial transactions.
- Product taxonomy includes "productive use" vs "speculative use."

---

## 5.5 SSB-Auditable Design

### 5.5.1 Traceability

- **Request ID**: Unique ID per application/request.
- **Input log**: Application data, product type, screening result.
- **Model version**: Model ID, version, training date.
- **Output log**: Decision, confidence, adverse action reasons.
- **Timestamp**: Full audit trail.

### 5.5.2 Explainability

- **SHAP/LIME**: Feature contributions for ML models.
- **Adverse action**: Human-readable reasons in Shariah-compliant language (payment capacity, asset liquidity, Shariah screening—never interest).
- **Memo**: Gen AI memo with reasoning chain; human reviews and approves.

### 5.5.3 Periodic SSB Review

- Annual or periodic review of AI logic, prompts, and sample outputs.
- SSB sign-off on methodology and compliance.
- Incident reporting (e.g. model drift, unexpected outputs).

---

## 5.6 Formal Verification (Research Frontier)

### 5.6.1 Concept

Encode AAOIFI-style rules in logic (e.g. SMT, theorem prover) and verify that model outputs do not imply riba or excessive gharar. Early stage; not yet production standard.

### 5.6.2 Example Checks

- Output does not contain forbidden keys: `interest_rate`, `apr`, `riba`.
- Adverse action reasons drawn from approved list only.
- Gharar score below threshold for auto-decision.

---

## 5.7 Implementation Checklist

- [ ] SSB approval for AI use in underwriting/screening
- [ ] No interest-based logic or outputs
- [ ] Halal-only data for Shariah scoring
- [ ] Data provenance documented
- [ ] Adverse action reasons: payment capacity, asset, Shariah—never interest
- [ ] Confidence threshold and human referral for low-confidence
- [ ] Full audit trail (request, model, output, timestamp)
- [ ] Periodic SSB review of AI
