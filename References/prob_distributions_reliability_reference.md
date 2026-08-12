# Probability Distributions Used in Reliability Engineering
## Reference Extract for Bayesian AIE Tutorial

**Source:** O'Connor, A. N., Modarres, M., & Mosleh, A. (2016).
*Probability Distributions Used in Reliability Engineering.*
Center for Risk and Reliability, University of Maryland.
ISBN: 978-0-9966468-1-9. Available free at: https://www.reliability.umd.edu/

> **How this file is used:** This is a curated text extract from the above reference,
> focused on the distributions and Bayesian estimation methods used in the
> AIE Bayesian Tutorial notebook. Sections irrelevant to the tutorial are omitted.
> Math notation is preserved as extracted; some symbols may render as Unicode.

---

## Table of Contents (Tutorial-Relevant Sections)

| Section | Topic | Notebook Use |
|---------|-------|-------------|
| 1.1.5 | Bayes' Law | Foundation for all Bayesian updates |
| 1.1.6 | Likelihood Functions | Exponential/Poisson likelihood construction |
| 1.4.6 | Bayesian Estimation | Prior → Likelihood → Posterior mechanics |
| Part 2 | Exponential Distribution | Failure time model; conjugate likelihood |
| Part 2 | Weibull Distribution | Shape parameter for failure mode identification |
| Part 4 | Gamma Distribution | Conjugate prior for Poisson/Exponential |
| Part 5 | Poisson Distribution | Count likelihood; HPP failure model |

---

## Part 1: Fundamentals of Probability Distributions

### 1.1.5 Bayes' Law

Bayes' rule for continuous and discrete θ respectively:

```
π(θ|E) = π₀(θ) L(E|θ) / ∫ π₀(θ) L(E|θ) dθ      (continuous)

P(θ|E) = P(θ) P(E|θ) / Σᵢ P(E|θᵢ) P(θ)           (discrete)
```

### 1.1.6 Likelihood Functions

The likelihood function is used in Bayesian inference and maximum likelihood
parameter estimation. For the exponential failure model with nF failures and
total time tT (complete + right-censored data):

```
L(E|λ) = λ^nF · exp(-λ · tT)     where  tT = Σ tFi + Σ tSi
```

---

### 1.4.6 Bayesian Estimation

Bayesian estimation uses a subjective interpretation of the theory of probability.
For parameter point estimation and confidence intervals, Bayes' rule updates our
state of knowledge of the unknown of interest (UIO).

#### The Prior Distribution π₀(θ)

The prior distribution captures our state of knowledge of θ **before** evidence
is observed.

- **Dispersed prior** → little known about the parameter
- **Concentrated prior** → good knowledge about likely values of θ

Priors must be proper probability distributions (integrate to one). When a prior
has a constant that does not affect the posterior (improper priors), it is
omitted from formulas.

#### Non-informative Priors

Used when it is not possible to express a subjective prior due to lack of
information, time, or cost.

**1. Improper Uniform Prior (Principle of Indifference)**

Assigns equal probability across all possible values. Uses an improper uniform
distribution with constant (usually 1) over the range of possible values.

Example — exponential likelihood with improper uniform prior, nF failures in tT:
```
Prior:     π₀(λ) = 1  ∝  Gamma(1, 0)
Likelihood: L(E|λ) = λ^nF · exp(-λ·tT)
Posterior:  λ ~ Gamma(λ; 1 + nF, tT)
```

**2. Proper Uniform Prior**

Assigns equal probability within bounds [a, b]. Posterior is a truncated Gamma:
```
Prior:     π₀(λ) = 1/(b-a) = c  ∝  Truncated Gamma(1, 0)
Posterior: π(λ) = c · Gamma(λ; 1 + nF, tT)  for a ≤ λ ≤ b
           π(λ) = 0  otherwise
```
> ⚠ Care must be taken choosing a and b — the posterior will always be zero
> outside these bounds regardless of evidence.

**3. Jeffrey's Prior**

`π₀(θ) = sqrt(det(I_θ))` where I_θ is the Fisher information matrix.
Not dependent on the parameterization chosen. Usually improper.

For exponential/Poisson:
```
Jeffrey's Prior: π₀(λ) = 1/√λ  ∝  Gamma(1/2, 0)
```

#### Conjugate Priors

A **conjugate prior** is one where the posterior is of the same form as the
prior. This reduces Bayesian updating to simple parameter arithmetic.

**Key example: Gamma prior is conjugate to Poisson likelihood**

```
Prior:     π₀(λ) = β^α · λ^(α-1) · exp(-βλ) / Γ(α)   [Gamma(α, β)]

Likelihood (single): Lᵢ(tᵢ|λ) = λ^kᵢ · tᵢ^kᵢ · exp(-λtᵢ) / kᵢ!

Posterior: π(λ|E) = Gamma(λ; α' = α + Σk, β' = β + Σtᵢ)
```

**Derivation result** (from section 1.4.6):

As can be seen the posterior is a gamma distribution with the parameters:
- `α' = α + Σk`   (shape: prior shape + total observed failures)
- `β' = β + Σtᵢ`  (rate: prior rate + total time on test)

> "Therefore the prior and posterior are of the same form, and Bayes' rule does
> not need to be re-calculated for each update. Instead the user can simply
> update the parameters with the new evidence."
> — O'Connor et al. (2016), Section 1.4.6

#### The Posterior Distribution π(θ|E)

The posterior distribution captures our state of knowledge of θ including **all**
prior information and the evidence.

**Point Estimate** (using quadratic loss function = posterior mean):
```
θ̂ = E[π(θ|E)] = ∫ θ π(θ|E) dθ = μ_π
```

---

## Part 2: Exponential Distribution (Common Life Distribution)

### Overview

The exponential distribution models components with **constant failure rate** —
no aging, no wear-out. It is the foundation of the Homogeneous Poisson Process (HPP).

| Function | Formula |
|----------|---------|
| PDF | f(t) = λ·exp(-λt) |
| CDF | F(t) = 1 - exp(-λt) |
| Reliability | R(t) = exp(-λt) |
| Hazard rate | h(t) = λ  **(constant — memoryless property)** |
| Mean (MTTF) | 1/λ |
| Variance | 1/λ² |

### Parameter Estimation (MLE)

For complete and right-censored data:
```
λ̂ = nF / tT     where  tT = Σ tFi + Σ tSi  (total time in test)
```

### Bayesian Estimation

**Non-informative Priors:**

| Type | Prior | Posterior |
|------|-------|-----------|
| Uniform Improper (λ ∈ [0,∞)) | 1 ∝ Gamma(1,0) | Gamma(λ; 1 + nF, tT) |
| Uniform Proper (λ ∈ [a,b]) | 1/(b-a) | c · Gamma(λ; 1 + nF, tT) for a≤λ≤b |
| Jeffrey's Prior | 1/√λ ∝ Gamma(1/2, 0) | Gamma(λ; 1/2 + nF, tT) |
| Novick and Hall | 1/λ ∝ Gamma(0,0) | Gamma(λ; nF, tT) |

**Conjugate Prior:**

| UOI | Likelihood | Evidence | Prior Dist | Prior Params | Posterior Params |
|-----|-----------|---------|-----------|-------------|-----------------|
| λ from Exp(t;λ) | Exponential | nF failures in tT | Gamma | k₀, Λ₀ | k = k₀ + nF, Λ = Λ₀ + tT |

### Example (Worked)

Three vehicle tires run 1000 km each. Punctures at: Tire 1: none; Tire 2: 400km, 900km; Tire 3: 200km.

Total time on test = 3 × 1000 = 3000 km. Total failures = 3.

```
MLE:  λ̂ = nF / tT = 3 / 3000 = 1×10⁻³ punctures/km

90% CI (conservative): [0.272×10⁻³, 2.584×10⁻³]
```

Using Jeffrey's non-informative prior Gamma(1/2, 0) → posterior Gamma(λ; 3.5, 3000):
```
Bayesian point estimate: λ̂ = E[Gamma(3.5, 3000)] = 3.5/3000 = 1.167×10⁻³

90% HDI: [0.361×10⁻³, 2.344×10⁻³]
```

### Applications in Reliability

- **HPP (Homogeneous Poisson Process)**: Components with exponential TTF under
  instantaneous renewal follow an HPP
- **Memoryless property**: The conditional survivor function m(x) = exp(-λx)
  does not depend on t — age has no effect on future failure probability
- **Special case of Gamma**: Exp(t; λ) = Gamma(t; k=1, λ)

---

## Part 2: Weibull Distribution (Common Life Distribution)

### Overview

The Weibull distribution is the most flexible life distribution. Its shape
parameter β identifies the **failure mechanism**.

| β value | Failure Mechanism | Hazard Rate | Phase |
|---------|------------------|-------------|-------|
| β < 1 | Infant mortality / early failures | Decreasing | Early life |
| β = 1 | Random failures (reduces to Exponential) | Constant | Useful life |
| β > 1 | Wear-out / aging | Increasing | End of life |

### Parameters

- **α** (or η): scale parameter (characteristic life — 63.2% failure point)
- **β**: shape parameter (determines failure mechanism)

### Key Reliability Relationships

```
PDF:         f(t) = (β/α)(t/α)^(β-1) · exp(-(t/α)^β)
Reliability: R(t) = exp(-(t/α)^β)
Hazard rate: h(t) = (β/α)(t/α)^(β-1)   [power-law in time]
MTTF:        α · Γ(1 + 1/β)
```

### Bayesian Estimation

When β is known, the scale parameter λ = α^(-β) has a Gamma conjugate prior:

| UOI | Likelihood | Evidence | Prior | Posterior Params |
|-----|-----------|---------|-------|-----------------|
| λ = α^(-β), known β | Weibull (known β) | nF failures at tᵢ | Gamma(k₀, λ₀) | k = k₀ + nF, λ = λ₀ + Σ tᵢ^β |

> Note: "MLE and Bayesian techniques can be used using numerical methods" when
> both α and β are unknown. (Murthy et al. 2003)

### Tutorial Relevance

In the notebook, the Weibull shape parameter is used to:
1. Identify failure mechanism from TTF data
2. Determine whether exponential (constant hazard) is appropriate
3. When k=1 (exponential regime), the conjugate Gamma-Poisson update applies directly

---

## Part 4: Gamma Distribution (Univariate Continuous)

### Overview

The Gamma distribution is the **conjugate prior for Poisson and Exponential**
failure processes. It naturally models rate parameters (λ ≥ 0).

### Parameters

- **k** (or α): shape parameter (k > 0)
- **λ** (or β): rate parameter (λ > 0)

> **Parameterization note:** Some references use scale θ = 1/λ. This notebook
> uses the **rate parameterization** Gamma(k, λ) where mean = k/λ.

### Key Formulas

| Property | Formula |
|----------|---------|
| PDF | f(t) = λ^k · t^(k-1) · exp(-λt) / Γ(k) |
| Mean | k/λ |
| Variance | k/λ² |
| Mode | (k-1)/λ  for k ≥ 1 |
| Skewness | 2/√k |
| Excess kurtosis | 6/k |

### Shape Parameter k Behavior

| k | Behavior |
|---|----------|
| k < 1 | f(0) = ∞, no mode (J-shaped) |
| k = 1 | Reduces to Exponential(λ), mode at t=0 |
| k > 1 | f(0) = 0, unimodal |
| Large k | Approaches Normal(μ=k/λ, σ=√(k/λ²)) |

### Conjugate Prior Table (Gamma as Prior)

| UOI | Likelihood | Evidence | Prior Params | Posterior Params |
|-----|-----------|---------|-------------|-----------------|
| λ from Exp(t;λ) | Exponential | nF failures in tT | k₀, λ₀ | k = k₀ + nF, λ = λ₀ + tT |
| λ from Pois(k;λt) | Poisson | nF failures in tT | k₀, λ₀ | k = k₀ + nF, λ = λ₀ + tT |
| λ from Weibull (known β) | Weibull | nF failures at tᵢ | k₀, λ₀ | k = k₀ + nF, λ = λ₀ + Σtᵢ^β |
| σ² from Normal (known μ) | Normal | nF failures at tᵢ | k₀, λ₀ | k = k₀ + nF/2, λ = λ₀ + ½Σ(tᵢ-μ)² |

### Non-informative Priors for Gamma-distributed Parameters

| Type | Prior | Posterior |
|------|-------|-----------|
| Uniform Improper | 1 | No Closed Form |
| Jeffrey's Prior | λ√(k·ψ'(k) - 1) | No Closed Form |

> "Note that this confidence interval uses the assumption of the parameters
> being normally distributed which is only true for large sample sizes.
> Therefore these confidence intervals may be inaccurate. Bayesian methods
> must be done numerically." — O'Connor et al. (2016), Section 4.x

### Applications

- **HPP Reliability**: Gamma(k, λ) is the probability distribution of the
  **k-th failure time** in an HPP, derived from convolution of k exponential RVs
- **System Failure with Backups**: Models system failure when there are k backup systems
- **Bayesian Analysis**: Primary conjugate prior for Poisson/Exponential failure processes
- **Life Distribution**: Flexible shape makes it a good approximation to life data

### Relationship to Other Distributions

| Special Case | Result |
|-------------|--------|
| Gamma(k=1, λ) | Exponential(λ) |
| Gamma(k=n/2, λ=1/2) | Chi-squared(n) |
| Sum of k i.i.d. Exp(λ) | Gamma(k, λ) |
| Gamma as k→∞ | Normal(μ=k/λ, σ=√(k/λ²)) |

### Worked Example (Conjugate Update)

5 components tested to failure: 38, 42, 44, 46, 55 hours.

MLE gives: k̂ = 21.377, λ̂ = 0.4749

For Bayesian update with Gamma prior Gamma(k₀, λ₀) and nF = 5 failures:
```
Posterior: Gamma(k' = k₀ + 5,  λ' = λ₀ + 225)
```

---

## Part 5: Poisson Distribution (Univariate Discrete)

### Overview

The Poisson distribution models the **number of events (failures) in a fixed
time interval** for a Homogeneous Poisson Process.

**Assumptions for Poisson process:**
1. Chance of two simultaneous events is negligible (or impossible for single-component renewal)
2. Expected number of events is proportional to interval size
3. Event counts in non-overlapping intervals are independent

### Parameters

- **μ** = λt: expected number of failures in time interval t
- **λ**: failure rate (events per unit time)
- **k**: observed number of failures (non-negative integer)

### Key Formulas

| Property | Formula |
|----------|---------|
| PMF | P(K=k) = μ^k · exp(-μ) / k! |
| Mean | μ |
| Variance | μ (equal to mean — a diagnostic property) |
| Mode | ⌊μ⌋ |
| Skewness | 1/√μ |
| Excess kurtosis | 1/μ |

### MLE Parameter Estimation

For n independent Poisson observations (kᵢ failures in time tᵢ):
```
μ̂ = (1/n) Σkᵢ   or   λ̂ = (1/tT) Σkᵢ = nF / tT
```

When there is only one Poisson process:
```
μ̂ = k   or   λ̂ = k/t
```

### Bayesian Estimation

**Non-informative Priors (in known time interval t):**

| Type | Prior | Posterior |
|------|-------|-----------|
| Uniform Proper (λ ∈ [a,b]) | 1/(b-a) | c · Gamma(λ; 1+k, t) for a≤λ≤b |
| Uniform Improper (λ ∈ [0,∞)) | 1 ∝ Gamma(1,0) | Gamma(λ; 1+k, t) |
| Jeffrey's Prior | 1/√λ ∝ Gamma(1/2, 0) | Gamma(λ; 1/2+k, t) |
| Novick and Hall | 1/λ ∝ Gamma(0,0) | Gamma(λ; k, t) |

**Conjugate Prior:**

| UOI | Likelihood | Evidence | Prior | Posterior Params |
|-----|-----------|---------|-------|-----------------|
| λ from Pois(k;μ) | Exponential | nF failures in tT | Gamma(k₀, Λ₀) | k = k₀+nF, Λ = Λ₀+tT |

### Worked Example (Bayesian Poisson Update)

Same tire data as Exponential example: 3 tires × 1000 km, 3 total punctures.

```
MLE: λ̂ = k/tT = 3/3000 = 1×10⁻³ punctures/km

Jeffrey's prior: Gamma(1/2, 0)
Posterior: Gamma(λ; 3.5, 3000)
Bayesian point estimate: λ̂ = 3.5/3000 = 1.167×10⁻³

90% HDI: [0.361×10⁻³, 2.344×10⁻³]
```

### μ Behavior

| μ value | Distribution Behavior |
|---------|----------------------|
| μ ≲ 10 | Skewed right |
| μ ≳ 10 | Approximates Normal(μ, σ=√μ) |
| μ > 1000 | Normal approximation is good |

### Applications

- **HPP counting**: Poisson gives probability that exactly k failures occur in (0, t]
- **Renewal Theory**: Counting function for HPP with exponential inter-arrival times
- **Rare Events**: Approximation to Binomial when n is large and p is small
- **Binomial Approximation**: Use when n ≥ 20 and p ≤ 0.05, or n ≥ 100 and np ≤ 10

### Relationship to Other Distributions

| Distribution | Relationship |
|-------------|-------------|
| Exponential(λ) | Time between Poisson arrivals |
| Gamma(k, λ) | Time to k-th Poisson arrival |
| Normal(μ, σ²=μ) | Limiting case as μ→∞ |
| Binomial(n, p) | Poisson = limiting case as n→∞, np=const |

---

## Summary: Gamma-Poisson Conjugate Update Rules

This is the **core Bayesian mechanism** used throughout the tutorial notebook:

```
Prior:     λ ~ Gamma(k₀, λ₀)
           Mean failure rate = k₀/λ₀
           MTBF prior = λ₀/k₀

Evidence:  nF failures observed in total time tT

Posterior: λ ~ Gamma(k', λ')
           k' = k₀ + nF          (add observed failures to shape)
           λ' = λ₀ + tT          (add total test time to rate)
           
           Posterior mean failure rate = k'/λ' = (k₀ + nF)/(λ₀ + tT)
           Posterior MTBF = λ'/k' = (λ₀ + tT)/(k₀ + nF)
```

### Interpretation

- **More failures (↑ nF)** → posterior mean failure rate increases
- **More time with no failures (↑ tT, fixed nF)** → posterior mean failure rate decreases
- **Vague prior (small k₀, λ₀)** → data dominates the posterior
- **Strong prior (large k₀, λ₀)** → prior beliefs resist being updated

### Expert Prior Elicitation (Path A in Notebook)

Given expert's 90% CI on MTBF = [MTBF_low, MTBF_high]:

```python
import scipy.stats as st
import scipy.optimize as opt

# Convert MTBF CI to failure rate CI (λ = 1/MTBF)
lambda_low  = 1 / MTBF_high   # note: inverse relationship
lambda_high = 1 / MTBF_low

# Fit Gamma(k, λ) to match 5th and 95th percentiles
def residuals(params):
    k, rate = params
    p05 = st.gamma.ppf(0.05, a=k, scale=1/rate)
    p95 = st.gamma.ppf(0.95, a=k, scale=1/rate)
    return [p05 - lambda_low, p95 - lambda_high]

k0, rate0 = opt.fsolve(residuals, x0=[2.0, 2.0 * 2/(lambda_low+lambda_high)])
# Prior: Gamma(k=k0, rate=rate0)
```

---

## Reference: Distribution Parameter Conventions

| Distribution | This Book (O'Connor et al.) | SciPy / PyMC | Conversion |
|-------------|----------------------------|--------------|------------|
| Gamma(k, λ) | shape=k, rate=λ | shape=k, scale=1/λ | `scipy.stats.gamma(a=k, scale=1/λ)` |
| Exponential(λ) | rate=λ | scale=1/λ | `scipy.stats.expon(scale=1/λ)` |
| Poisson(λt) | rate=λ, time=t | mu=λ*t | `scipy.stats.poisson(mu=λ*t)` |
| Weibull(α, β) | scale=α, shape=β | c=β, scale=α | `scipy.stats.weibull_min(c=β, scale=α)` |

---

## Additional References from This Book

- **Conjugate prior derivations:** Section 1.4.6 (pp. 27–30)
- **Gamma distribution full reference:** Section 4.x (pp. 100–107)
- **Poisson distribution full reference:** Section 5.x (pp. 166–171)
- **Exponential distribution full reference:** Section 2.x (pp. 40–67)
- **Weibull distribution full reference:** Section 2.x (pp. 68–84)
- **Bayesian confidence intervals:** Section 1.4.7

### Cross-references to Tutorial Notebook Sections

| Notebook Section | Book Section | Key Concept |
|-----------------|-------------|-------------|
| Section 4 (LLM Parser) | 1.4.6 | Prior distribution selection logic |
| Section 5 (VoI) | 1.1.6, 1.4.5 | Likelihood function construction |
| Section 6 Path A (Expert Prior) | 1.4.6 | Conjugate prior elicitation |
| Section 6 Path B (Data Prior) | 2.x, 1.4.5 | MLE → Gamma prior fitting |
| Section 7 (Bayesian Update) | 1.4.6, Table | k'=k₀+nF, λ'=λ₀+tT update rule |
| Section 9 (Monte Carlo) | 4.x, 5.x | Sampling from Gamma posterior |

---

---

# Complete Distribution Reference — All 22 Distributions
## O'Connor, Modarres & Mosleh (2016) — Full Catalog for Future Notebooks

> **Why this section exists:** The first half of this file covers only the 4 distributions
> used in the current tutorial. This section catalogs all 22 distributions in the book,
> enabling you to build notebooks for problems beyond Gamma-Poisson HPP reliability.
> Math notation that was garbled during PDF extraction has been reconstructed from the
> book's conjugate prior tables and standard references.

---

## How to Use This Catalog

For each distribution you will find:
- **Parameters** and their PyMC names
- **When to use** (reliability problem type)
- **Conjugate prior** (if one exists — enables analytical Bayesian updates)
- **PyMC code snippet** for the prior/likelihood
- **Relationships** to other distributions

### Conjugate Prior Quick-Reference Table (All 22)

| Likelihood | UOI (Unknown) | Conjugate Prior | Posterior Update Rule |
|-----------|--------------|----------------|----------------------|
| Exponential(λ) | λ (failure rate) | Gamma(k₀, β₀) | k' = k₀+nF, β' = β₀+tT |
| Poisson(λt) | λ (event rate) | Gamma(k₀, β₀) | k' = k₀+nF, β' = β₀+tT |
| Weibull(known β) | λ = α⁻β | Gamma(k₀, β₀) | k' = k₀+nF, β' = β₀+Σtᵢᵝ |
| Normal(known σ²) | μ (mean) | Normal(μ₀, σ₀²) | μ' = (μ₀/σ₀²+Σtᵢ/σ²)/(1/σ₀²+n/σ²), σ'² = 1/(1/σ₀²+n/σ²) |
| Normal(known μ) | σ² (variance) | Inverse-Gamma(α₀, β₀) | α' = α₀+n/2, β' = β₀+½Σ(tᵢ-μ)² |
| Lognormal(known μN) | σN² (log-variance) | Inverse-Gamma(k₀, λ₀) | k' = k₀+n/2, λ' = λ₀+½Σ(ln(tᵢ)-μN)² |
| Lognormal(known σN²) | μN (log-mean) | Normal(μ₀, σ₀²) | Standard Normal update on ln(t) data |
| Bernoulli(p) | p (success prob) | Beta(α₀, β₀) | α' = α₀+k, β' = β₀+1-k |
| Binomial(n, p) | p (success prob) | Beta(α₀, β₀) | α' = α₀+k, β' = β₀+n-k |
| Geometric(p) | p (success prob) | Beta(α₀, β₀) | α' = α₀+1, β' = β₀+k-1 |
| Negative Binomial(r,p) | p | Beta(α₀, β₀) | α' = α₀+r·n, β' = β₀+Σkᵢ |
| Pareto(known θ) | α (shape) | Gamma(k₀, β₀) | k' = k₀+n, β' = β₀+Σln(tᵢ/θ) |
| Gamma(known k) | λ (rate) | Gamma(η₀, Λ₀) | η' = η₀+n·k, Λ' = Λ₀+tT |
| Multinomial(n,p) | p (probability vector) | Dirichlet(α₀) | α'ᵢ = α₀ᵢ+kᵢ |
| Uniform(0, b) | b (upper bound) | Pareto(α₀, b₀) | b' = max(data, b₀), α' = α₀+n |
| All others | — | No closed form | Use MCMC (PyMC) |

---

## Part 2: Common Life Distributions

### 2.2 Lognormal Distribution

**When to use:** Failure data that spans orders of magnitude; fatigue crack growth; wear-out
with log-normal degradation; populations where the failure mechanism acts multiplicatively.
Arises naturally when failure is the product of many small independent random multiplicative
factors (Central Limit Theorem on the log scale).

**Parameters:**
- **μN** (mu): mean of the natural log of failure times (location on log scale)
- **σN²** (sigma²): variance of the natural log (shape; controls spread)

> Note: In PyMC, `pm.Lognormal('x', mu=μN, sigma=σN)` where mu and sigma are the
> log-scale parameters (NOT the mean and std of the raw data).

| Property | Formula |
|----------|---------|
| Mean (MTTF) | exp(μN + σN²/2) |
| Median | exp(μN) |
| Mode | exp(μN - σN²) |
| Variance | (exp(σN²)-1)·exp(2μN+σN²) |
| Hazard rate | Non-monotone; rises then falls — NOT memoryless |

**Bayesian Conjugate Priors:**

| UOI | Known | Prior | Posterior Params |
|-----|-------|-------|-----------------|
| σN² | μN known | Inverse-Gamma(k₀, λ₀) | k' = k₀+n/2, λ' = λ₀+½Σ(ln(tᵢ)-μN)² |
| μN | σN² known | Normal(μ₀, σ₀²) | σ'² = 1/(1/σ₀²+n/σN²), μ' = σ'²·(μ₀/σ₀²+Σln(tᵢ)/σN²) |
| Both unknown | — | Normal-Inverse-Gamma | No simple closed form; use PyMC |

**PyMC snippet:**
```python
import pymc as pm
with pm.Model():
    mu_N   = pm.Normal('mu_N', mu=mu0, sigma=sigma0)       # prior on log-mean
    sigma_N = pm.HalfNormal('sigma_N', sigma=1.0)           # prior on log-std
    obs    = pm.Lognormal('obs', mu=mu_N, sigma=sigma_N, observed=ttf_data)
```

**Reliability applications:**
- Material fatigue strength; bearing wear; crack propagation
- Electronic component lifetime in harsh environments
- Field failure data that spans 2+ orders of magnitude
- When Weibull shape β is close to 2.5 (log-normal and Weibull are indistinguishable near this value)

**Key insight:** Unlike Weibull, the Lognormal hazard rate is NOT monotone — it rises to a
peak then decreases. This means very old components that have not yet failed have a
*decreasing* failure rate. This can be realistic for some fatigue mechanisms but is
physically unusual — validate with domain knowledge.

**Relationship to Normal:** If T ~ Lognormal(μN, σN²) then ln(T) ~ Normal(μN, σN²).
MLE on ln(tᵢ) data = standard Normal MLE.

---

### 2.3 Weibull Distribution *(already in main reference — key additions below)*

**Bathtub curve via Weibull mixture:** When β₁ < 1 and β₂ > 1, a 2-fold mixed Weibull
(Section 3.1) produces the classic bathtub hazard shape — useful when modeling mixed
populations (e.g., defective units in a batch + healthy units that wear out).

---

## Part 3: Bathtub Life Distributions

The three bathtub distributions model the full product lifecycle hazard curve (infant
mortality + useful life + wear-out). **No closed-form conjugate priors exist** — always
use PyMC MCMC.

### 3.1 2-Fold Mixed Weibull Distribution

**When to use:** Mixed populations where some units have early-life defects (β₁ < 1) and
survivors are governed by a second wear-out mechanism (β₂ > 1).

**Parameters:** α₁, β₁ (subpopulation 1 Weibull), α₂, β₂ (subpopulation 2 Weibull), p (mixing weight 0≤p≤1)

```
f(t) = p·f₁(t) + (1-p)·f₂(t)
```

**PyMC snippet:**
```python
with pm.Model():
    p    = pm.Beta('p', alpha=1, beta=1)
    a1   = pm.HalfNormal('alpha1', sigma=10)
    b1   = pm.HalfNormal('beta1', sigma=1)     # expect < 1
    a2   = pm.HalfNormal('alpha2', sigma=10)
    b2   = pm.HalfNormal('beta2', sigma=5)     # expect > 1
    # Mixture likelihood requires custom potential or pm.Mixture
```

**Applications:** Semiconductor lifetime; mixed-quality manufacturing batches; field
populations with defect-induced early failures and wear-out dominant later failures.

---

### 3.2 Exponentiated Weibull Distribution

**When to use:** Unimodal or bathtub hazard rates not well-fit by standard Weibull;
adds a second shape parameter for additional flexibility.

**Parameters:** α (scale), β (shape 1), λ (shape 2 — exponent)

```
F(t) = [1 - exp(-(t/α)^β)]^λ
```

**Applications:** Complex systems with two competing failure mechanisms; degradation
data where both early-life and wear-out failure rates are observed in the same dataset.

---

### 3.3 Modified Weibull Distribution

**When to use:** Systems with a linear increase in wear plus an exponential aging
term — models mechanical components where wear-rate accelerates exponentially.

**Parameters:** α (scale), β (linear wear shape), λ (exponential aging rate)

```
h(t) = (α·β·t^(β-1) + α·λ)·exp(λt)
```

**Applications:** Rolling contact fatigue; gear tooth wear; any mechanism where
the hazard rate accelerates faster than a power law (faster than standard Weibull).

---

## Part 4: Univariate Continuous Distributions

### 4.1 Beta Distribution

**When to use:** Modeling **probabilities and proportions** (0 ≤ p ≤ 1). The natural
prior for any Bernoulli/Binomial unknown probability. Also used for bounded continuous
quantities like efficiency, availability, or reliability over a mission.

**Parameters:**
- **α** (alpha): shape 1 (> 0) — analogous to prior "successes"
- **β** (beta): shape 2 (> 0) — analogous to prior "failures"

| Property | Formula |
|----------|---------|
| Mean | α/(α+β) |
| Variance | αβ/[(α+β)²(α+β+1)] |
| Mode | (α-1)/(α+β-2) for α,β > 1 |

**Shape behavior:**
| α, β | Shape |
|------|-------|
| α=β=1 | Uniform (non-informative for p) |
| α>1, β>1 | Unimodal |
| α<1, β<1 | U-shaped (anti-modal) |
| α=β | Symmetric about 0.5 |
| α>1, 0<β<1 | J-shaped (bathtub hazard) |

**Bayesian Conjugate Priors:**

| UOI | Likelihood | Evidence | Posterior Params |
|-----|-----------|---------|-----------------|
| p from Bernoulli(p) | Bernoulli | k successes in 1 trial | α' = α₀+k, β' = β₀+1-k |
| p from Binomial(n,p) | Binomial | k successes in n trials | α' = α₀+k, β' = β₀+n-k |
| p from Geometric(p) | Geometric | 1 success after k trials | α' = α₀+1, β' = β₀+k-1 |

> "Beta(0,0) is an improper prior sometimes used to represent ignorance. Beta(1,1) is
> the standard uniform prior — the minimally informative proper prior for p."
> — O'Connor et al. (2016)

**Interpreting the prior as pseudo-data:** Beta(α₀, β₀) is equivalent to having observed
α₀ prior successes and β₀ prior failures before collecting any real data. This makes
prior elicitation intuitive: ask "how many successes and failures in an equivalent
prior experiment?"

**PyMC snippet:**
```python
with pm.Model():
    p   = pm.Beta('p', alpha=alpha0, beta=beta0)     # prior on probability
    obs = pm.Binomial('obs', n=n_trials, p=p, observed=k_successes)
    # Posterior: Beta(alpha0+k, beta0+n-k)  [analytical]
```

**Applications:**
- **Probability of Failure on Demand (PFD):** safety valve actuation probability
- **Availability:** fraction of time a system is operational
- **Defect rate modeling:** fraction of components that are defective
- **Bayesian reliability demonstration:** updating belief about p from pass/fail test data
- **Proportional hazard covariates:** bounded regression coefficients

---

### 4.2 Birnbaum-Saunders Distribution

**When to use:** Fatigue failure driven by cumulative damage from stress cycles.
Derived from a physical model of crack growth under cyclic loading.

**Parameters:**
- **α** (shape): coefficient of variation of cycle-to-failure counts
- **β** (scale): median number of cycles to failure

| Property | Formula |
|----------|---------|
| Mean | β(1 + α²/2) |
| Variance | (αβ)²(1 + 5α²/4) |

> No closed-form conjugate prior. Use PyMC with HalfNormal priors on α and β.

**PyMC snippet (via Wald/InverseGaussian reparameterization):**
```python
# Birnbaum-Saunders is not natively in PyMC; approximate with:
with pm.Model():
    alpha_bs = pm.HalfNormal('alpha', sigma=1)
    beta_bs  = pm.HalfNormal('beta', sigma=100)
    # Use custom likelihood via pm.Potential or DensityDist
```

**Applications:**
- Fatigue life of metals under cyclic stress (rotating machinery, pressure vessels)
- Crack growth under random loading spectra
- Reliability of structural components (bridges, aircraft wings)

---

### 4.3 Gamma Distribution *(already in main reference)*

---

### 4.4 Logistic Distribution

**When to use:** Symmetric, heavier-tailed alternative to Normal. Models proportional
hazard regression (logistic regression for binary outcomes). The log-logistic distribution
(exponentiated logistic) is used for unimodal hazard rates.

**Parameters:**
- **μ** (location/mean)
- **s** (scale, s > 0) — related to std by σ = s·π/√3

| Property | Formula |
|----------|---------|
| Mean | μ |
| Variance | s²π²/3 |
| Hazard rate | Non-monotone (rises then falls for log-logistic variant) |

> No closed-form conjugate prior for the Logistic distribution. Use Normal or Student-T
> as weakly informative priors on μ and s in PyMC.

**PyMC snippet:**
```python
with pm.Model():
    mu  = pm.Normal('mu', mu=0, sigma=10)
    s   = pm.HalfNormal('s', sigma=5)
    obs = pm.Logistic('obs', mu=mu, s=s, observed=data)
```

**Applications:**
- **Logistic regression** for binary failure/survival outcomes
- **Dose-response reliability:** probability of failure as function of stress
- **Log-logistic distribution** (log of data ~ Logistic): unimodal hazard rate,
  used when Weibull doesn't fit well and the hazard rises then falls
- Accelerated Life Testing (ALT) stress models

---

### 4.5 Normal (Gaussian) Distribution

**When to use:** Symmetric bell-curve data; measurement errors; residuals from regression;
large-sample limit for most distributions (CLT). Less common in reliability (requires
truncation at 0 for lifetime data) but widely used for stress/strength analysis and
for parameters that can be negative.

**Parameters:**
- **μ** (mu): mean (location)
- **σ²** (sigma²): variance (scale)

| Property | Formula |
|----------|---------|
| Mean | μ |
| Variance | σ² |
| Skewness | 0 (symmetric) |
| Hazard rate | Increasing (not memoryless) |

**Bayesian Conjugate Priors (three cases):**

**Case 1 — μ unknown, σ² known:**
```
Prior:     μ ~ Normal(μ₀, σ₀²)
Posterior: μ ~ Normal(μ', σ'²)
           σ'² = 1 / (1/σ₀² + n/σ²)
           μ'  = σ'² · (μ₀/σ₀² + Σtᵢ/σ²)
```

**Case 2 — σ² unknown, μ known:**
```
Prior:     σ² ~ Inverse-Gamma(α₀, β₀)   [equivalently: 1/σ² ~ Gamma(α₀, β₀)]
Posterior: α' = α₀ + n/2
           β' = β₀ + ½Σ(tᵢ-μ)²
```

**Case 3 — Both unknown (most common):**
```
Joint prior: Normal-Inverse-Gamma(μ₀, λ₀, α₀, β₀)
Posterior: Normal-Inverse-Gamma with updated parameters
(No simple closed-form — use PyMC MCMC in practice)
```

**PyMC snippet (μ unknown, σ known):**
```python
with pm.Model():
    mu  = pm.Normal('mu', mu=mu0, sigma=sigma0)       # conjugate prior
    obs = pm.Normal('obs', mu=mu, sigma=sigma_known, observed=data)
    # Analytical posterior: Normal(mu', sigma'^2) per formula above
```

**PyMC snippet (both unknown):**
```python
with pm.Model():
    sigma = pm.HalfNormal('sigma', sigma=10)
    mu    = pm.Normal('mu', mu=0, sigma=100)
    obs   = pm.Normal('obs', mu=mu, sigma=sigma, observed=data)
```

**Applications:**
- **Stress-strength reliability:** P(Strength > Stress) where both are Normal
- **Measurement uncertainty:** sensor noise, instrument error
- **Structural load models:** wind, seismic, traffic loads
- **Regression residuals:** in reliability growth models
- **Prior on μN in Lognormal models** (conjugate; see Lognormal section)
- Parameter for Normal process quality control (SPC)

**Reliability note:** For lifetime data use Truncated Normal (Section 4.8) since
lifetimes must be positive. Full Normal is appropriate for stress, strength, and
load measurements that can theoretically be negative.

---

### 4.6 Pareto Distribution

**When to use:** Heavy-tailed failure distributions; modeling extreme events; "80/20"
(Pareto principle) cost or defect distributions; rare but catastrophic failures.
Arises when the failure mechanism has a power-law tail.

**Parameters:**
- **θ** (theta): minimum value (scale/location, θ > 0)
- **α** (alpha): shape (tail index, α > 0; smaller α = heavier tail)

| Property | Formula |
|----------|---------|
| Mean | αθ/(α-1) for α > 1 |
| Variance | αθ²/[(α-1)²(α-2)] for α > 2 |
| Median | θ·2^(1/α) |
| Hazard rate | α/t (decreasing — infant-mortality-like) |

**Bayesian Conjugate Prior (θ known):**

| UOI | Likelihood | Evidence | Prior | Posterior Params |
|-----|-----------|---------|-------|-----------------|
| α | Pareto(θ, α), θ known | n failures at tᵢ | Gamma(k₀, λ₀) | k' = k₀+n, λ' = λ₀+Σln(tᵢ/θ) |

**PyMC snippet:**
```python
with pm.Model():
    alpha_p = pm.Gamma('alpha', alpha=k0, beta=lambda0)   # conjugate prior
    obs     = pm.Pareto('obs', alpha=alpha_p, m=theta_known, observed=data)
```

**Applications:**
- **Risk/catastrophe modeling:** insurance losses, extreme weather events
- **Defect cost distributions:** few defects cause most of the cost (Pareto principle)
- **Software bug severity:** rare catastrophic bugs vs. common minor ones
- **Heavy-tailed repair time distributions**

---

### 4.7 Triangle Distribution

**When to use:** Expert elicitation when only min, most likely (mode), and max are
known — the "three-point estimate." Easy to parameterize directly from subject matter
expert interviews. Not derived from a physical failure mechanism.

**Parameters:**
- **a** (lower bound, minimum)
- **c** (mode, most likely value)
- **b** (upper bound, maximum)

| Property | Formula |
|----------|---------|
| Mean | (a + b + c)/3 |
| Variance | (a²+b²+c²-ab-ac-bc)/18 |

> No conjugate prior. Use as a prior distribution itself (not as a likelihood model).

**PyMC snippet:**
```python
with pm.Model():
    x = pm.Triangular('x', lower=a, upper=b, c=c)
```

**Applications:**
- **PERT scheduling:** task duration estimates (a=optimistic, c=most likely, b=pessimistic)
- **Expert-elicited cost distributions:** when no historical data exists
- **Monte Carlo input when data is sparse:** easy to explain to non-statisticians
- **Initial prior for VoI analysis** before collecting real failure data

---

### 4.8 Truncated Normal Distribution

**When to use:** Normal distribution bounded below at 0 (or any other bound). Use
whenever the underlying process is approximately Normal but the data cannot be negative
(e.g., lifetimes, repair times, positive physical quantities).

**Parameters:**
- **μ** (location, unrestricted)
- **σ** (scale, σ > 0)
- **lower** (truncation point, typically 0)
- **upper** (upper truncation, typically +∞)

> No closed-form conjugate prior. Use PyMC with Normal prior on μ and HalfNormal on σ.

**PyMC snippet:**
```python
with pm.Model():
    mu    = pm.Normal('mu', mu=mu0, sigma=sigma0)
    sigma = pm.HalfNormal('sigma', sigma=10)
    obs   = pm.TruncatedNormal('obs', mu=mu, sigma=sigma,
                                lower=0, upper=np.inf, observed=data)
```

**Applications:**
- **Positive measurement data** that would naturally follow a Normal distribution
- **Repair time distributions** (cannot be negative)
- **Residual stress distributions** when bounded by material limits
- **Fatigue life near threshold** where below-threshold values are impossible

---

### 4.9 Uniform Distribution

**When to use:** Complete ignorance about a parameter within a bounded range (non-informative
prior); random sampling; Monte Carlo input when only bounds are known.

**Parameters:**
- **lower** (a): lower bound
- **upper** (b): upper bound

| Property | Formula |
|----------|---------|
| Mean | (a+b)/2 |
| Variance | (b-a)²/12 |
| Hazard rate | Increasing (1/(b-t) for t < b) |

**Bayesian Conjugate Prior (b unknown, a known):**

| UOI | Likelihood | Evidence | Prior | Posterior |
|-----|-----------|---------|-------|-----------|
| b (upper bound) | Uniform(0, b) | n samples, max = m | Pareto(α₀, b₀) | b' = max(m, b₀), α' = α₀+n |

**PyMC snippet:**
```python
with pm.Model():
    # Non-informative prior for a parameter bounded in [low, high]
    theta = pm.Uniform('theta', lower=low, upper=high)
```

**Applications:**
- **Non-informative prior** when only physical bounds are known
- **Random component placement** in spatial reliability models
- **Monte Carlo sensitivity analysis** input when no distributional shape is known
- **Discrete uniform**: failure mode selection, random load scenarios

---

## Part 5: Univariate Discrete Distributions

### 5.1 Bernoulli Distribution

**When to use:** Single binary trial — pass/fail, survived/failed, operates/fails-on-demand.
The simplest reliability test: one trial, one outcome.

**Parameter:**
- **p**: probability of failure (success in the Bernoulli sense) — 0 ≤ p ≤ 1

| Property | Formula |
|----------|---------|
| Mean | p |
| Variance | p(1-p) |

**Bayesian Conjugate Prior:**

```
Prior:     p ~ Beta(α₀, β₀)
Likelihood: Bernoulli(p)   — 1 trial, k ∈ {0,1}
Posterior: p ~ Beta(α₀+k, β₀+1-k)
```

> Each Bernoulli trial adds either 1 to α (failure observed) or 1 to β (success observed).

**PyMC snippet:**
```python
with pm.Model():
    p   = pm.Beta('p', alpha=alpha0, beta=beta0)
    obs = pm.Bernoulli('obs', p=p, observed=k_outcomes)   # k_outcomes: 0s and 1s
```

**Applications:**
- **Probability of Failure on Demand (PFD):** each demand is one Bernoulli trial
- **Component pass/fail testing:** each unit is tested once
- **Fault tree leaf nodes:** binary component state (works/fails)
- **Logistic regression:** binary outcome with covariate

**Relationship to Binomial:** Binomial(n, p) = sum of n i.i.d. Bernoulli(p) trials.

---

### 5.2 Binomial Distribution

**When to use:** Count of failures (or successes) in a fixed number of independent
binary trials, each with the same failure probability p. Use when: (1) n is fixed before
testing, (2) trials are independent, (3) each trial is binary, (4) p is constant.

**Parameters:**
- **n**: number of trials (fixed, integer)
- **p**: probability of failure per trial (0 ≤ p ≤ 1)

| Property | Formula |
|----------|---------|
| PMF | C(n,k)·pᵏ·(1-p)^(n-k) |
| Mean | np |
| Variance | np(1-p) |
| Skewness | (1-2p)/√(np(1-p)) |

**Bayesian Conjugate Prior:**

```
Prior:     p ~ Beta(α₀, β₀)
Likelihood: Binomial(n, p)   — k failures in n trials
Posterior: p ~ Beta(α₀+k, β₀+n-k)
```

> Interpretation: α₀ = prior "virtual failures", β₀ = prior "virtual successes".
> Beta(1,1) = uniform (flat) prior — purely data-driven.

**PyMC snippet:**
```python
with pm.Model():
    p   = pm.Beta('p', alpha=alpha0, beta=beta0)
    obs = pm.Binomial('obs', n=n_trials, p=p, observed=k_failures)
    # Analytical posterior: Beta(alpha0+k, beta0+n-k)
```

**Applications:**
- **Reliability demonstration testing:** k failures in n units tested
- **Acceptance sampling:** lot quality (fraction defective)
- **Fault-tree analysis:** probability of k-of-n components failing
- **Bayesian reliability growth:** tracking failure probability across test phases
- **Software testing:** defects found in n code reviews

**Key relationships:**
- Poisson approximation: when n large, p small, Binomial(n,p) ≈ Poisson(np)
- As n→∞ with fixed np: Binomial → Poisson(np)
- For large np: Binomial ≈ Normal(np, np(1-p))

---

## Part 6: Bivariate and Multivariate Distributions

### 6.1 Bivariate Normal Distribution

**When to use:** Joint modeling of two correlated continuous variables (e.g., two
failure modes with correlated rates; stress and strength that are correlated).

**Parameters:** μ₁, μ₂ (means), σ₁², σ₂² (variances), ρ (correlation coefficient, -1 ≤ ρ ≤ 1)

> Conjugate posterior exists but is complex. Use PyMC MvNormal for MCMC.

**PyMC snippet:**
```python
import pymc as pm
import numpy as np
with pm.Model():
    mu    = pm.Normal('mu', mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal('sigma', sigma=5, shape=2)
    corr  = pm.LKJCorr('corr', n=2, eta=2)
    cov   = pm.Deterministic('cov', pm.math.diag(sigma) @ corr @ pm.math.diag(sigma))
    obs   = pm.MvNormal('obs', mu=mu, cov=cov, observed=bivariate_data)
```

**Applications:**
- **Correlated failure modes:** two components that degrade together
- **Stress-strength with correlated loads:** wind + seismic acting simultaneously
- **Joint reliability:** two performance metrics both must exceed thresholds

---

### 6.2 Dirichlet Distribution

**When to use:** Multivariate generalization of Beta — models a probability vector
(p₁, p₂, ..., pK) that sums to 1. The natural conjugate prior for Multinomial data.

**Parameters:** α = (α₁, α₂, ..., αK) concentration parameter vector (αᵢ > 0)

| Property | Formula |
|----------|---------|
| E[pᵢ] | αᵢ/Σαⱼ |
| Var[pᵢ] | αᵢ(Σαⱼ-αᵢ)/[(Σαⱼ)²(Σαⱼ+1)] |

**Bayesian Conjugate Prior:**

```
Prior:     p ~ Dirichlet(α₀)
Likelihood: Multinomial(n, p)   — kᵢ outcomes in category i
Posterior: p ~ Dirichlet(α₀ + k)   [componentwise addition]
```

**PyMC snippet:**
```python
with pm.Model():
    p   = pm.Dirichlet('p', a=alpha0_vector)
    obs = pm.Multinomial('obs', n=n_total, p=p, observed=k_counts)
```

**Applications:**
- **Failure mode allocation:** fraction of failures attributed to each mode
- **Maintenance action probabilities:** probability each intervention type is needed
- **Multi-state reliability:** component is in one of K degradation states

---

### 6.3 Multivariate Normal Distribution

**When to use:** Joint modeling of K correlated continuous variables simultaneously.
Extension of Bivariate Normal to K dimensions.

**Parameters:** μ (K×1 mean vector), Σ (K×K positive-definite covariance matrix)

> Conjugate posterior for μ (known Σ): Multivariate Normal.
> Conjugate posterior for Σ (known μ): Inverse-Wishart.

**PyMC snippet:**
```python
with pm.Model():
    mu  = pm.MvNormal('mu', mu=mu0, cov=cov0, shape=K)
    cov = pm.InverseGamma('cov', ...) # or Wishart prior
    obs = pm.MvNormal('obs', mu=mu, cov=cov, observed=multivariate_data)
```

**Applications:**
- **System-level reliability** with correlated component failure modes
- **Multi-variate degradation models** (multiple wear indicators simultaneously)
- **Structural reliability** with correlated loads

---

### 6.4 Multinomial Distribution

**When to use:** Count data with more than 2 categories — generalizes Binomial to K outcomes.
Each of n trials results in exactly one of K categories.

**Parameters:**
- **n**: total number of trials (fixed)
- **p = (p₁,...,pK)**: probability vector (Σpᵢ = 1)

| Property | Formula |
|----------|---------|
| E[kᵢ] | n·pᵢ |
| Var[kᵢ] | n·pᵢ·(1-pᵢ) |
| Cov[kᵢ,kⱼ] | -n·pᵢ·pⱼ (negative covariance) |

**Bayesian Conjugate Prior: Dirichlet(α₀)**

```
Posterior: Dirichlet(α₀ + k)   where k = (k₁,...,kK) observed counts
```

**PyMC snippet:**
```python
with pm.Model():
    p   = pm.Dirichlet('p', a=alpha0_vector)
    obs = pm.Multinomial('obs', n=n_total, p=p, observed=k_counts)
```

**Applications:**
- **Multi-failure mode data:** categorize n failures into K root-cause categories
- **Maintenance action types:** n work orders of K types
- **Fault tree gate probabilities** with more than 2 outcomes

---

## PyMC Distribution Name Reference

| Book Name | PyMC Class | Key Parameters (PyMC names) |
|-----------|-----------|----------------------------|
| Exponential | `pm.Exponential` | `lam` (rate = 1/scale) |
| Lognormal | `pm.Lognormal` | `mu` (log-mean), `sigma` (log-std) |
| Weibull | `pm.Weibull` | `alpha` (shape=β), `beta` (scale=α) |
| 2-Fold Mixed Weibull | `pm.Mixture` | Two Weibull components + weight |
| Beta | `pm.Beta` | `alpha`, `beta` |
| Birnbaum-Saunders | `pm.DensityDist` | Custom implementation |
| Gamma | `pm.Gamma` | `alpha` (shape=k), `beta` (rate=λ) |
| Logistic | `pm.Logistic` | `mu`, `s` (scale) |
| Normal | `pm.Normal` | `mu`, `sigma` |
| Pareto | `pm.Pareto` | `alpha` (shape), `m` (min value=θ) |
| Triangle | `pm.Triangular` | `lower`, `c` (mode), `upper` |
| Truncated Normal | `pm.TruncatedNormal` | `mu`, `sigma`, `lower`, `upper` |
| Uniform | `pm.Uniform` | `lower`, `upper` |
| Bernoulli | `pm.Bernoulli` | `p` |
| Binomial | `pm.Binomial` | `n`, `p` |
| Poisson | `pm.Poisson` | `mu` (= λ·t) |
| Bivariate Normal | `pm.MvNormal` | `mu` (vector), `cov` (2×2 matrix) |
| Dirichlet | `pm.Dirichlet` | `a` (concentration vector) |
| Multivariate Normal | `pm.MvNormal` | `mu` (vector), `cov` (K×K matrix) |
| Multinomial | `pm.Multinomial` | `n`, `p` (probability vector) |
| Inverse-Gamma | `pm.InverseGamma` | `alpha`, `beta` |
| Half-Normal | `pm.HalfNormal` | `sigma` |

> **Parameterization warning:** PyMC's `pm.Gamma` uses `alpha` (shape) and `beta` (rate).
> SciPy's `scipy.stats.gamma` uses `a` (shape) and `scale` (= 1/rate). Always verify
> which parameterization a library uses before coding.

---

## Decision Guide: Choosing a Distribution for Your Problem

```
Is my data continuous or discrete?
├── DISCRETE (counts, pass/fail)
│   ├── Binary outcome (0/1) per trial?
│   │   ├── One trial → Bernoulli(p)
│   │   └── n trials, count k → Binomial(n, p)
│   ├── Count events in time/space?
│   │   ├── Constant rate → Poisson(λt)
│   │   └── Overdispersed → Negative Binomial
│   └── More than 2 outcome categories?
│       └── Multinomial(n, p)
│
└── CONTINUOUS (time-to-failure, lifetimes)
    ├── Is failure rate CONSTANT (memoryless)?
    │   └── Exponential(λ)  ← simplest, conjugate Gamma prior
    │
    ├── Is failure rate MONOTONE?
    │   ├── Increasing (wear-out) → Weibull(β>1) or Gamma(k>1)
    │   └── Decreasing (infant mortality) → Weibull(β<1)
    │
    ├── Is hazard rate NON-MONOTONE (rises then falls)?
    │   ├── Lognormal
    │   └── Log-logistic
    │
    ├── Is there a BATHTUB curve (all three phases)?
    │   ├── 2-Fold Mixed Weibull
    │   ├── Exponentiated Weibull
    │   └── Modified Weibull
    │
    ├── Is the data MULTIPLICATIVE / log-scale?
    │   └── Lognormal
    │
    ├── Is it FATIGUE / CRACK GROWTH from cycle loading?
    │   └── Birnbaum-Saunders
    │
    ├── Is it a PROPORTION or PROBABILITY (0 to 1)?
    │   └── Beta(α, β)
    │
    ├── Is it BOUNDED (min and max known, no other info)?
    │   ├── Continuous → Uniform or Triangular (if mode known)
    │   └── Truncated Normal (if approximately symmetric)
    │
    └── Is it SYMMETRIC, bell-shaped?
        ├── Normal (if unbounded or far from 0)
        └── Truncated Normal (if bounded below at 0)
```

---

## Notebook Blueprint Templates

For each problem type below, the key distributions and conjugate update are listed:

### Blueprint A: HPP Failure Counting (Current Tutorial)
```
Prior:      failure rate λ ~ Gamma(k₀, β₀)
Likelihood: failures k   ~ Poisson(λ·t)
Posterior:  λ ~ Gamma(k₀+k, β₀+t)
Decision:   sample λ from posterior → Monte Carlo cost simulation
```

### Blueprint B: Pass/Fail Reliability Demonstration
```
Prior:      failure probability p ~ Beta(α₀, β₀)
Likelihood: k failures in n tests ~ Binomial(n, p)
Posterior:  p ~ Beta(α₀+k, β₀+n-k)
Decision:   P(p < threshold) from posterior CDF
```

### Blueprint C: Time-to-Failure with Wear-Out
```
Prior:      Weibull scale λ ~ Gamma(k₀, β₀)  [β fixed or estimated]
Likelihood: n TTF observations ~ Weibull(α, β)
Posterior:  Gamma(k₀+n, β₀+Σtᵢᵝ)  [analytical when β known]
Decision:   MTTF estimates and B10/B50 life predictions
```

### Blueprint D: Log-Normal Fatigue Life
```
Prior:      log-mean μN ~ Normal(μ₀, σ₀²)   [σN assumed known]
Likelihood: ln(TTF) observations ~ Normal(μN, σN²)
Posterior:  μN ~ Normal(μ', σ'²)  [analytical Normal conjugate]
Decision:   P(T > design life) = P(Z > (ln(design_life)-μN')/σN)
```

### Blueprint E: Multi-Failure Mode Attribution
```
Prior:      mode probabilities p ~ Dirichlet(α₀)
Likelihood: failure mode counts k ~ Multinomial(n, p)
Posterior:  p ~ Dirichlet(α₀+k)
Decision:   rank modes by posterior E[pᵢ] for maintenance prioritization
```

### Blueprint F: On-Demand Safety System (PFD)
```
Prior:      PFD p ~ Beta(α₀, β₀)
Likelihood: k demands resulted in failure ~ Binomial(n, p)
Posterior:  p ~ Beta(α₀+k, β₀+n-k)
Decision:   P(PFD < SIL target) from posterior
            VoI: cost of additional proof tests vs. uncertainty reduction
```
