# Quant Equity Workbench

Local web app για valuation και short-term forecasting μετοχών με έμφαση σε:

- `DCF` valuation anchor
- `Residual Income` fallback / cross-check
- `Monte Carlo` uncertainty
- `Markov regime switching`
- `AR(1) + GARCH/GJR-GARCH-t`
- `bootstrap + jump Monte Carlo`
- `cross-asset geopolitical and market-stress proxies`
- open-access `GPR index` where available
- reliability diagnostics for valuation and forecasting
- local disk cache για να αποφεύγονται άσκοπα API calls

## Τι δείχνει το app

Για κάθε μετοχή εμφανίζει:

- τρέχουσα τιμή
- median intrinsic value
- margin of safety
- probability ότι η μετοχή είναι undervalued
- forecast για `1 Week`, `3 Weeks`, `1 Month`
- bear / base / bull valuation scenarios
- DCF sensitivity matrix
- regime probabilities και transition matrix
- uncertainty bands
- valuation confidence και forecast quality diagnostics

## Data Stack

Η τρέχουσα free-mode έκδοση είναι χτισμένη πάνω σε:

- `FMP free` για prices, company profile και market context assets
- `Yahoo Finance` fallback για daily price history όταν κάποια μικρότερα / πιο δύσκολα symbols δεν καλύπτονται καλά από το FMP free
- `Stooq` σαν extra backup για daily price history
- `SEC companyfacts` για annual fundamentals όπου υπάρχουν
- open-access `Caldara-Iacoviello GPR` data για geopolitical risk signal όπου είναι διαθέσιμο

Πρακτικά:

- το app δίνει όσο πιο δυνατό valuation γίνεται χωρίς paid fundamentals plan
- το app δείχνει πριν από κάθε run πόσα νέα external requests θα κάνει
- το local cache μειώνει πολύ τα επαναλαμβανόμενα calls
- το valuation δουλεύει καλύτερα για `US-listed SEC filers`
- small-cap / lower-liquidity names μπορούν πλέον να δουλέψουν καλύτερα χάρη στο fallback history stack, αλλά όχι όλα με την ίδια ποιότητα
- όταν δεν υπάρχουν αρκετά SEC fundamentals, το app μένει χρήσιμο σαν `forecast + risk workbench`

## Πώς το τρέχεις σε Mac

1. Άνοιξε `Terminal`

2. Μπες στον φάκελο του project:

```bash
cd "/Users/konstantinosevagorou/Documents/New project"
```

3. Αν δεν υπάρχει virtual environment, φτιάξ' το:

```bash
python3 -m venv .venv
```

4. Ενεργοποίησέ το:

```bash
source .venv/bin/activate
```

5. Εγκατέστησε τα packages:

```bash
pip install -r requirements.txt
```

6. Φτιάξε το `.env`:

```bash
cp .env.example .env
```

7. Άνοιξε το `.env`:

```bash
open -e .env
```

8. Μέσα στο `.env` βάλε το FMP key σου και ένα SEC user agent:

```text
FMP_API_KEY=to_kleidi_sou_edw
SEC_USER_AGENT=ToOnomaSou to_email_sou@example.com
```

9. Τρέξε την εφαρμογή:

```bash
python3 -m streamlit run app.py
```

10. Άνοιξε στον browser:

```text
http://localhost:8501
```

## Πώς το ανεβάζεις δωρεάν στο internet χωρίς να τρέχει στο laptop σου

Η πιο απλή λύση για αυτό το app είναι το `Streamlit Community Cloud`.

Τι κερδίζεις:

- το app τρέχει σε cloud server και όχι στο laptop σου
- ανοίγει από browser και στο κινητό σου
- μπορείς να το κρατήσεις `private`
- όταν κάνεις update στο GitHub repo, γίνεται νέο deploy

### 1. Φτιάξε GitHub account

Αν δεν έχεις, πήγαινε εδώ:

- [GitHub](https://github.com/)

### 2. Φτιάξε private repository

Στο GitHub:

1. πάτα `New repository`
2. βάλε όνομα, π.χ. `quant-equity-workbench`
3. διάλεξε `Private`
4. πάτα `Create repository`

### 3. Ανέβασε τον κώδικα στο GitHub

Από το Terminal μέσα στο project:

```bash
cd "/Users/konstantinosevagorou/Documents/New project"
git init
git add .
git commit -m "Initial app"
git branch -M main
git remote add origin TO_URL_TOU_REPO_SOU
git push -u origin main
```

Αν το repo έχει ήδη remote, απλώς κάνεις:

```bash
git add .
git commit -m "Update app"
git push
```

### 4. Άνοιξε Streamlit Community Cloud

Πήγαινε εδώ:

- [Streamlit Community Cloud](https://share.streamlit.io/)

### 5. Κάνε deploy το app

Συνδέεις το GitHub account σου και μετά:

1. επιλέγεις το private repo σου
2. branch: `main`
3. main file path: `app.py`
4. πάτα `Deploy`

### 6. Βάλε τα secrets του app

Στο app settings του Streamlit Cloud βάλε τα secrets από το αρχείο:

- [.streamlit/secrets.toml.example](/Users/konstantinosevagorou/Documents/New%20project/.streamlit/secrets.toml.example)

Δηλαδή:

```toml
FMP_API_KEY = "to_kleidi_sou"
SEC_USER_AGENT = "To Onoma Sou to_email_sou@example.com"
```

### 7. Κάνε το app private

Στο Streamlit Community Cloud άνοιξε τα settings του app και βάλε το visibility σε `Private`.
Μετά δίνεις access μόνο στο δικό σου email.

Έτσι:

- μπαίνεις εσύ από υπολογιστή
- μπαίνεις και από το κινητό σου
- δεν χρειάζεται να τρέχει το laptop σου

## Πώς κάνεις updates μετά

Κάθε φορά που αλλάζεις κώδικα:

```bash
cd "/Users/konstantinosevagorou/Documents/New project"
git add .
git commit -m "Describe your update"
git push
```

Μετά το Streamlit Cloud κάνει νέο deploy από μόνο του.

## Πώς να τη χρησιμοποιείς σωστά

1. Γράφε κατά προτίμηση `ticker`, π.χ. `AAPL`, `MSFT`, `NVDA`
2. Κοίτα πρώτα το panel `Before You Run`
3. Δες πόσα `estimated API calls` θα κάνει
4. Πάτα `Run Analysis`
5. Διάβασε πρώτα το `Overview`
6. Μετά πήγαινε στα `Valuation` και `Forecast`

## Τι αλλάζει κάθε setting

- `Historical window`: όσο μεγαλώνει, τόσο πιο σταθερά γίνονται τα regime/volatility estimates
- `DCF forecast years`: όσο μεγαλώνει, τόσο πιο πολύ εξαρτάται το valuation από τις μακροχρόνιες υποθέσεις
- `DCF simulations`: περισσότερα simulations δίνουν πιο σταθερό valuation distribution
- `Forecast simulations`: περισσότερα paths δίνουν πιο ομαλά forecast bands
- `Use live quote`: προσπαθεί να κάνει 1 extra FMP request για πιο φρέσκια τιμή
- `Refresh from API now`: αγνοεί το local cache και ζητά νέα δεδομένα

## Τι κάνει το caching

- αν μια μετοχή έχει ήδη φορτωθεί πρόσφατα, το app ξαναχρησιμοποιεί local cache
- αν ξανατρέξεις το ίδιο ticker, συχνά τα νέα calls πέφτουν πολύ
- αν βάλεις όνομα εταιρείας αντί για ticker, μπορεί να χρειαστεί έξτρα lookup call
- το app δείχνει και `Actual Fresh Calls` μετά το run
- οι calls αφορούν πλέον και `FMP` και `SEC` requests

## Σημαντικά

- Δεν υπάρχει τέλειο predictive model για short-term investing
- Το forecast είναι probabilistic, όχι βεβαιότητα
- Το valuation είναι sensitivity-heavy, ειδικά όταν το terminal value έχει μεγάλο βάρος
- Οι γεωπολιτικοί κίνδυνοι δεν προβλέπονται απευθείας. Το app τους προσεγγίζει με proxy variables όπως oil, gold, dollar, rates και risk-off behavior
- Όταν φορτώνει σωστά το `GPR` dataset, το app προσθέτει και άμεσο geopolitical risk signal πάνω από τα market proxies
- Το app είναι για research / εκπαίδευση και όχι προσωπική επενδυτική συμβουλή

## Forecast model comparison

The forecast engine compares nine daily log-return drift models: a zero-drift
random walk, historical drift, AR(1), regularized AR(5), CAPM, empirical Markov
regimes, an adaptive-agent market, online learning agents, and dynamic ARX
(regression using lagged stock and market returns). CAPM estimates beta from aligned SPY returns
with HAC standard errors. Its expected return excludes fitted alpha, uses the
historical market premium, and assumes the current risk-free rate throughout.
CAPM is omitted when market history is insufficient; no extra API key is needed
beyond the existing data stack.

The agent market aggregates trend followers, mean reverters, and noise traders.
Strategy shares adapt to past profitability. The mean-reverting anchor is an
exponentially weighted geometric price average, not independently estimated
intrinsic value. Noise enters through the shared stochastic return process.
These populations are modeling assumptions, not measured investor positions.

Each batch candidate is refitted at chronological one-day forecast origins using
only earlier returns (up to 504 training observations). Online agents process the
available history sequentially with forgetting, issuing each forecast before
observing its target. The earlier validation block
selects the drift model; a later separate holdout reports its error. Each block
contains up to 63 trading days, depending on history length. A candidate must
reduce validation MSE by at least 2% to replace the random walk. Positive holdout
MSE skill indicates improvement against that benchmark; it is not guaranteed.
The 2% rule is a heuristic, not a significance test.

The selected drift drives Monte Carlo paths; hidden Markov regimes, GARCH/GJR,
market-stress proxies, and jumps still shape their risk. Student-t and empirical
innovations are sampled as a mixture to avoid shrinking variance through a
weighted sum. Nonconverged hidden Markov and GARCH fits fall back to supported
alternatives. The random-walk comparison evaluates drift, not a separate full
random-walk volatility model.

Reliability diagnostics now use the later one-day holdout with causal EWMA
volatility and bands calibrated on previously observed standardized errors.
These conformal-style bands adapt as outcomes arrive; serial dependence prevents
claiming distribution-free coverage guarantees. A HAC confidence interval for
holdout MSE improvement reports whether evidence supports a gain against the
random walk. These diagnostics do **not** establish calibration of the full
Monte Carlo distribution or the multi-day forecasts. Real-market evaluation,
longer out-of-sample histories, and transaction costs are needed before claiming
better trading performance.

Run the offline numerical checks from the project directory:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Online learning and computational risk

The learning population contains trend, mean-reversion, autoregressive, and
market specialists plus a fixed random-walk expert. Each specialist learns
regression coefficients with forgetting-factor recursive least squares and
bounded residual updates. Discounted prediction errors determine ensemble
weights, with a small weight floor so specialists can recover after a regime
change. Agents share observed outcomes through the ensemble; generated forecasts
are never treated as ground truth. This is supervised online machine learning,
not an autonomous agent inventing or validating trading strategies.

Learning occurs when analysis processes newly available **daily returns**. There
is no background training service. An unchanged app run does not count as new
training; the existing app cache may return an earlier report until refreshed.
Repeated analyses or more Monte Carlo paths do not create new market evidence.
Use the existing API refresh control when fresh history is needed.

Versioned JSON checkpoints are stored per ticker under ignored `.cache/learning/`.
An exact historical prefix allows incremental updates; unchanged data reuses the
checkpoint without retraining. Revised data, a shifted history window, an older
as-of cutoff, changed market observations, or incompatible checkpoints trigger a
clean chronological replay. Historical comparisons use stored pre-target forecasts,
never final coefficients trained on the holdout. Checkpoint writes are atomic;
unavailable local storage falls back to in-memory learning with a UI warning.
No secrets or credentials are stored in the learning checkpoints.

Learned coefficients and ensemble weights are frozen during Monte Carlo
simulation. Lagged signals evolve along each path, but simulated returns never
update saved learning parameters. Unknown future market signals decay toward
zero as a scenario assumption, not a forecast of market prices. The older
adaptive-agent candidate remains a distinct behavioral scenario whose population
shares can change inside simulations.

The governance panel shows learning counts, agent weights/losses, holdout evidence,
and simulated 95% Value at Risk and Expected Shortfall. Confidence bounds on the
simulated upside probability describe finite Monte Carlo sampling error only;
they exclude model uncertainty. Tail-risk estimates depend on the assumed model
and exclude trading costs. A Streamlit restart/redeployment can remove local
checkpoints; the model rebuilds from available history. No external model API,
additional API key, new dependency, or paid training service is required.
