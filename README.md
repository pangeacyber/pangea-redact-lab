<a href="https://pangea.cloud?utm_source=github&utm_medium=python-sdk" target="_blank" rel="noopener noreferrer">
  <img src="https://pangea-marketing.s3.us-west-2.amazonaws.com/pangea-color.svg" alt="Pangea Logo" height="40" />
</a>

<br />

[![documentation](https://img.shields.io/badge/documentation-pangea-blue?style=for-the-badge&labelColor=551B76)](https://pangea.cloud/docs/redact/)

# Efficacy Testing Tool for Pangea Redact

This **command‑line utility** measures how accurately the [Pangea Redact](https://pangea.cloud/docs/redact/) service identifies and redacts PII.  
It runs your examples against the any Redact deployment (SaaS, Private Cloud/Edge), compares results to ground‑truth data, and reports **accuracy, precision, recall, and F1‑score**.  
False‑positive and false‑negative examples can be exported for deeper analysis.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
|Python | **≥3.10** |
|Poetry | **≥1.x** |
|Pangea account | Free sign‑up |

1. **Signup or login** → <https://pangea.cloud/signup>.  
2. In the UserConsole, enable **Redact** (sidebar →Redact→Enable).  
3. From the service **Overview** page copy:  
   * **BaseURL** (e.g. `https://redact.aws.us.pangea.cloud`)  
   * **DefaultToken** (API access token)  
4. Set them as environment variables **or** put them in a `.env` file:

   ```bash
   export PANGEA_BASE_URL="https://redact.<domain>"
   export PANGEA_REDACT_TOKEN="<default-token>"
   ```

5. Install dependencies:

   ```bash
   poetry install --no-root
   ```

*Python ≥3.10 (3.11 supported)*

---

## Dataset Format (JSONL)

Each line contains the test text and an **array of expected PII entity types**:

```json
{"text": "Contact Jane at jane.doe@example.com", "expected_entities": ["EMAIL_ADDRESS", "PERSON"]}
```

Supply one or more JSONL files; the tool aggregates them.

---

## Usage

```text
poetry run python redact_lab.py -i FILE [FILE ...]
                    [--output_metrics METRICS_JSON]
                    [--fp_out FP_JSONL]
                    [--fn_out FN_JSONL]
                    [--include_partials]
                    [--rps RPS]
                    [--verbose]
```

The script prints a legend explaining *Base/Correct/Factual* tiers after the metrics table.

---

### Key Flags

| Flag | Purpose |
|------|---------|
|`-i`,`--input_files` | JSONL datasets to evaluate |
|`--output_metrics` | Save summary metrics (JSON) |
|`--fp_out`,`--fn_out` | Save FP / FN examples (JSONL) |
|`--include_partials` | Count **PARTIAL /MISMATCH** as failures (strict) |
|`--rps` | Requests per second (default1) |
|`--verbose` | Print FP / FN examples to console |

---

## Example Commands

```bash
#Quick run with verbose output
poetry run python redact_lab.py -i data/sample.jsonl --verbose

#Strict mode + export errors
poetry run python redact_lab.py -i data/sample.jsonl \
    --include_partials \
    --output_metrics results.json \
    --fp_out fp.jsonl \
    --fn_out fn.jsonl

#Higher throughput
poetry run python redact_lab.py -i data/large.jsonl --rps 5
```

---

## Metrics Reported

| Metric                | Formula / Meaning                                     |
|-----------------------|-------------------------------------------------------|
| Accuracy              | Fully‑correct examples / total                        |
| Precision(P)         | TP / (TP + FP)                                        |
| Recall(R)            | TP / (TP + FN)                                        |
| F1‑score              | 2·(P·R)/(P+R)                                 |
| Specificity           | TN / (TN + FP)                                        |
| FP‑rate               | FP / (FP + TN)                                        |
| FN‑rate               | FN / (TP + FN)                                        |
| LatencyP50 / P95 / P99 | 50th,95th,99th‑percentile server‑side latency     |

**Efficacy tiers**

1. **Base** (“effective”) – *entity label only*  
2. **Offset** (“correct”) – entity label **plus** exact start/end character offsets  
3. **Overall** (“factual”) – entity label, exact offsets, **and** correct redaction method
---

## Sample Dataset

A sample dataset is available at `data/redact_test.jsonl`.

---

## EdgeDeployments

Point `PANGEA_BASE_URL` to your edge/on‑prem Redact instance (e.g. viaportforwarding) and run the tool normally. See the [Edge guide](https://pangea.cloud/docs/deployment-models/edge/) for details.

---

Made with ❤️ by the Pangeateam — contributions welcome!
