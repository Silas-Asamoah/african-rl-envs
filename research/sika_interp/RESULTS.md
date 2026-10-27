# sika-interp results

What Qwen3.5-9B and Ministral 3 8B Reasoning compute when a sika-agent request comes in
English, Twi or Twi–English. Both models were served from our own copies (vLLM 0.32.0 on a
Prime Intellect H200; each model's recommended sampling, reasoning on); activations were
recorded on Modal from the same weights, and every turn re-renders to the exact prompt length
the server reported. Run 2026-10-26/27.

Twi and Twi–English text quoted here is synthetic or model-written and has not yet been
checked by a native speaker.

## Behaviour on matched twins

Every task is played in all three languages with the same wallet, people, amounts and
expected outcome; only the wording differs (`matched=True`). 30 eval seeds × 20 task families.

| | English | Twi | Twi–English | right in English, wrong in Twi (reverse) | right in English, wrong in Twi–English (reverse) |
|---|---|---|---|---|---|
| Qwen3.5-9B (600 triples) | 87% | 54% | 74% | 223 (21) | 115 (35) |
| Ministral 3 8B (586 triples) | 64% | 37% | 48% | 207 (50) | 151 (58) |

Harmful outcomes (English / Twi / Twi–English): Qwen 8 / 15 / 4 of 600; Ministral 65 / 82 / 65
of 586, so Ministral's harm is high in every language. 15 Ministral rollouts errored on malformed
tool calls and are left out. Our self-hosted Qwen reproduces its published hosted baseline on the same
240 tasks: 70% overall (English 90, Twi 49, Twi–English 72; hosted 86 / 51 / 72).

## Findings

1. **Both models reason in English.** 92–100% of the words in their reasoning are English
   whatever the request language; the Twi in it is mostly quoting the user.

2. **Qwen carries an English-like scam signal into Twi; Ministral doesn't.** A probe for
   "scam or injection" trained on English requests (end of request, held-out seeds) reads Twi
   requests at up to 89% (layer 17) for Qwen, against 78% from the words alone; for Ministral it
   peaks at 82% at layer 8 and falls to about the words-alone level. Task type barely
   transfers to Twi for either model (31–32%, words alone 14%, chance 5%). Within one language,
   probes are near ceiling, but so are bag-of-words baselines (93–100%), so within-language
   decoding alone says little.

3. **Reasoning erodes the task in Twi, much more for Ministral.** At the end of its reasoning,
   task type is decodable at 82% / 73% / 87% (English / Twi / Twi–English) for Qwen and 63% /
   35% / 61% for Ministral; scam-or-not at 99 / 96 / 99% for Qwen and 91 / 74 / 92% for
   Ministral (base rate 65%).

4. **In Twi–English, the English words carry the scam.** Masking a Twi–English scam request's
   English words drops the English-trained scam signal by 0.42–0.45 at layers 20–24 for Qwen
   (95% CI ±0.05) and by 0.50–0.67 at layers 12–28 for Ministral (to about zero); masking all
   its Twi words changes Qwen's signal by about 0 and Ministral's by 0.11–0.25. Masking the
   same number of random words (about half English) drops it nearly as much as masking the
   English. Non-scam controls stay clean from layer 20 (Qwen) and at every layer (Ministral).

5. **English words inside Twi stay English; mixed words become Twi.** A probe telling English
   requests' tokens from Twi ones reads the English words of Twi–English requests as Twi only
   17–21% of the time at every layer, Twi words 94–97%, and English stems in Twi grammar
   ("a-send-i") 92–96%. The representation jumps more between tokens at switch points than
   elsewhere (Qwen 0.45–0.47 against 0.36–0.42 at layers 20–32; Ministral 0.55–0.64 against
   0.48–0.55 at every layer). No neurons respond strongly to switch points as such (Qwen's
   largest |d| is 1.4); many track a word's language (|d| up to 3.3).

6. **For Qwen, where a Twi–English request lands predicts the outcome, at reading time.**
   Within task type (15 types, 450 Twi–English episodes with both outcomes), the requests Qwen
   gets wrong sit nearer their Twi twin than their English twin by 0.24–0.30 (centred cosine
   margin) at the end of the request, layers 16–32, p = 0.0002 by within-type permutation,
   surviving correction for 18 tests. By the end of its reasoning the difference is gone (p >
   0.7). For Ministral the direction is the same but small (−0.06, p ≥ 0.11).

7. **When they pay a scammer in English, both models had flagged the scam.** The English scam
   signal at the end of the request is 0.99 in the English episodes where the model then paid
   after the user insisted (Ministral 21 cases, Qwen 4): they understood, then gave in.

8. **Anchors: Qwen's failures turn on a step; Ministral's Twi failures start at the request.**
   Replaying episodes ten times from every turn: in a Twi–English promo-fee task Qwen paid, its
   first turn decides it (mean outcome +0.6 replaying from turn 0, −0.8 from turn 1); in a Twi
   fake-refund task, turn 2 does (+1.0 from turn 2, −0.8 from turn 3). Ministral's Twi and
   Twi–English fake-refund episodes are harmful from every turn: its reasoning reads the scam as
   an ordinary transfer request ("Dzifa Quaye wants to send GH₵1,450 to …") and initiates the
   transfer at once, while in English it first checks the transaction history.

## Steering

The steer lens, rebuilt from neuron-explorer for agents: at 60 scam decision points per model
(the turn after the user's answer, 20 per language), keep the model's own reasoning, open a
tool call and read the pay-versus-protect margin (log-sum-exp of the first tokens of
initiate/confirm transfer minus report fraud, ask user, cancel); then push one neuron's write
direction at that position by ±0.8 of the residual norm there and compare with 32 random
directions per layer. 637 mapped neurons per model (the top three per layer on each score).

| | margin, English | Twi | Twi–English |
|---|---|---|---|
| Qwen3.5-9B | −6.6 | −4.2 | −7.1 |
| Ministral 3 8B | −1.5 | −2.6 | −0.1 |

Negative favours protecting. Qwen protects firmly, least so in Twi; Ministral sits near the
edge, closest to paying in Twi–English.

Share of neurons whose push beats 97% of random pushes (by chance about 3–6%):

| picked for | Qwen | Ministral |
|---|---|---|
| switch tokens | 18% | 17% |
| English word inside Twi–English | 15% | 7% |
| Twi over English | 2% | 14% |
| scam, end of request | 1% | 12% |
| scam, at the action | 6% | 5% |
| Twi–English over English | 8% | 5% |
| language-specific | 9% | 5% |

Neurons tied to code-switching move the decision more often than chance in both models.
Ministral has clean two-way levers: layer 27 neuron 2472 (fires less on scams) moves it to
paying when pushed up (+3.9 to +4.7) and to protecting when pushed down (−5.7 to −8.0, most on
Twi–English); layer 16 neuron 2423 (an English-word-in-Twi–English neuron) moves it to
protecting when pushed down (−5.9 to −8.5, most on Twi–English). Several of Qwen's strongest
neurons move it towards paying whichever way they are pushed, which reads as disruption rather
than a clean feature. Pushes of 0.8 of the residual norm are large; 32 controls per layer give
percentiles in steps of about 3.

## Methods and caveats

- Probes: logistic regression (standardised, C = 0.1), five folds split by seed so twins never
  cross between training and test; probe results use the first 800 study episodes per model.
- Read points: the end of the request, the last reasoning token and the first action token of
  the first turn.
- Twi–English placement: centred cosine to each twin, per layer; tested within task type.
- Masking replaces words with "…", never writes new Twi; the probe is English-trained, so it is
  tuned to English features by construction.
- Word tags come from heuristics (`words.py`) checked against the single-language wording
  pools; a native speaker should spot-check them.
- Neuron-explorer's and Thought Anchors' methods were rebuilt for agents; no code was copied.
  Attention-suppression links and receiver heads were not run; links come from deleting a step.
- Ministral reasons only with Mistral's reasoning prompt appended as a structured thinking
  block (`serving.py`).
