# Field benchmark

Use the benchmark to measure the shipped checkpoint and candidate warning
policies on audio that matches actual uploads and calls. The runner evaluates
consented local WAV or FLAC fixtures against the canonical manifest format.

- Include genuine speech and current TTS or voice conversion, using similar
  recording paths and codec variants for both classes. For codecs such as Opus
  or G.711, decode the encoded variants to WAV or FLAC and record the original
  channel condition in the manifest.
- Use `dev` to select operating thresholds and reserve `test` for final metrics.
  Keep speakers, source utterances, and derived recordings in one split. Reserve
  generator families for test and inspect the reported holdout list.
- Record the audio hash, consent or license reference, generator family, language,
  codec, and duration. Audio paths can be absolute or relative to the manifest.
  For an external corpus, follow the
  [purpose-specific data protocol](scientific_validation_protocol.md) before
  using it. ASVspoof 5 remains pending in the corpus registry.

```bash
uv run python -m voiceshield.evaluation.field \
  --manifest /path/to/field-manifest.jsonl \
  --target-fpr 0.01 > /path/to/field-report.json
```

The command checks manifest splits, speaker and utterance overlap, duplicate
checksums, and actual audio hashes. It runs the file pipeline and replays 20 ms
PCM frames through the streaming core with synchronous inference. Streaming alert
time is measured on the audio timeline. End-to-end browser latency, network
jitter, and compute backpressure require a separate transport run.

The JSON report gives test recall, false-positive clip rate, false alerts per
hour of genuine audio, median time to the first alert, and abstention, with
breakdowns by generator, language, codec, and duration. An alert episode starts
when a stream enters `HIGH`. Upload warnings are counted once per file and timed
at the end of the recording. Abstentions count in the recall and false-positive
denominators. For candidates, abstention means insufficient eligible scores for
the rule. For the shipped rules, it means no decisive `LOW` or `HIGH` result.

Compare the current file decision to top-two mean pooling with and without
short-file scoring. Compare the current stream policy to an unsmoothed two-of-three
rule over consecutive eligible windows. Candidate thresholds are fit on bona
fide development clips at the requested false-positive budget. The current
policies use the supplied config. Compare both the measured test recall and
test false-positive rates. For a 1% target, 100 genuine dev clips permit only
one false-positive clip. Use substantially more genuine clips to choose and
assess thresholds with useful precision.

`windowing.allow_short_files` is an upload-only opt-in. It checks speech
duration and audio quality on the original audio before the AASIST-L adapter
applies its native cyclic padding. The production default requires a complete
four-second window.

## Model comparison

Freeze the field manifest before training. Train on an authorized training split
and apply comparable codec, replay, gain, and noise conditions to both classes.
Run one experiment with stock AASIST-L, one with a fine-tuned AASIST-L checkpoint,
and one using a wav2vec 2.0 or XLS-R front end. Repeat each evaluation with the
same dev and test partitions. A custom AASIST-L checkpoint can be supplied using
`model.model_id: custom/aasist-l` and `model.checkpoint_path` in a config passed
with `--config`. A separate architecture can use the same replay runner by
implementing the existing `BaseDetector` interface. Compare the resulting alert
rates and recall alongside inference latency. If probability display is needed,
fit a calibrator on a held-out calibration split and validate it separately.
