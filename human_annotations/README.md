To read the human evaluation results, you can simply run the following commands:
```yaml
python3 parse_annotations.py --f MD_versus_{}.json
```
where {} is in ['CD','CS'] for contrastive decoding, contrastive search.

For instance, to see the evaluation results between momentum decoding and contrastive decoding:
```yaml
python3 parse_annotations.py --f MD_versus_CD.json
```

The expected output is displayed as below.

```
========================================== Domain: Wikinews ==========================================
momentum_decoding is better at 57.0%; Two methods are comparable at 3.0%; contrastive_decoding is better at 40.0%
------------------------------------------------------------------------------------------------------

========================================== Domain: Wikitext ==========================================
momentum_decoding is better at 62.5%; Two methods are comparable at 4.5%; contrastive_decoding is better at 33.0%
------------------------------------------------------------------------------------------------------

========================================== Domain: Story ==========================================
momentum_decoding is better at 58.0%; Two methods are comparable at 3.0%; contrastive_decoding is better at 39.0%
------------------------------------------------------------------------------------------------------
```
