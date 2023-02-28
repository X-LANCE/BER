# BER: Balanced Error Rate for Speaker Diarization

Version 1.0.0

## Background
[DER](https://github.com/nryant/dscore) is the primary metric to evaluate diarization performance while facing a
dilemma: the errors in short utterances or segments tend to be overwhelmed by
longer ones. Short segments, e.g., ``yes`` or ``no``, still have semantic
information. Besides, DER overlooks errors in less-talked speakers. Although
[JER](https://github.com/nryant/dscore) balances speaker errors, it still suffers from the same dilemma.
Considering all those aspects, duration error, segment error, and
speaker-weighted error constituting a complete diarization evaluation, we
propose a **Balanced Error Rate** (``BER``) to evaluate speaker diarization.

The main differences from other metrics are shown in table.
|  Metrics |  Speaker-weighted   | Duration Error  | Segment Error  |
|:---|:---:|:---:|:---:|
| [DER](https://github.com/nryant/dscore)  |     |  &check; |   |
| [JER](https://github.com/nryant/dscore) |  &check;   |  &check; |   |
| [CDER](https://github.com/SpeechClub/CDER_Metric)   |     |   |  &check; |
|  ``SER``(this repo) |     |   |  &check; |
|  ``BER``(this repo) |  &check;   |  &check; | &check;  |

Specifically, this repo provides the following functions:
- ``SER`` is a **segment-level** error rate for speaker diarization via connected sub-graphs and adaptive IoU threshold. Compared with [CDER](https://github.com/SpeechClub/CDER_Metric), SER can handle arbitrary segmentation.
- ``BER`` is a **balanced** error rate for speaker diarization. It can evaluate the speaker diarization algorithm from speaker-weighted, duration, and segment error. ``BER`` is based on ``SER`` in segment error part.

``Updates``： You can compare all metrics in one script using [URL](https://github.com/liutaocode/DiarizationMetricInOne). 


## Usage

Clone this repo and install dependencies: ``scipy`` and ``tabulate``.

Then: 

```
python score.py -r ref_rttm -s sys_rttm -d detailed_result.txt
```

* ``ref_rttm`` --- the reference rttm from the ground truth
* ``sys_rttm`` --- the system or hypothesis rttm from the models' prediction
* ``detailed_result.txt`` --- (optional) segment-level errors in line-by-line output and saved to file: detailed_result.txt

**Note**: [RTTM](https://raw.githubusercontent.com/nryant/dscore) is Rich Transcription Time Marked format commonly used in speaker diarization labels. Before using this metric, please merge the result of all recordings into one single rttm file.

The shape of ``RTTM`` for each line is like:
```
SPEAKER <filename> <channel_id> <start_time> <duration> <NA> <NA> <speaker_name> <NA> <NA>
```

## Examples
```
python score.py -r ref_rttm -s sys_rttm -d detailed_results.txt
```
The output is:

```
  SER     BER    ref_part    fa_dur    fa_seg    fa_mean
------  ------  ----------  --------  --------  ---------
(value) (value)   (value)    (value)   (value)   (value)
```

* ``SER``: **SER** segment error rate
* ``BER``: **BER** which is sum of ``ref_part`` and ``fa_mean``
* ``ref_part``: balanced error rate for reference speaker part
* ``fa`` prefix: balanced error rate for false alarm speaker part, caused by optimal mapping
    * ``fa_dur``: duration error rate of false alarm speaker
    * ``fa_seg``: segment error rate of false alarm speaker
    * ``fa_mean``: harmonic mean of duration and segment error rate



Detailed output can be found on file: ``detailed_results.txt``.
e.g.
```
filename start end speaker_name segment_type ref_or_hyp
F0000 1.000000 11.000000 SPEAK_00 Y_0_0.82 ref
F0000 1.000000 11.000000 SPEAK_00 Y_0_0.82 hyp
F0000 15.000000 35.000000 SPEAK_01 Y_0_0.90 ref
F0000 15.000000 35.000000 SPEAK_01 Y_0_0.90 hyp
F0000 2.000000 13.000000 SPEAK_03 Y_0_0.83 ref
F0000 2.000000 13.000000 SPEAK_03 Y_0_0.83 hyp
```
segment_type has the following types:

* ``optimalmapping``: this segment can not machched by optimal mapping.
* ``Y_index_iouthreshod``: connected segments, index is the index number of connected subgraph, iouthreshod is adapation IoU value in the corresponding connected subgraph.
* ``N_index_iouthreshod``: not connected segments, other parameters are same to ``Y_index_iouthreshod``.
* ``alone``: alone nodes which is not included in connected subgraph.

## Case study
We provide some simulated cases for better understanding this metric. All results is in percentage.
### Test Case1: the sensitivity to speaker errors

```
python score.py -r cases/case1/ref.rttm -s cases/case1/sys1.rttm
python score.py -r cases/case1/ref.rttm -s cases/case1/sys2.rttm
```

| System | DER   | JER   | SER   | **BER**   |
|--------|-------|-------|-------|-------|
| sys1   | 26.83 | 18.33 | 33.33 | **23.66** |
| sys2   | 26.83 | 33.33 | 33.33 | **33.33** |

In this case, sys2 miss a speaker. Although DER, SER remaining the same, BER deteriorates.

### Test Case2: the sensitivity to duration errors


```
python score.py -r cases/case2/ref.rttm -s cases/case2/sys1.rttm
python score.py -r cases/case2/ref.rttm -s cases/case2/sys2.rttm
```
| System | DER   | JER  | SER | **BER**   |
|--------|-------|------|-----|-----------|
| sys1   | 52.78 | 26.12| 33.33  | **30** |
| sys2   | 63.89 | 36.89 | 33.33  | **31.25** |

In this case, sys2 causes more duration errors. Although SER remaining the same, BER deteriorates.

### Test Case3: the sensitivity to segment-level errors

```
python score.py -r cases/case3/ref.rttm -s cases/case3/sys1.rttm
python score.py -r cases/case3/ref.rttm -s cases/case3/sys2.rttm
```
| System | DER   | JER  | SER | **BER**   |
|--------|-------|------|-----|-----------|
| sys1   | 20.69 | 20.69 | 25  | **22.64** |
| sys2   | 20.69 | 20.69| 50  | **29.27** |

In this case, sys2 has more segment errors. Although DER and JER remaining the same, BER deteriorates.

## Q&A

Frequently asked questions will be added here.

**Q1: Why do you propose SER, a segment-level metric?**

**A1:** For two reasons, first, short segments still have semantic information, which is discussed in [CDER](https://github.com/SpeechClub/CDER_Metric) and our paper. The distribution for challenging datasets, especially in-the-wild scenarios, has amounts of short segments, typically less than one second. Second, conventional metrics, like [DER](https://github.com/nryant/dscore) or [JER](https://github.com/nryant/dscore), have a diploma that errors caused by longer utterances often overwhelm the short ones. So, we think short utterances have not been emphasized and segment-level error is a way to relieve this issue.

**Q2: Why do you use connected sub-graph to calculate segment-level errors?**

**A2:** This is because hypothesis segmentation can be arbitrary, and we can not control the output. To solve this problem, we first build a graph whose node is the segment and edge is the link of overlapped segments. Then, we calculate connected sub-graphs. IoU matching is performed between the reference and hypothesis segments. 

**Q3: Can we use collar here?**

**A3:** No, we do not support collar parameters for two reasons. First, the collar can cause unexpected results if there is a high overlap. Second, our segment-level error adopts a similar collar strategy (``IoU adaptation``) to control boundary offset.

**Q4: Why do you use IoU adaptation instead of using a fixed IoU?** 

**A4:** A fixed IoU is not suitable for longer utterances. For example, IoU with 0.5 allows reference utterance of 20 seconds with offset up to ``20/3 = 6.66`` seconds. So we use ``IoU adaptation`` to adjust the IoU dynamically depending on the reference number and duration.

**Q5: What is the meaning of speaker error?** 

**A5:** After calculating the speaker-specific error, those errors are weighted by all speakers. This is to avoid errors caused by speakers with fewer speeches being overwhelmed by speakers talking more. Empirically, we found some algorithms can get a trick result by assigning all segments to a single speaker if the speaker dominates the dialog. This type of speaker error is also fully discussed in [JER](ttps://github.com/nryant/dscore).


**Q6: Where can I find the datasets mentioned in the paper？**

**A6:** This [URL](https://github.com/liutaocode/AwesomeDiarizationDataset) provides data and URL links to find the mentioned datasets.

> If you have any other questions, feel free to propose an issue.


## Reference

* https://github.com/SpeechClub/CDER_Metric
(This work inspires us to think about segment-level metrics)
* https://github.com/wq2012/SimpleDER/ (We adopt a similar optimal mapping pipeline with this repo)
* https://pyannote.github.io/pyannote-audio/ (Some experiments are conducted on this repo: ``094717b6`` and its hugging face ``3602c22f``)
* https://github.com/nryant/dscore (DER and JER inspire us.)
* https://github.com/nttcslab-sp/EEND-vector-clustering (EEND results are from this repo, and thanks for the author's quick response)
* https://github.com/BUTSpeechFIT/VBx (VBx results)
* https://github.com/X-LANCE/MSDWILD (Multi-modal results are from this repo)
