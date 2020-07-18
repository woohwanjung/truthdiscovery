## TDH EAI
This repository is an implementation of our TDH model and EAI algorithm in EDBT 2019 paper
"[Crowdsourced Truth Discovery in the Presence of Hierarchies for Knowledge Fusion](https://doi.org/10.5441/002/edbt.2019.19)".
It is implemented in Python 3.6.


## Running

Example)

```
python experiment.py --data birthplaces --truth_inference TDH --crowdsourcing True --task_assignment EAI
```



## Citation

```
@inproceedings{jung2019crowdsourced,
  title={Crowdsourced Truth Discovery in the Presence of Hierarchies for Knowledge Fusion},
  author={Jung, Woohwan and Kim, Younghoon and Shim, Kyuseok},
  year={2019},
  booktitle = {Advances in Database Technology - 22nd International Conference on
               Extending Database Technology, {EDBT} 2019, Lisbon, Portugal, March
               26-29, 2019},
  pages     = {205--216},
  doi       = {10.5441/002/edbt.2019.19},
}
```