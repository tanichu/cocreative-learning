# Co-Creative Learning via Metropolis-Hastings Interaction between Humans and AI

This repository contains the data, experimental programs, and analysis scripts used in the study:

**"Co-Creative Learning via Metropolis-Hastings Interaction between Humans and AI"**

- **Date:** June 11, 2025  
- **Author:** Tadahiro Taniguchi  
- **Contact:** [taniguchi@i.kyoto-u.ac.jp](mailto:taniguchi@i.kyoto-u.ac.jp)

---

## 📁 Repository Contents

This repository provides the following components:

- **Experimental Programs (oTree):**  
  Web application programs used for the experiment, implemented using [oTree](https://www.otree.org/).

- **Experimental Data:**  
  All data collected via the CrowdWorks crowdsourcing platform.

- **Analysis Programs (Python):**  
  Scripts for computing ARI, bipartite agreement scores, and parameter estimation.

- **Documentation:**  
  Participant instructions and detailed data column descriptions.

---

## 📊 Data Overview

### 1. `hists`-related Files

Posterior distributions generated during the experiment:

- `gibs_hists_100`: Gibbs sampling result  
- `MH_A_hists_100`: MH method (Human as Agent A)  
- `other_B_hists_100`: MH method (Agent B side)  
- `sA_hists_90nin`: Human-inferred posterior (per participant)  
- `sB_hists_90nin`: Agent-inferred posterior (per participant)  

### 2. oTree Program

- `oTree.zip`: Full web-based experimental system

### 3. Agreement Data

- `otree_ab`: Agreement scores per condition  
- `otree_r`: Acceptance probabilities (`r`) per participant  

### 4. Dataset Files

- `ALL`: All attributes merged  
- `A_D5`, `B_D5`: Feature values for human and agent, respectively  
- `shin`: Ground truth IDs

### 5. Summary

- Graphs and figures used in the paper

### 6. File List

- `all_apps_wide-2024-02-28.csv`: Main data table from CrowdWorks  
- `data description`: Data explanation document  
- `oTree.zip`: Full program and assets bundle

---

## 📄 Data Column Descriptions

File: `all_apps_wide-2024-02-28.csv`

### Participant Info

- `participant.id_in_session`, `participant.code`: Unique IDs  
- `participant._index_in_pages`, `participant._max_page_index`: Page index info  
- `participant.time_started_utc`: Start time  
- `participant.order_list`: Image display order  
- `participant.com_model`: AI strategy (0: MH, 1: Accept-All, 2: Reject-All)

### Session Info

- `session.code`: Session ID  
- `session.config.name`: Application name

### Intro Section

- `CrowdWorks_ID`, `sex`, `age`, `is_accepted`: Demographic info and consent

### Communication Part

Naming game interaction data:

- `my_sign`, `com_sign`: Signs proposed  
- `my_accept`, `com_accept`: Acceptance  
- `img{0-9}_sign`, `com_img{0-9}_sign`: Image-level sign assignment  
- `order_{A/B/C}{0-9}`: Drag-and-drop image sorting  
- `r`: AI's acceptance probability  
- `round_number`: Round tracking

### End Section

- `anket_`: Free-text survey  
- `finished`: Whether the experiment was completed

---

## 🧪 Analysis Programs

Python scripts used for post-experiment analysis:

- `ARI_kappa_otree_9onin.py`: Calculates ARI and Kappa coefficient  
- `gurafu_yobi_human_cpu_otree_hon.py`: Plotting graphs from filtered data  
- `mabiki_test.py`: Participant filtering logic  
- `nibu_90nin_100*.py`: Bipartite matching (filtered/unfiltered variants)  
- `otree_datayomikomi_test.py`: CSV reading test  
- `param_suiron_hon_yobi_human_cpu_otree_hon.py`: Parameter inference  
- `T-test.py`: T-test implementation  
- `fitting.py`: Linear regression (`y = ax + b`)

---

## 📬 Contact

For questions, please contact:

**Tadahiro Taniguchi**  
Kyoto University  
📧 [taniguchi@i.kyoto-u.ac.jp](mailto:taniguchi@i.kyoto-u.ac.jp)

---

> This project demonstrates human-AI co-creative learning using probabilistic interaction. The materials provided here are intended to support reproducibility and further research.



Sample figures of the experimental results can be found in:
https://www.dropbox.com/scl/fi/0g19u7ovztulv0kg5trdk/summary.zip?rlkey=6q6f6pq8i0ri5pdaaoooccd7b&dl=0
