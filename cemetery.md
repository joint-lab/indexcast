# Project Cemetery

This file records deprecated, removed, or unused components for historical reference.

---

## Date: Nov. 26

### Dagster Assets
- **prompt generation**  
  Generated prompts for each index question, prompting for relevance to the index question using market information.  
  Removed in favor of direct relevance scoring when classifying.

- **index relevance asset**  
  Produced a relevance score measuring how closely market information aligned with the H5N1 index.  
  Removed.

- **temporal relevance asset**  
  Produced a relevance score based on the timing of market information relative to the H5N1 index.  
  Removed.

- **geographic relevance asset**  
  Produced a relevance score based on the geographic context of market information relative to the H5N1 index.  
  Removed.

---

### Text Files
- **geographic_relevance_prompt.j2**  
  Template used to generate prompts for geographic relevance scoring.  
  Removed.

- **index_question_relevance_prompt.j2**  
  Template used to generate prompts for index question relevance scoring.  
  Removed.

- **temporal_relevance_prompt.j2**  
  Template used to generate prompts for temporal relevance scoring.  
  Removed.

- **meta_prompt.j2**  
  Template intended for prompt generation dagster asset.  
  Marked unused.
- **rules_meta_prompt.j2**  
  Template intended for generating a prompts for index rules generation.  
  Marked unused.

---

### Tables
- **prompts table**  
  Stored prompt definitions for downstream tasks.  
  Dropped since prompting for prompts is no longer required.

---

### classification.py
- **classification pipeline**  
  Outdated pipeline that used pre‑trained models from the binary folder to classify markets.  
  Removed in favor of lm prompting for labels.

---

### Binary Folder
- Contained:  
  - **initial classifier**: labeled markets as H5N1 or not.  
  - **relevance classifier**: scored markets for relevance.  
  - **pre‑prompting filter**: applied eligibility labels before prompting.  
  Removed.  

- Models in this folder were trained using the following notebooks (all removed):  
  - `market_classification_summary.ipynb`  
    Trained and summarized classification models for market labeling.  
  - `MLP_for_rule_eligibilty.ipynb`  
    Developed a multilayer perceptron for eligibility labeling.  
  - `second_classifier.ipynb`  
    Built a secondary classifier for market relevance.  

---

### meta_prompting.py
- Framework for prompting for prompts.  
- Marked unused.

---
