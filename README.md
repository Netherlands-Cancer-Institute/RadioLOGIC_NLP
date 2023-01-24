# RadioLOGIC
RadioLOGIC: A general model for processing unstructured reports and making decisions in healthcare

(RadioLOGIC, RadioLogical repOmics driven General model Incorporating medical token Cognition).
### Requirements:

* pytorch 1.11.0
* huggingface-hub 0.5.1
* tokenizers 0.12.1
* numpy 1.21.5
* pandas 1.3.5
* scikit-learn 1.0.2


### Pre-training
* Framework
![image](https://github.com/Netherlands-Cancer-Institute/NLP_RadioLOGIC/blob/main/Figure/Pre-training.png)
Note: Framework and results of our natural language processing model for pre-training. (a) Structure of the model for pre-training-based on Bidirectional Encoder Representations from Transformers. (b) Detailed architecture of the transformer encoder. (c) Flow chart of this study. (d) Schematic diagram of the pre-training process.

### Downstream-tasks
* Flowchart
![image](https://github.com/Netherlands-Cancer-Institute/NLP_RadioLOGIC/blob/main/Figure/model.png)
Note: Flowchart and information about target tasks. (A) Flow chart of imaging feature extraction. (B) Architecture of the final model incorporating medical token cognition in this study.
* Results (BI-RADS scores & Pathological outcome)
![image](https://github.com/Netherlands-Cancer-Institute/NLP_RadioLOGIC/blob/main/Figure/Downstream-tasks.png)
Note: Prediction results for downstream tasks. (A) Confusion matrix results for predicting BI-RADS scores in the independent test cohort using RadioLOGIC without transfer learning. (B) Confusion matrix results for predicting BI-RADS scores in the independent test cohort using RadioLOGIC via transfer learning. (C) Receiver operating characteristic curves for predicting pathological outcome in the independent test cohort using RNN. (D) Receiver operating characteristic curves for predicting pathological outcome in the independent test cohort using RadioLOGIC. The 95% confidence intervals are shown as a shaded area for the ROC curve. ATT, Attention mechanism. BI-RADS, Breast Imaging-Reporting and Data System. RadioLOGIC, Radiological repomics driven general model incorporating medical token cognition. RNN, Recurrent neural networks. TF, Transfer learning.


### Visualization
* Word/sentence
![image](https://github.com/Netherlands-Cancer-Institute/NLP_RadioLOGIC/blob/main/Figure/Visualization.png)
Note: Visualizations of words and sentence. (A) Word cloud based on all radiological reports. (B) Visualization of word co-occurrence. (C) Association of Top 50 co-occurrence words. (D) Associations between words in a given report after pre-training. (E) The correlation between the selected word and other words in the report.

* Repomics

In line with the current terminology for extraction of quantitative data out of unstructured data (e.g. radiomics, pathomics etc.), we propose the we propose the term “repomics” (report omics) for extracting valuable features from electronic health records/reports.
![image](https://github.com/Netherlands-Cancer-Institute/NLP_RadioLOGIC/blob/main/Figure/Repomics.png)
Note: Examples of repomics feature extraction from corresponding images and radiological reports. (A) Mammography. (B) Ultrasound. (C) MRI. “***” in the radiological report indicates the patient’s private information.


### Contact details
If you have any questions please contact us. 

Email: ritse.mann@radboudumc.nl; r.mann@nki.nl; taotanjs@gmail.com

Links: [Netherlands Cancer Institute](https://www.nki.nl/) [Radboud university medical center](https://www.radboudumc.nl/en/patient-care)

<img src="https://github.com/Netherlands-Cancer-Institute/Multimodal_attention_DeepLearning/blob/main/Figures/NKI.png" width="253" height="132"/> <img src="https://github.com/Netherlands-Cancer-Institute/Multimodal_attention_DeepLearning/blob/main/Figures/RadboudUMC.png" width="350" height="113"/>
