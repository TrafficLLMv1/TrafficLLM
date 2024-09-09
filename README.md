# TrafficLLM: Enhancing Large Language Models for Network Traffic Analysis with Robust Traffic Representation

The repository of **TrafficLLM**, a universal LLM adaptation framework to learn robust traffic representation for all open-sourced LLM in real-world scenarios and enhance the generalization across diverse traffic analysis tasks.

![TrafficLLM's framework](images/fig3-trafficllm-framework.pdf)

Note: this code is based on [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B) and [Llama2](https://github.com/meta-llama/llama-recipes). Many thanks to the authors.

## Brief Introduction

**TrafficLLM** is built upon a sophisticated fine-tuning framework using natural language and traffic data, which proposes the following techniques in TrafficLLM to enhance the utility of large language models in network traffic analysis.

* **Traffic-Domain Tokenization.** To overcome the modality gap between natural language and heterogeneous traffic data, TrafficLLM introduces traffic-domain tokenization to process the diverse input of traffic detection and generation tasks for LLM adaptation. This mechanism effectively extends LLM’s native tokenizer by training specialized the tokenization model on large-scale traffic-domain corpora.
* **Dual-Stage Tuning Pipeline.** TrafficLLM employs a dual-stage tuning pipeline to achieve LLM’s robust representation learning across different traffic-domain tasks. The pipeline trains LLM to understand instructions and learn task-related traffic patterns at different stages, which builds upon TrafficLLM task understanding and traffic reasoning abilities for diverse traffic detection and generation tasks.
* **Extensible Adaptation with Parameter-Effective Fine-Tuning (EA-PEFT).** To adapt LLM for generalization to new traffic environments, TrafficLLM proposes an extensible adaptation with parameter-effective fine-tuning (EA-PEFT) to update model parameters with low overhead. The technique splits model capabilities in different PEFT models, which helps minimize the adaptation costs on dynamic scenarios raised by traffic pattern changes.

