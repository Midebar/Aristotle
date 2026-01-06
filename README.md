# Aristotle Framework on Open-Weight SLM with Bahasa Indonesia Dataset (ProntoQA only)

Link to thesis' experiment repository: [Github Link](https://github.com/Midebar/Aristotle)

Original Paper: [**"Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework"**](<https://arxiv.org/abs/2412.16953>)

**Introduction**
-----
In the context of large language models (LLMs), current advanced reasoning methods have made impressive strides in various reasoning tasks. However, when it comes to logical reasoning tasks, significant challenges remain in both efficacy and efficiency. This is rooted in the fact that these systems fail to fully leverage the inherent structure of logical tasks throughout the reasoning processes, including decomposition, search, and resolution. To address this, this paper proposes a logic-complete reasoning framework, Aristotle. The framework consists of three key components: Logical Decomposer, Logical Search Router, and Logical Resolver, in which symbolic expressions and logical rules are comprehensively integrated into the entire reasoning process, significantly alleviating the bottlenecks of logical reasoning, i.e., reducing sub-task complexity, minimizing search errors, and resolving logical contradictions. Experimental results demonstrate that Aristotle consistently outperforms state-of-the-art reasoning frameworks in both accuracy and efficiency, particularly excelling in complex logical reasoning scenarios.

![My Image](aristotle.png)

![Framework](framework.png)


**Running the Experiment**
-----

You can use one of the python notebook (qwen, sealion, sahabatai) to run the pipeline or make similar python notebook on different model