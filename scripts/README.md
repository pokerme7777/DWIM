<h1 align="center">
  🚀 Introduction to the DWIM Pipeline
</h1>

The **DWIM** (Discrepancy-aware Workflow generation & Instruct-Masking) is designed to collect structured reasoning workflows and transform them into training data for instruct-masking fine tuning. Below is a step-by-step overview of how to use it:

---

### 🧩 Step 1: Collecting Workflows via Prompting

Use the prompt file [`prompt-training.xml`](./prompt-training.xml) along with in-context examples from [`In_context_learning_examples.yaml`](./In_context_learning_examples.yaml).  
These inputs are fed to the LLM, which interacts with your custom environment to generate complete reasoning workflows for each training query.

🗂️ Output: `collected_workflow.jsonl`

---

### 🏷️ Step 2: Flagging Effective Actions

Apply a rule-based strategy to identify effective actions within each generated workflow.  
You can design your own rule-based method.

---

### 🪄 Step 3: Instruct-Masking Transformation

For every **effective action**, mask the original action and replace it with an instruction that allows the model to reproduce it.

This process transforms raw workflows into **instruction-masked** format — ready for model training.

🗂️ Output: `IM_data.jsonl`

---

### 🧠 Step 4: Model Training

Train your model using any framework of your choice (e.g., Hugging Face, OpenLLM, DeepSpeed, etc.) on the instruction-masked workflows.

---

### 📌 Summary

| Stage | Output |
|-------|--------|
| Prompt-based workflow generation | `collected_workflow.jsonl` |
| Action effectiveness labeling    | ✅ Flagged effective steps |
| Instruction masking              | `IM_data.jsonl` |
| Training                         | Your preferred library |

---

Feel free to open an issue or reach out if you have any questions or contributions!
