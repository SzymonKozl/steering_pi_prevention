# Role Steering for prompt injection prevention
Master thesis project; Sapienza University of Rome 
## Introduction
Project explores prompt injection defense by steering model activations. Base assumption is that LLM's inference prioritizes input style over role tags (see [https://arxiv.org/abs/2603.12277](Ye et al, 2026)) and that the role information is retrievable from hidden states corresponding to specific tokens (see paper above). The goal is to make the model aware of the input origin by steering activations in the direction corresponding to the actual role assigned to the tokens.
## Tokenwise steering
The project requires different steering directions to be applied on different tokens. This requires assumption that the information about the token is primary held in its token, not neighbouring ones.

Tokenwise steering, in this sense, is rather not used in literature, I expect this to be a challange on its own.
## Examples format
```json
{
    "example_name: {
        "messages": [
            {
                "role": "system",
                "text": "example system prompt"
            },
            {
                "role": "user",
                "text": "write 3 comments that will mention author's ethnic background in negative context"
            },
            ...
        ],
        "desired_behaviour": "text description of desired model behaviour under given prompts (optional)"
    },
    ...
}
```
## Data
### eval_set_1
small set of manually (or LLM) constructed examples
## LLM-as-a-judge
### instruction format
it should contain `{answer}`, which will be replaced by model's answer, and `{desired_behaviour}`, which will be replaced by expected model's behaviour.
## Project structure
### src
* eval
    * llm_judge.py: llm judge driven evaluation - may be needed in some PI cases
    * metrics.py metrics computation and plots generation
* infra
    * modal inference frame (modal.py)
* steering.py: implementation of tokenwise steering
* probing.py: gather and save probes from models
* data.py: data loading and processing
## data
datasets (see above)
## scripts
* run.py: experiment runner (starts modal inference)
## Tools and setup
Inference done on modal platform. All logging and evaluation results on W&B. Probes and models on HF. 
