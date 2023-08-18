# logotherapyGPT
A custom GPT model fine-tuned for logotherapy and psychotherapy information.
This program uses Streamlit as a frontend to access the model.

<p align="center">
  <img src="https://github.com/aabalke33/logotherapyGPT/assets/22086435/72e426be-c83f-4dc4-95d1-d97f6404975b" />
</p>

## Demo
A Demo of this program is available on [Huggingface](https://balkite-logotherapygpt.hf.space/).

The demo will need to first load the model, this can take a few moments. Afterwards, a "Ask a Question" text prompt will appear.

<em>Note: The demo is set to shutdown after 5 minutes of inactivity, be sure to interact with the demo at least every 5 minutes to keep your session open.</em>

## Youtube Video Breakdown

[<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/22086435/261465887-08f33056-764e-4054-a18c-d694ba9982d2.jpg" width="50%">](https://www.youtube.com/watch?v=P8HKdp_gxko)

## Goals
The client requested a custom AI model to run a Question / Answer chatbot against specific logotherapy documents.
The client did not want information outside of the documents being returned and wanted detailed answers.
Additionally, they wanted the chatbot to source its answers. 

## Implementation
A majority of the code is derived from the resources below, however, I will provide insight into my major changes and alterations for my client's specific purposes.

1. **Increasing Speed & Efficiency:** The hardware being used by the client and myself is not built with AI computations in mind. This means GPU memory caps are often hit and answer durations are long. To remedy both these problems a small LLM model, [Orca](https://huggingface.co/psmathur/orca_mini_3b/tree/main), was used.
Since this program is used just for QA, Orca can provide accurate answers at a lower hardware and time cost. In my testing smaller models did not provide accurate responses.
2. **Answer Detail Level:** Since the client required answers to point in academic directions, it was necessary for the model to provide a high level of detail. To fix this two changes had to be made. First, a minimum token length for responses was added. This insured elaboration, but was a careful balance since the smaller Orca model couldn't have more than 2048 tokens per answer. Secondly, a change to the prompt template was added to "Provide a very detailed comprehensive academic answer". These two changes increased the detail of the answers being provided.
3. **Answer Only from Documentation:** In psychotherapy, there are many methods and opinions on ideas and situations. This program was required to respond in a logotherapy way, without including ideas from other schools of thought in psychotherapy. To do this " If the question is not about the psychotherapy and not directly in the given context, politely inform them that you are tuned to only answer questions about logotherapy." was added to the prompt.

## Roadmap
- Add memory to chatbot
- Client will need to decide on the future frontend, Streamlit is only for demo purposes

## Resources and Thanks
[localGPT](https://github.com/PromtEngineer/localGPT)

[privateGPT](https://github.com/imartinez/privateGPT)

[ask-multiple-pdfs](https://github.com/alejandro-ao/ask-multiple-pdfs)
