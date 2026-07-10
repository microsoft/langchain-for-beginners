# Assignment: Deploy a LangChain Agent to Microsoft Foundry

## Overview

Practice turning a local LangChain agent into a Microsoft Foundry hosted agent. You'll start from the sample in this chapter, replace the sample retrieval tool with your own Chapter 8 agentic RAG logic, and deploy it to a Foundry project.

## Prerequisites

- Completed this [chapter](./README.md)
- Completed [Building Agentic RAG Systems](../08-agentic-rag-systems/README.md)
- Have access to a Microsoft Foundry project with a deployed chat model
- Have Microsoft Foundry Toolkit installed in VS Code

---

## Challenge: Deploy Your Agentic RAG Agent

**Goal**: Deploy the agentic RAG agent you built in Chapter 8 as a Microsoft Foundry hosted agent.

**Tasks**:

1. Copy or adapt your Chapter 8 retrieval tool into `09-deploy-to-microsoft-foundry/code/main.py`
2. Keep the `create_agent()` pattern and update the system prompt for your knowledge base
3. Configure your Foundry project endpoint and model deployment name
4. Test the hosted wrapper locally at `http://localhost:8088/responses`
5. Deploy the agent with Foundry Toolkit from VS Code
6. Ask at least 3 questions in the Foundry Toolkit playground:
   - One general question the agent can answer directly
   - One document-specific question that should use retrieval
   - One question that is not covered by your documents

**Success Criteria**:

- Agent starts locally without errors
- `/responses` accepts a non-streaming test request
- Deployed hosted agent appears in the Foundry Toolkit explorer
- Agent uses retrieval for knowledge-base questions
- Agent answers gracefully when the knowledge base does not contain an answer
- Deployment instructions and environment variables are documented for your project

**Hints**:

```python
def build_agent():
    return create_agent(
        build_chat_model(),
        tools=[search_my_docs],
        system_prompt=(
            "You are a helpful assistant with access to my knowledge base. "
            "Use search_my_docs for document-specific questions and cite sources."
        ),
    )
```

Do not add an API key to `.env` or the deployment configuration. Keep only non-secret endpoint and model names in `.env`, run `az login` for local testing, and let `DefaultAzureCredential` use the Foundry-provided agent identity after deployment.

> [!TIP]
> **🤖 Get help from [GitHub Copilot](../docs/copilot.md):** Open `main.py` and ask Copilot to help you adapt your Chapter 8 retrieval tool for the hosted-agent sample.

---

## Bonus Challenge: Document Repeatable Deployment

**Goal**: Prepare a repeatable team deployment checklist without changing the ContainerAgent YAML used by Foundry Toolkit.

**Tasks**:

1. Keep `09-deploy-to-microsoft-foundry/code/azure.yaml` in the ContainerAgent schema format
2. Confirm the `code/` folder contains `main.py`, `requirements.txt`, and `azure.yaml`
3. List the Foundry project endpoint and model deployment name your team should select or configure
4. Test locally with `python main.py` and a `/responses` request
5. Capture the Foundry Toolkit deployment choices your team should use for future deployments

**Success Criteria**:

- The existing `code/azure.yaml` remains a ContainerAgent YAML file
- The checklist names the correct project endpoint and model deployment
- Local invocation works before deployment
- Hosted deployment path is documented for the team
- Cleanup plan is clear for any resources created only for this exercise

---

## Submission Checklist

Before completing this chapter:

- [ ] Local hosted wrapper runs successfully
- [ ] Foundry project and model deployment are configured
- [ ] Agent deploys from VS Code with Foundry Toolkit
- [ ] Hosted agent answers test questions in the playground
- [ ] Optional: `azd` deployment path is documented

---

## Need Help?

- **Hosted agent quickstart**: Review the [Microsoft Foundry quickstart](https://learn.microsoft.com/azure/foundry/agents/quickstarts/quickstart-hosted-agent?pivots=vscode)
- **Agentic RAG basics**: Review [Building Agentic RAG Systems](../08-agentic-rag-systems/README.md)
- **Course setup**: Review [Course Setup](../00-course-setup/README.md)
- **Any question**: Use the [Challenge Tutor agent](../docs/copilot.md#challenge-tutor-agent) in GitHub Copilot
- **Still stuck**: Join our [Discord community](https://aka.ms/foundry/discord)